"""
Data utilities for LLaMA-Omni training
Handles loading and preprocessing of InstructS2S-200K dataset for both training stages
"""

import json
import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import librosa
from omni_speech.datasets.preprocess import preprocess, preprocess_multimodal
from omni_speech.constants import DEFAULT_SPEECH_TOKEN, IGNORE_INDEX
import fairseq
from fairseq.models.hubert import HubertModel


class InstructS2SDataset(Dataset):
    """Dataset for InstructS2S-200K with speech instructions and responses"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        data_args,
        stage: int = 1,
        kmeans_model_path: Optional[str] = None
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.stage = stage
        
        # Load dataset
        self.data = self.load_data()
        
        # For stage 2, load HuBERT and K-means for unit extraction
        if stage == 2:
            assert kmeans_model_path is not None, "K-means model path required for stage 2"
            self.load_speech_quantizer(kmeans_model_path)
    
    def load_data(self) -> List[Dict]:
        """Load the InstructS2S-200K dataset in the provided format"""
        if os.path.isfile(self.data_path):
            with open(self.data_path, 'r') as f:
                if self.data_path.endswith('.json'):
                    raw_data = json.load(f)
                else:  # jsonl format
                    raw_data = [json.loads(line) for line in f]
        else:
            # If directory, load all json/jsonl files
            raw_data = []
            for file in os.listdir(self.data_path):
                if file.endswith(('.json', '.jsonl')):
                    file_path = os.path.join(self.data_path, file)
                    with open(file_path, 'r') as f:
                        if file.endswith('.json'):
                            raw_data.extend(json.load(f))
                        else:
                            raw_data.extend([json.loads(line) for line in f])
        
        # Convert to expected format: group human-gpt pairs
        data = []
        for i in range(0, len(raw_data), 2):
            if i + 1 < len(raw_data):
                human_entry = raw_data[i]
                gpt_entry = raw_data[i + 1]
                
                if human_entry['from'] == 'human' and gpt_entry['from'] == 'gpt':
                    data.append({
                        'speech_instruction_path': human_entry['speech'],
                        'text_instruction': human_entry['text'],
                        'text_response': gpt_entry['text'],
                        'speech_response_path': gpt_entry['speech'],
                        'speech_units': gpt_entry.get('unit', None)
                    })
        
        return data
    
    def load_speech_quantizer(self, kmeans_model_path: str):
        """Load HuBERT and K-means model for speech unit extraction"""
        # Load HuBERT model
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            ["/path/to/hubert_base_ls960.pt"]  # Update with actual path
        )
        self.hubert_model = models[0].eval()
        
        # Load K-means model
        import joblib
        self.kmeans_model = joblib.load(kmeans_model_path)
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        if self.data_args.input_type == "mel":
            # Load audio and convert to mel spectrogram
            audio, sr = librosa.load(audio_path, sr=16000)
            mel = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=self.data_args.mel_size,
                n_fft=400,
                hop_length=160
            )
            mel = torch.from_numpy(mel).float()
            return mel.transpose(0, 1)  # (time, mel_dim)
        else:
            # Load raw audio
            audio, sr = torchaudio.load(audio_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)
            return audio.squeeze(0)
    
    def extract_speech_units(self, units_string: str = None, audio_path: str = None) -> List[int]:
        """Extract discrete speech units - either from provided string or using HuBERT + K-means"""
        if units_string:
            # Parse units from string format like "<776><1312><4299>..."
            units = []
            unit_str = units_string.strip()
            if unit_str.startswith('<') and unit_str.endswith('>'):
                # Split by ><
                unit_tokens = unit_str[1:-1].split('><')
                units = [int(token) for token in unit_tokens if token.isdigit()]
            return units
        
        elif audio_path and hasattr(self, 'hubert_model'):
            # Extract using HuBERT + K-means
            audio = self.load_audio(audio_path)
            
            # Extract HuBERT features
            with torch.no_grad():
                features = self.hubert_model.extract_features(audio.unsqueeze(0))[0]
                features = features.squeeze(0).numpy()
            
            # Quantize using K-means
            units = self.kmeans_model.predict(features)
            
            # Remove consecutive duplicates (as mentioned in paper)
            units = [units[0]] + [units[i] for i in range(1, len(units)) if units[i] != units[i-1]]
            
            return units
        
        else:
            return []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Load speech instruction
        speech_path = item['speech_instruction_path']
        speech_features = self.load_audio(speech_path)
        
        # Prepare conversation format
        conversation = [
            {"from": "human", "value": DEFAULT_SPEECH_TOKEN + "\n" + item.get('text_instruction', '')},
            {"from": "gpt", "value": item['text_response']}
        ]
        
        # Preprocess conversation
        sources = preprocess_multimodal([conversation], self.data_args)
        data_dict = preprocess(sources, self.tokenizer, has_speech=True)
        
        result = {
            'input_ids': data_dict['input_ids'][0],
            'labels': data_dict['labels'][0],
            'speech_features': speech_features,
        }
        
        # For stage 2, also include speech response units
        if self.stage == 2:
            # Try to get units from the dataset first, then fallback to extraction
            speech_units = []
            if 'speech_units' in item and item['speech_units']:
                speech_units = self.extract_speech_units(units_string=item['speech_units'])
            elif 'speech_response_path' in item and item['speech_response_path']:
                speech_units = self.extract_speech_units(audio_path=item['speech_response_path'])
            
            if speech_units:
                result['speech_units'] = torch.tensor(speech_units, dtype=torch.long)
            else:
                # If no speech units available, skip this item
                return None
        
        return result


class DataCollator:
    """Data collator for batching InstructS2S data"""
    
    def __init__(self, tokenizer, stage: int = 1):
        self.tokenizer = tokenizer
        self.stage = stage
    
    def __call__(self, instances: List[Dict]) -> Dict:
        # Filter out None instances
        instances = [inst for inst in instances if inst is not None]
        
        if len(instances) == 0:
            return {}
        
        # Pad input_ids and labels
        input_ids = [inst['input_ids'] for inst in instances]
        labels = [inst['labels'] for inst in instances]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        # Pad speech features
        speech_features = [inst['speech_features'] for inst in instances]
        max_speech_len = max(feat.size(0) for feat in speech_features)
        
        padded_speech = []
        speech_lengths = []
        for feat in speech_features:
            speech_lengths.append(feat.size(0))
            if feat.size(0) < max_speech_len:
                padding = torch.zeros(max_speech_len - feat.size(0), feat.size(1))
                feat = torch.cat([feat, padding], dim=0)
            padded_speech.append(feat)
        
        result = {
            'input_ids': input_ids,
            'labels': labels,
            'speech_features': torch.stack(padded_speech),
            'speech_lengths': torch.tensor(speech_lengths),
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id)
        }
        
        # For stage 2, also include speech units
        if self.stage == 2:
            if 'speech_units' in instances[0]:
                speech_units = [inst['speech_units'] for inst in instances]
                max_unit_len = max(units.size(0) for units in speech_units)
                
                padded_units = []
                unit_lengths = []
                for units in speech_units:
                    unit_lengths.append(units.size(0))
                    if units.size(0) < max_unit_len:
                        padding = torch.full((max_unit_len - units.size(0),), -1, dtype=torch.long)
                        units = torch.cat([units, padding], dim=0)
                    padded_units.append(units)
                
                result['speech_units'] = torch.stack(padded_units)
                result['unit_lengths'] = torch.tensor(unit_lengths)
        
        return result


def create_data_loader(
    data_path: str,
    tokenizer,
    data_args,
    batch_size: int,
    stage: int = 1,
    kmeans_model_path: Optional[str] = None,
    num_workers: int = 4,
    shuffle: bool = True
):
    """Create data loader for training"""
    dataset = InstructS2SDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        stage=stage,
        kmeans_model_path=kmeans_model_path
    )
    
    collator = DataCollator(tokenizer, stage=stage)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )