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
        self.data = self.load_data()
        if stage == 2:
            assert kmeans_model_path is not None, "K-means model path required for stage 2"
            self.load_speech_quantizer(kmeans_model_path)
    
    def load_data(self) -> List[Dict]:
        if os.path.isfile(self.data_path):
            with open(self.data_path, 'r') as f:
                if self.data_path.endswith('.json'):
                    raw_data = json.load(f)
                else:
                    raw_data = [json.loads(line) for line in f]
        else:
            raw_data = []
            for file in os.listdir(self.data_path):
                if file.endswith(('.json', '.jsonl')):
                    file_path = os.path.join(self.data_path, file)
                    with open(file_path, 'r') as f:
                        if file.endswith('.json'):
                            raw_data.extend(json.load(f))
                        else:
                            raw_data.extend([json.loads(line) for line in f])
        data = []
        
        if isinstance(raw_data, list) and len(raw_data) > 0:
            if 'conversation' in raw_data[0]:
                for item in raw_data:
                    conversation = item['conversation']
                    for i in range(0, len(conversation), 2):
                        if i + 1 < len(conversation):
                            human_entry = conversation[i]
                            gpt_entry = conversation[i + 1]
                            
                            if human_entry['from'] == 'human' and gpt_entry['from'] == 'gpt':
                                speech_instruction_path = human_entry['speech'].replace('data/multiturn_instruction/en/', '')
                                speech_response_path = gpt_entry['speech'].replace('data/multiturn_instruction/en/', '')
                                if (os.path.exists(os.path.join(os.path.dirname(self.data_path), speech_instruction_path)) and
                                    os.path.exists(os.path.join(os.path.dirname(self.data_path), speech_response_path))):
                                    data.append({
                                        'speech_instruction_path': speech_instruction_path,
                                        'text_instruction': human_entry['text'],
                                        'text_response': gpt_entry['text'],
                                        'speech_response_path': speech_response_path,
                                        'speech_units': gpt_entry.get('unit', None)
                                    })
            else:
                for i in range(0, len(raw_data), 2):
                    if i + 1 < len(raw_data):
                        human_entry = raw_data[i]
                        gpt_entry = raw_data[i + 1]
                        
                        if human_entry['from'] == 'human' and gpt_entry['from'] == 'gpt':
                            speech_instruction_path = human_entry['speech'].replace('data/multiturn_instruction/en/', '')
                            speech_response_path = gpt_entry['speech'].replace('data/multiturn_instruction/en/', '')
                            if (os.path.exists(os.path.join(os.path.dirname(self.data_path), speech_instruction_path)) and
                                os.path.exists(os.path.join(os.path.dirname(self.data_path), speech_response_path))):
                                data.append({
                                    'speech_instruction_path': speech_instruction_path,
                                    'text_instruction': human_entry['text'],
                                    'text_response': gpt_entry['text'],
                                    'speech_response_path': speech_response_path,
                                    'speech_units': gpt_entry.get('unit', None)
                                })
        
        print(f"Loaded {len(data)} valid samples with existing audio files")
        return data
    
    def load_speech_quantizer(self, kmeans_model_path: str):
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            ["../hubert_base_ls960.pt"]
        )
        self.hubert_model = models[0].eval()
        
        import joblib
        self.kmeans_model = joblib.load(kmeans_model_path)
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        if self.data_args.input_type == "mel":
            audio, sr = librosa.load(audio_path, sr=16000)
            mel = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=self.data_args.mel_size,
                n_fft=1024,  # Whisper uses 1024
                hop_length=160,  # 10ms hop length
                fmin=0,
                fmax=8000
            )
            # Convert to log scale (dB)
            mel = librosa.power_to_db(mel, ref=np.max)
            # Normalize to [-1, 1] range as expected by Whisper
            mel = np.clip((mel + 80) / 80, -1, 1)
            mel = torch.from_numpy(mel).float()
            mel_transposed = mel.transpose(0, 1)  # (time, mel_dim)
            
            # Whisper expects fixed length sequences (3000 frames for 30-second audio)
            target_length = 3000
            current_length = mel_transposed.shape[0]
            
            if current_length > target_length:
                # Truncate if too long
                mel_transposed = mel_transposed[:target_length]
            elif current_length < target_length:
                # Pad if too short
                padding = torch.zeros(target_length - current_length, mel_transposed.shape[1])
                mel_transposed = torch.cat([mel_transposed, padding], dim=0)
            
            if torch.isnan(mel_transposed).any() or torch.isinf(mel_transposed).any():
                mel_transposed = torch.zeros_like(mel_transposed)
            
            return mel_transposed
        else:
            audio, sr = torchaudio.load(audio_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)
            return audio.squeeze(0)
    
    def extract_speech_units(self, units_string: str = None, audio_path: str = None) -> List[int]:
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
            audio = self.load_audio(audio_path)
            with torch.no_grad():
                features = self.hubert_model.extract_features(audio.unsqueeze(0))[0]
                features = features.squeeze(0).numpy()
            units = self.kmeans_model.predict(features)
            units = [units[0]] + [units[i] for i in range(1, len(units)) if units[i] != units[i-1]]
            
            return units
        
        else:
            return []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        speech_path = os.path.join(os.path.dirname(self.data_path), item['speech_instruction_path'])
        speech_features = self.load_audio(speech_path)
        conversation = [
            {"from": "human", "value": DEFAULT_SPEECH_TOKEN + "\n" + item.get('text_instruction', '')},
            {"from": "gpt", "value": item['text_response']}
        ]
        sources = preprocess_multimodal([conversation], self.data_args)
        data_dict = preprocess(sources, self.tokenizer, has_speech=True)
        
        result = {
            'input_ids': data_dict['input_ids'][0],
            'labels': data_dict['labels'][0],
            'speech_features': speech_features,
        }
        if self.stage == 2:
            speech_units = []
            if 'speech_units' in item and item['speech_units']:
                speech_units = self.extract_speech_units(units_string=item['speech_units'])
            elif 'speech_response_path' in item and item['speech_response_path']:
                response_audio_path = os.path.join(os.path.dirname(self.data_path), item['speech_response_path'])
                speech_units = self.extract_speech_units(audio_path=response_audio_path)
            
            if speech_units:
                result['speech_units'] = torch.tensor(speech_units, dtype=torch.long)
            else:
                return None
        
        return result


class DataCollator:
    def __init__(self, tokenizer, stage: int = 1):
        self.tokenizer = tokenizer
        self.stage = stage
    
    def __call__(self, instances: List[Dict]) -> Dict:
        instances = [inst for inst in instances if inst is not None]
        
        if len(instances) == 0:
            return {}
        input_ids = [inst['input_ids'] for inst in instances]
        labels = [inst['labels'] for inst in instances]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
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
            'attention_mask': input_ids.ne(pad_token_id)
        }
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