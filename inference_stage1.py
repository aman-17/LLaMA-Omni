import os
import sys
import torch
import json
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from omni_speech.model.builder import load_pretrained_model
from omni_speech.constants import DEFAULT_SPEECH_TOKEN
from omni_speech.utils import disable_torch_init
from omni_speech.conversation import conv_templates
from omni_speech.datasets.preprocess import tokenizer_speech_token
import whisper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Stage1Inferencer:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.load_model()
        
    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        device = torch.device(device)
        logger.info(f"Using device: {device}")
        return device
    
    def load_model(self):
        logger.info(f"Loading model from {self.model_path}")
        
        # Disable torch init for faster loading (following official script)
        disable_torch_init()
        
        # Load training config from parent directory to get model_base
        training_config_path = os.path.join(os.path.dirname(self.model_path), "training_config.json")
        model_base = None
        if os.path.exists(training_config_path):
            with open(training_config_path, 'r') as f:
                training_config = json.load(f)
            
            # Extract model base from training config
            model_args_dict = training_config.get('model_args', {})
            model_base = model_args_dict.get('model_name_or_path', 'allenai/OLMo-2-0425-1B-Instruct')
            
            logger.info(f"Loaded training config - model base: {model_base}")
        else:
            # Fallback defaults
            model_base = 'allenai/OLMo-2-0425-1B-Instruct'
            logger.warning(f"No training config found, using default model base: {model_base}")
        
        # Load model using official pattern (same as omni_speech/infer/infer.py)
        try:
            self.tokenizer, self.model, self.context_len = load_pretrained_model(
                model_path=self.model_path,
                model_base=model_base,
                is_lora=False,
                s2s=False
            )
            
            self.model.eval()
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_audio(self, audio_path: str, input_type: str = "mel", mel_size: int = 80) -> torch.Tensor:
        """
        Preprocess audio file following official inference pattern (with librosa fallback)
        
        Args:
            audio_path: Path to audio file
            input_type: Type of input processing ("mel" or "raw")
            mel_size: Number of mel frequency bins
            
        Returns:
            Preprocessed audio tensor
        """
        logger.info(f"Processing audio: {audio_path}")
        
        # Load audio using librosa (fallback when ffmpeg not available)
        try:
            speech, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {e}")
            raise
        
        if input_type == "raw":
            speech = torch.from_numpy(speech)
            if hasattr(self.model.config, 'speech_normalize') and self.model.config.speech_normalize:
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
        elif input_type == "mel":
            # Pad or trim to 30 seconds (like whisper.pad_or_trim)
            target_length = 16000 * 30  # 30 seconds at 16kHz
            if len(speech) > target_length:
                speech = speech[:target_length]
            elif len(speech) < target_length:
                speech = np.pad(speech, (0, target_length - len(speech)))
            
            # Create mel spectrogram matching training parameters exactly
            mel_spec = librosa.feature.melspectrogram(
                y=speech, 
                sr=16000, 
                n_mels=mel_size,
                n_fft=1024,  # Match training
                hop_length=160,  # Match training
                fmin=0,
                fmax=8000
            )
            
            # Convert to log scale (dB) - match training
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [-1, 1] range as expected by Whisper - match training
            log_mel = np.clip((log_mel + 80) / 80, -1, 1)
            
            # Convert to tensor and transpose to (time, mel_bins)
            mel_tensor = torch.from_numpy(log_mel).float()
            mel_transposed = mel_tensor.transpose(0, 1)  # (time, mel_dim)
            
            # IMPORTANT: Apply exact same fixed-length padding as training (3000 frames)
            target_frames = 3000  # Exact same as training
            current_frames = mel_transposed.shape[0]
            
            if current_frames > target_frames:
                # Truncate if too long
                speech = mel_transposed[:target_frames]
            elif current_frames < target_frames:
                # Pad if too short
                padding = torch.zeros(target_frames - current_frames, mel_transposed.shape[1])
                speech = torch.cat([mel_transposed, padding], dim=0)
            else:
                speech = mel_transposed
            
            # Check for NaN/Inf values
            if torch.isnan(speech).any() or torch.isinf(speech).any():
                speech = torch.zeros_like(speech)
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")
        
        return speech
    
    def inference(self, audio_path: str, question: str = "Transcribe the speech:", 
                  conv_mode: str = "olmo", input_type: str = "mel", mel_size: int = 80, 
                  max_new_tokens: int = 256, temperature: float = 0, top_p: float = None, 
                  num_beams: int = 1) -> str:
        """
        Perform speech-to-text inference following official pattern
        
        Args:
            audio_path: Path to input audio file
            question: Question to ask about the speech
            conv_mode: Conversation template mode
            input_type: Audio input type ("mel" or "raw")
            mel_size: Number of mel frequency bins
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_beams: Number of beams for beam search
            
        Returns:
            Generated text
        """
        # Preprocess audio following official pattern
        speech_tensor = self.preprocess_audio(audio_path, input_type, mel_size)
        speech_tensor = speech_tensor.unsqueeze(0).to(dtype=torch.float16, device=self.device)
        speech_length = torch.tensor([speech_tensor.shape[1]], dtype=torch.long, device=self.device)
        
        # Create conversation following official pattern
        qs = f"<speech>\n{question}"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize using official tokenizer_speech_token function
        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to(device=self.device)
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Audio features shape: {speech_tensor.shape}")
        
        # Generate response following official pattern
        with torch.inference_mode():
            try:
                outputs = self.model.generate(
                    input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=128004,
                )
                
                # Decode output following official pattern
                output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                
                return output_text
                
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                raise
    
    def inference_batch(self, audio_paths: list, max_new_tokens: int = 128) -> list:
        """
        Perform batch inference on multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            List of generated texts
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.inference(audio_path, max_new_tokens)
                results.append(result)
                logger.info(f"✓ {audio_path}: {result}")
            except Exception as e:
                logger.error(f"✗ {audio_path}: Failed - {e}")
                results.append(f"ERROR: {str(e)}")
        
        return results


def load_validation_samples(val_data_path: str, num_samples: int = 5) -> list:
    """Load a few samples from validation dataset"""
    if not os.path.exists(val_data_path):
        logger.error(f"Validation data not found: {val_data_path}")
        return []
    
    with open(val_data_path, 'r') as f:
        val_data = json.load(f)
    
    # Get first few samples that have audio files
    samples = []
    data_root = os.path.dirname(val_data_path)  # Should be InstructS2S-200K directory
    
    for item in val_data[:num_samples * 5]:  # Check more in case some files are missing
        if 'conversation' in item:
            # Look through conversation turns for speech data
            for turn in item['conversation']:
                if 'speech' in turn and 'text' in turn and turn['from'] == 'human':
                    # Construct full audio path
                    relative_audio_path = turn['speech']
                    full_audio_path = os.path.join(data_root, relative_audio_path)
                    
                    if os.path.exists(full_audio_path):
                        sample = {
                            'audio': full_audio_path,
                            'text': turn['text'],
                            'id': item.get('id', 'unknown')
                        }
                        samples.append(sample)
                        logger.info(f"Found valid sample: {full_audio_path}")
                        break  # Only take first human turn per conversation
        
        if len(samples) >= num_samples:
            break
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Speech-to-Text Inference")
    parser.add_argument("--model_path", type=str, 
                       default="./outputs/stage1/best_model",
                       help="Path to trained model checkpoint")
    parser.add_argument("--audio_file", type=str, default=None,
                       help="Single audio file to transcribe")
    parser.add_argument("--val_data", type=str,
                       default="./InstructS2S-200K/instruct_en_val_small.json",
                       help="Validation dataset JSON file")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of validation samples to test")
    parser.add_argument("--max_tokens", type=int, default=256,
                       help="Maximum tokens to generate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--conv_mode", type=str, default="olmo",
                       help="Conversation template mode")
    parser.add_argument("--input_type", type=str, default="mel",
                       help="Audio input type (mel/raw)")
    parser.add_argument("--mel_size", type=int, default=80,
                       help="Number of mel frequency bins")
    parser.add_argument("--question", type=str, default="Transcribe the speech:",
                       help="Question to ask about the speech")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model checkpoint not found: {args.model_path}")
        return
    
    # Initialize inferencer
    try:
        inferencer = Stage1Inferencer(args.model_path, args.device)
    except Exception as e:
        logger.error(f"Failed to initialize inferencer: {e}")
        return
    
    # Single file inference
    if args.audio_file:
        if not os.path.exists(args.audio_file):
            logger.error(f"Audio file not found: {args.audio_file}")
            return
        
        logger.info(f"Transcribing: {args.audio_file}")
        try:
            result = inferencer.inference(
                args.audio_file, 
                question=args.question,
                conv_mode=args.conv_mode,
                input_type=args.input_type,
                mel_size=args.mel_size,
                max_new_tokens=args.max_tokens
            )
            print(f"\n🎧 Audio: {args.audio_file}")
            print(f"📝 Transcription: {result}\n")
        except Exception as e:
            logger.error(f"Inference failed: {e}")
        return
    
    # Validation set inference
    logger.info(f"Loading validation samples from: {args.val_data}")
    samples = load_validation_samples(args.val_data, args.num_samples)
    
    if not samples:
        logger.error("No valid audio samples found in validation set")
        return
    
    print(f"\n🚀 Running inference on {len(samples)} validation samples...\n")
    
    for i, sample in enumerate(samples, 1):
        audio_path = sample['audio']
        expected_text = sample['text']
        sample_id = sample['id']
        
        print(f"{'='*80}")
        print(f"Sample {i}/{len(samples)} (ID: {sample_id})")
        print(f"🎧 Audio: {audio_path}")
        print(f"📋 Expected: {expected_text}")
        
        try:
            result = inferencer.inference(
                audio_path,
                question=args.question,
                conv_mode=args.conv_mode,
                input_type=args.input_type,
                mel_size=args.mel_size,
                max_new_tokens=args.max_tokens
            )
            print(f"📝 Generated: {result}")
            print(f"✅ Status: Success")
        except Exception as e:
            print(f"❌ Status: Failed - {e}")
        
        print()


if __name__ == "__main__":
    main()