# Adopted from https://github.com/haotian-liu/LLaVA. We modify the code to support speech input. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from omni_speech.model import *
from omni_speech.model.speech_encoder.builder import build_speech_encoder
from omni_speech.conversation import set_default_conversation


def is_olmo_model(model_path):
    """Check if the model is an OLMo model"""
    try:
        config = AutoConfig.from_pretrained(model_path)
        # Check for OLMo model identifiers
        if hasattr(config, 'model_type'):
            return 'olmo' in config.model_type.lower()
        # Check model name/path for OLMo indicators
        if 'olmo' in model_path.lower():
            return True
        return False
    except:
        return False


def load_pretrained_model(model_path, model_base, is_lora=False, s2s=False, load_8bit=False, load_4bit=False, device="cuda", use_flash_attn=False, model_args=None, **kwargs):
    # Set conversation template based on model_args version if provided
    if model_args and hasattr(model_args, 'version') and model_args.version:
        if model_args.version in ["olmo"]:
            set_default_conversation(model_args.version)
        elif model_args.version.startswith("v1"):
            set_default_conversation("v1")
        else:
            # Keep default for other versions
            pass
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    
    # Determine model class based on the base model type
    if model_base and is_olmo_model(model_base):
        model_cls = OmniSpeechOlmoForCausalLM
        set_default_conversation("olmo")
    elif model_path and is_olmo_model(model_path):
        model_cls = OmniSpeechOlmoForCausalLM
        set_default_conversation("olmo")
    else:
        # model_cls = OmniSpeech2SLlamaForCausalLM if s2s else OmniSpeechLlamaForCausalLM
        model_cls = OmniSpeechLlamaForCausalLM

    # Load OmniSpeech model
    if is_lora:
        assert model_base is not None, "model_base is required for LoRA models."
        if model_cls == OmniSpeechOlmoForCausalLM:
            from omni_speech.model.language_model.omni_speech_olmo import OmniSpeechOlmoConfig
            lora_cfg_pretrained = OmniSpeechOlmoConfig.from_pretrained(model_path)
            lora_cfg_pretrained.model_name = model_base
        else:
            from omni_speech.model.language_model.omni_speech_llama import OmniSpeechConfig
            lora_cfg_pretrained = OmniSpeechConfig.from_pretrained(model_path)
        
        if model_cls == OmniSpeechOlmoForCausalLM:
            # Use the base OLMo model tokenizer, not the JSON file
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        print('Loading OmniSpeech from base model...')
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=False, config=lora_cfg_pretrained, **kwargs)
        print('Loading additional OmniSpeech weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:
        print('Loading OmniSpeech from base model...')
        if model_cls == OmniSpeechOlmoForCausalLM:
            # Use the base OLMo model tokenizer, not the JSON file
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if model_cls == OmniSpeechOlmoForCausalLM:
            cfg_pretrained.model_name = model_base
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=False, config=cfg_pretrained, **kwargs)
        
        speech_projector_weights = torch.load(os.path.join(model_path, 'speech_projector.bin'), map_location='cpu')
        speech_projector_weights = {k: v.to(torch.float16) for k, v in speech_projector_weights.items()}
        model.load_state_dict(speech_projector_weights, strict=False)
        model = model.to(device=device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if model_cls == OmniSpeechOlmoForCausalLM:
            # For OLMo models, set the model_name to the model_path for loading
            cfg_pretrained.model_name = model_path
        model = model_cls.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            config=cfg_pretrained,
            **kwargs
        )
        model = model.to(device=device)

    # Add speech encoder configuration to model config if not present
    if model_args:
        if hasattr(model_args, 'speech_encoder') and model_args.speech_encoder:
            model.config.speech_encoder = model_args.speech_encoder
        elif not hasattr(model.config, 'speech_encoder'):
            model.config.speech_encoder = "openai/whisper-large-v3"
            
        if hasattr(model_args, 'speech_encoder_type') and model_args.speech_encoder_type:
            model.config.speech_encoder_type = model_args.speech_encoder_type
        elif not hasattr(model.config, 'speech_encoder_type'):
            model.config.speech_encoder_type = "whisper"
    else:
        if not hasattr(model.config, 'speech_encoder'):
            model.config.speech_encoder = "openai/whisper-large-v3"
        if not hasattr(model.config, 'speech_encoder_type'):
            model.config.speech_encoder_type = "whisper"
    
    model.get_model().speech_encoder = build_speech_encoder(model.config)
    model.get_model().speech_encoder.to(device=device, dtype=torch.float16)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
