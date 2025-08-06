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

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from olmo_omni.conversation import set_default_conversation
from olmo_omni.model import *
from olmo_omni.model.language_model.omni2_speech2s_olmo2 import Omni2Speech2SConfig
from olmo_omni.model.language_model.omni2_speech_olmo2 import Omni2SpeechOlmo2Config
from olmo_omni.model.speech_encoder.builder import build_speech_encoder


def load_pretrained_model(
    model_path,
    model_base,
    is_lora=False,
    s2s=False,
    load_8bit=False,
    load_4bit=False,
    device="cuda",
    use_flash_attn=False,
    model_args=None,
    **kwargs,
):
    set_default_conversation("olmo2")
    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float32

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    model_cls = Omni2Speech2SOlmo2ForCausalLM if s2s else Omni2SpeechOlmo2ForCausalLM

    if is_lora:
        assert model_base is not None, "model_base is required for LoRA models."
        if model_cls == Omni2SpeechOlmo2ForCausalLM:
            lora_cfg_pretrained = Omni2SpeechOlmo2Config.from_pretrained(model_path)
            lora_cfg_pretrained.model_name = model_base
        else:
            raise ValueError("LoRA is not supported for Omni2Speech2SOlmo2ForCausalLM.")

        if model_cls in [Omni2SpeechOlmo2ForCausalLM]:
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

        model = model_cls.from_pretrained(
            model_base, low_cpu_mem_usage=False, config=lora_cfg_pretrained, **kwargs
        )

        if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
            non_lora_trainables = torch.load(
                os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu"
            )
        non_lora_trainables = {
            (k[11:] if k.startswith("base_model.") else k): v
            for k, v in non_lora_trainables.items()
        }
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith("model.") else k): v
                for k, v in non_lora_trainables.items()
            }
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel

        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, model_path)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
        print("Model is loaded...")

    elif model_base is not None:
        print("Loading OmniSpeech from base model...")
        if model_cls in [Omni2SpeechOlmo2ForCausalLM]:
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)

        if model_cls in [Omni2SpeechOlmo2ForCausalLM]:
            cfg_pretrained.model_name = model_base
        model = model_cls.from_pretrained(
            model_base, low_cpu_mem_usage=False, config=cfg_pretrained, **kwargs
        )

        speech_projector_weights = torch.load(
            os.path.join(model_path, "speech_projector.bin"), map_location="cpu"
        )
        speech_projector_weights = {
            k: v.to(torch.float16) for k, v in speech_projector_weights.items()
        }
        model.load_state_dict(speech_projector_weights, strict=False)
        model = model.to(device=device)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if model_cls in [Omni2SpeechOlmo2ForCausalLM]:
            cfg_pretrained.model_name = model_path
        model = model_cls.from_pretrained(
            model_path, low_cpu_mem_usage=False, config=cfg_pretrained, **kwargs
        )
        model = model.to(device=device)

    if model_args:
        if hasattr(model_args, 'speech_encoder') and model_args.speech_encoder:
            model.config.speech_encoder = model_args.speech_encoder
        elif not hasattr(model.config, 'speech_encoder'):
            model.config.speech_encoder = "large-v3"

        if hasattr(model_args, 'speech_encoder_type') and model_args.speech_encoder_type:
            model.config.speech_encoder_type = model_args.speech_encoder_type
        elif not hasattr(model.config, 'speech_encoder_type'):
            model.config.speech_encoder_type = "whisper"
            
        if hasattr(model_args, 'speech_encoder_hidden_size') and model_args.speech_encoder_hidden_size:
            model.config.speech_encoder_hidden_size = model_args.speech_encoder_hidden_size
        elif not hasattr(model.config, 'speech_encoder_hidden_size'):
            model.config.speech_encoder_hidden_size = 1280
            
        if hasattr(model_args, 'speech_projector_type') and model_args.speech_projector_type:
            model.config.speech_projector_type = model_args.speech_projector_type
        elif not hasattr(model.config, 'speech_projector_type'):
            model.config.speech_projector_type = "linear"
            
        if hasattr(model_args, 'speech_encoder_ds_rate') and model_args.speech_encoder_ds_rate:
            model.config.speech_encoder_ds_rate = model_args.speech_encoder_ds_rate
        elif not hasattr(model.config, 'speech_encoder_ds_rate'):
            model.config.speech_encoder_ds_rate = 5
    else:
        if not hasattr(model.config, 'speech_encoder'):
            model.config.speech_encoder = "large-v3"
        if not hasattr(model.config, 'speech_encoder_type'):
            model.config.speech_encoder_type = "whisper"
        if not hasattr(model.config, 'speech_encoder_hidden_size'):
            model.config.speech_encoder_hidden_size = 1280
        if not hasattr(model.config, 'speech_projector_type'):
            model.config.speech_projector_type = "linear"
        if not hasattr(model.config, 'speech_encoder_ds_rate'):
            model.config.speech_encoder_ds_rate = 5

    model.get_model().speech_encoder = build_speech_encoder(model.config)
    # Detect the model's dtype and use it for speech components
    model_dtype = next(model.parameters()).dtype
    model.get_model().speech_encoder.to(device=device, dtype=model_dtype)
    
    if not hasattr(model.get_model(), 'speech_projector') or model.get_model().speech_projector is None:
        from .speech_projector.builder import build_speech_projector
        model.get_model().speech_projector = build_speech_projector(model.config)
        model.get_model().speech_projector.to(device=device, dtype=model_dtype)

    if hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    else:
        context_len = 2048

    return tokenizer, model, context_len
