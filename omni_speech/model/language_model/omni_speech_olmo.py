# OLMo model integration for OmniSpeech
# Based on the LLaMA implementation but uses AutoModelForCausalLM for OLMo models

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

# Patch timm import issue
import sys
class MockImageNetInfo:
    pass

# Mock the problematic import
try:
    from timm.data import ImageNetInfo
except ImportError:
    import timm.data
    timm.data.ImageNetInfo = MockImageNetInfo
    timm.data.infer_imagenet_subset = lambda x: None
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..omni_speech_arch import OmniSpeechMetaModel, OmniSpeechMetaForCausalLM


class OmniSpeechOlmoConfig(PretrainedConfig):
    model_type = "omni_speech_olmo"
    
    def __init__(self, model_name=None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name


class OmniSpeechOlmoModel(nn.Module):
    config_class = OmniSpeechOlmoConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize speech components manually (implementing OmniSpeechMetaModel functionality)
        if hasattr(config, "speech_encoder"):
            from ..speech_encoder.builder import build_speech_encoder
            from ..speech_projector.builder import build_speech_projector
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)
        
        # Initialize embedding tokens - these will be set by the parent class
        self.embed_tokens = None

    def get_speech_encoder(self):
        speech_encoder = getattr(self, 'speech_encoder', None)
        if type(speech_encoder) is list:
            speech_encoder = speech_encoder[0]
        return speech_encoder

    def initialize_speech_modules(self, model_args, fsdp=None):
        self.config.speech_encoder = getattr(model_args, "speech_encoder", None)
        self.config.speech_encoder_type = getattr(model_args, "speech_encoder_type", None)
        self.config.speech_projector_type = getattr(model_args, 'speech_projector_type', 'linear')
        self.config.speech_encoder_ds_rate = getattr(model_args, 'speech_encoder_ds_rate', 5)
        self.config.speech_encoder_hidden_size = getattr(model_args, 'speech_encoder_hidden_size', 1280)

        if self.get_speech_encoder() is None:
            from ..speech_encoder.builder import build_speech_encoder
            speech_encoder = build_speech_encoder(self.config)
            if fsdp is not None and len(fsdp) > 0:
                self.speech_encoder = [speech_encoder]
            else:
                self.speech_encoder = speech_encoder

        if getattr(self, 'speech_projector', None) is None:
            from ..speech_projector.builder import build_speech_projector
            self.speech_projector = build_speech_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.speech_projector.parameters():
                p.requires_grad = True

        if hasattr(model_args, 'pretrain_speech_projector') and model_args.pretrain_speech_projector is not None:
            pretrain_speech_projector_weights = torch.load(model_args.pretrain_speech_projector, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.speech_projector.load_state_dict(get_w(pretrain_speech_projector_weights, 'speech_projector'))


class OmniSpeechOlmoForCausalLM(nn.Module, OmniSpeechMetaForCausalLM):
    config_class = OmniSpeechOlmoConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize base model and components
        if hasattr(config, 'model_name') and config.model_name:
            # Load OLMo model more specifically to avoid auto-loading issues
            try:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            except ImportError as e:
                if "ImageNetInfo" in str(e):
                    # Fallback: try without auto-loading problematic models
                    import os
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    try:
                        self.base_model = AutoModelForCausalLM.from_pretrained(
                            config.model_name,
                            trust_remote_code=True,
                            torch_dtype=torch.float16,
                            local_files_only=False
                        )
                    finally:
                        os.environ.pop("TRANSFORMERS_OFFLINE", None)
                else:
                    raise e
            self.vocab_size = self.base_model.config.vocab_size
            self.lm_head = self.base_model.lm_head
            
            # Set up the OmniSpeech model wrapper with the base model's config
            config.vocab_size = self.vocab_size
            config.hidden_size = self.base_model.config.hidden_size
            self.model = OmniSpeechOlmoModel(config)
            
            # Copy embed_tokens from base model
            if hasattr(self.base_model, 'get_input_embeddings'):
                self.model.embed_tokens = self.base_model.get_input_embeddings()
            elif hasattr(self.base_model, 'embed_tokens'):
                self.model.embed_tokens = self.base_model.embed_tokens
            else:
                # Fallback: look for embeddings in the model structure
                for name, module in self.base_model.named_modules():
                    if 'embed' in name.lower() and hasattr(module, 'weight'):
                        self.model.embed_tokens = module
                        break
        else:
            self.base_model = None
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.model = OmniSpeechOlmoModel(config)

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        pass

    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, config=None, **kwargs):
        """Load a pretrained OLMo model and wrap it with OmniSpeech functionality."""
        if config is None:
            base_config = AutoConfig.from_pretrained(model_name_or_path)
        else:
            base_config = config
        
        # Convert to OmniSpeechOlmoConfig
        if not isinstance(base_config, OmniSpeechOlmoConfig):
            # Create OmniSpeechOlmoConfig with base config attributes
            config_dict = base_config.to_dict() if hasattr(base_config, 'to_dict') else base_config.__dict__.copy()
            # Remove model_name from config_dict to avoid conflict
            config_dict.pop('model_name', None)
            olmo_config = OmniSpeechOlmoConfig(model_name=model_name_or_path, **config_dict)
            config = olmo_config
        else:
            config.model_name = model_name_or_path
            
        return cls(config)

    def get_model(self):
        return self.model

    @property
    def device(self):
        """Return the device of the model parameters"""
        if self.base_model is not None:
            return next(self.base_model.parameters()).device
        else:
            return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_speech_and_text(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech,
                speech_lengths
            )

        # Forward through the base OLMo model if available, otherwise use parent implementation
        if self.base_model is not None:
            return self.base_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            # Fallback to parent class implementation
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if speech is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_speech_and_text(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if self.base_model is not None:
            return self.base_model.generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        else:
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )

AutoConfig.register("omni_speech_olmo", OmniSpeechOlmoConfig)
AutoModelForCausalLM.register(OmniSpeechOlmoConfig, OmniSpeechOlmoForCausalLM)