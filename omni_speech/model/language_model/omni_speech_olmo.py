# OLMo model integration for OmniSpeech
# Based on the LLaMA implementation but uses AutoModelForCausalLM for OLMo models

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..omni_speech_arch import OmniSpeechMetaModel, OmniSpeechMetaForCausalLM


class OmniSpeechOlmoConfig(AutoConfig):
    model_type = "omni_speech_olmo"


class OmniSpeechOlmoModel(OmniSpeechMetaModel):
    config_class = OmniSpeechOlmoConfig

    def __init__(self, config):
        super(OmniSpeechOlmoModel, self).__init__(config)
        
        # Initialize base model components if model_name is provided
        if hasattr(config, 'model_name') and config.model_name:
            self.base_model = AutoModelForCausalLM.from_pretrained(config.model_name, config=config)
            # Copy necessary attributes from base model
            if hasattr(self.base_model, 'embed_tokens'):
                self.embed_tokens = self.base_model.get_input_embeddings()
        else:
            # For cases where we load from a pre-trained OmniSpeech model
            self.base_model = None


class OmniSpeechOlmoForCausalLM(OmniSpeechMetaForCausalLM):
    config_class = OmniSpeechOlmoConfig

    def __init__(self, config):
        # Don't call super().__init__ to avoid double initialization
        nn.Module.__init__(self)
        self.config = config
        
        # Initialize base model and components
        if hasattr(config, 'model_name') and config.model_name:
            self.base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
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
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

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