# OLMo model integration for OmniSpeech
# Based on the Qwen2 implementation pattern

from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM

# Import OLMo classes - we'll handle import errors gracefully
try:
    from transformers import OlmoConfig, OlmoModel, OlmoForCausalLM
except ImportError:
    try:
        from transformers.models.olmo import OlmoConfig, OlmoModel, OlmoForCausalLM
    except ImportError:
        OlmoConfig = None
        OlmoModel = None
        OlmoForCausalLM = None

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..speech_encoder.builder import build_speech_encoder
from ..speech_projector.builder import build_speech_projector
from omni_speech.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX


class OmniSpeechMetaModel:

    def __init__(self, config):
        super(OmniSpeechMetaModel, self).__init__(config)
        if hasattr(config, "speech_encoder"):
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)

    def get_speech_encoder(self):
        speech_encoder = getattr(self, "speech_encoder", None)
        return speech_encoder
    
    def get_speech_projector(self):
        speech_projector = getattr(self, "speech_projector", None)
        return speech_projector

    def initialize_speech_modules(self, model_args):
        self.config.speech_encoder = getattr(model_args, "speech_encoder", None)
        self.config.speech_encoder_type = getattr(model_args, "speech_encoder_type", None)
        self.config.speech_projector_type = getattr(model_args, 'speech_projector_type', 'linear')
        self.config.speech_encoder_ds_rate = getattr(model_args, 'speech_encoder_ds_rate', 5)
        self.config.speech_encoder_hidden_size = getattr(model_args, 'speech_encoder_hidden_size', 1280)

        if self.get_speech_encoder() is None:
            self.speech_encoder = build_speech_encoder(self.config)
        if self.get_speech_projector() is None:
            self.speech_projector = build_speech_projector(self.config)


class OmniSpeechMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_speech_encoder(self):
        return self.get_model().get_speech_encoder()
    
    def get_speech_projector(self):
        return self.get_model().get_speech_projector()

    def encode_speech(self, speech, speech_lengths):
        speech_encoder_type = self.config.speech_encoder_type
        speech_encoder = self.get_speech_encoder()
        if "whisper" in speech_encoder_type.lower():
            encoder_outs = speech_encoder(speech.permute(0, 2, 1))
            speech_lengths = (speech_lengths + 1) // 2
        else:
            raise ValueError(f"Unknown speech encoder type: {speech_encoder_type}")
        
        speech_projector_type = self.config.speech_projector_type
        speech_projector = self.get_speech_projector()
        if speech_projector_type == "linear":
            encoder_outs = speech_projector(encoder_outs)
            speech_lengths = speech_lengths // speech_projector.k
        else:
            raise ValueError(f"Unknown speech projector type: {speech_projector_type}")
        
        speech_features = [encoder_outs[i, :speech_lengths[i]] for i in range(len(encoder_outs))]
        return speech_features

    def prepare_inputs_labels_for_speech_and_text(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        speech, speech_lengths
    ):
        speech_encoder = self.get_speech_encoder()
        if speech_encoder is None or speech is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        speech_features = self.encode_speech(speech, speech_lengths)
        
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_speech_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speech = (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
            speech_token_indices = [-1] + torch.where(cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_nospeech = []
            cur_labels = labels[batch_idx]
            cur_labels_nospeech = []
            for i in range(len(speech_token_indices) - 1):
                cur_input_ids_nospeech.append(cur_input_ids[speech_token_indices[i]+1:speech_token_indices[i+1]])
                cur_labels_nospeech.append(cur_labels[speech_token_indices[i]+1:speech_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_nospeech]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nospeech))
            cur_input_embeds_no_speech = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_speech + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_speech[i])
                cur_new_labels.append(cur_labels_nospeech[i])
                if i < num_speech:
                    cur_speech_features = speech_features[cur_speech_idx]
                    cur_speech_idx += 1
                    cur_new_input_embeds.append(cur_speech_features)
                    cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        
        assert cur_speech_idx == len(speech_features)

        # Truncate sequences to max length as speech features can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


class OmniSpeechOlmoConfig(OlmoConfig if OlmoConfig else AutoConfig):
    model_type = "omni_speech_olmo"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class OmniSpeechOlmoModel(OmniSpeechMetaModel, OlmoModel if OlmoModel else nn.Module):
    config_class = OmniSpeechOlmoConfig

    def __init__(self, config):
        if OlmoModel:
            super(OmniSpeechOlmoModel, self).__init__(config)
        else:
            # Fallback initialization
            nn.Module.__init__(self)
            OmniSpeechMetaModel.__init__(self, config)
            self.config = config


class OmniSpeechOlmoForCausalLM(OlmoForCausalLM if OlmoForCausalLM else AutoModelForCausalLM, OmniSpeechMetaForCausalLM):
    config_class = OmniSpeechOlmoConfig

    def __init__(self, config):
        if OlmoForCausalLM:
            super(OlmoForCausalLM, self).__init__(config)
            self.model = OmniSpeechOlmoModel(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            # Fallback: use AutoModelForCausalLM
            AutoModelForCausalLM.__init__(self, config)
            OmniSpeechMetaForCausalLM.__init__(self)
            
            # Load base OLMo model using AutoModel
            base_model_name = getattr(config, 'model_name', 'allenai/OLMo-2-1124-7B-Instruct')
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            self.model = OmniSpeechOlmoModel(config)
            self.vocab_size = self.base_model.config.vocab_size
            self.lm_head = self.base_model.lm_head
            
            # Copy embeddings from base model
            if hasattr(self.base_model, 'get_input_embeddings'):
                source_embeddings = self.base_model.get_input_embeddings()
                self.model.embed_tokens = nn.Embedding(source_embeddings.num_embeddings, source_embeddings.embedding_dim)
                self.model.embed_tokens.weight.data.copy_(source_embeddings.weight.data)

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
            return_dict=return_dict
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

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if speech is not None:
            inputs['speech'] = speech
            inputs['speech_lengths'] = speech_lengths
        return inputs


AutoConfig.register("omni_speech_olmo", OmniSpeechOlmoConfig)
AutoModelForCausalLM.register(OmniSpeechOlmoConfig, OmniSpeechOlmoForCausalLM)