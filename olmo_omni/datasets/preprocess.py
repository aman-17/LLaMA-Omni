# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import copy
from typing import Dict, Sequence

import tokenizers
import torch
import transformers
from packaging import version

from olmo_omni import conversation as conversation_lib
from olmo_omni.arguments import DataArguments
from olmo_omni.constants import DEFAULT_SPEECH_TOKEN, IGNORE_INDEX, SPEECH_TOKEN_INDEX
from olmo_omni.model import *


def tokenizer_speech_token(
    prompt, tokenizer, speech_token_index=SPEECH_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<speech>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [speech_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_SPEECH_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_SPEECH_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_SPEECH_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()

    return sources


def preprocess_olmo2(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_speech:
        input_ids = torch.stack(
            [
                tokenizer_speech_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.OLMO2
    for conversation, target in zip(conversations, targets):
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        total_len = int(target.ne(pad_token_id).sum())
        assistant_start = f"<|assistant|>\n"
        conversation_parts = conversation.split(assistant_start)

        if len(conversation_parts) >= 2:
            prefix = conversation_parts[0] + assistant_start

            if has_speech:
                instruction_len = len(tokenizer_speech_token(prefix, tokenizer))
            else:
                instruction_len = len(tokenizer(prefix).input_ids)

            if instruction_len > 0:
                instruction_len = min(instruction_len, total_len)
                target[:instruction_len] = IGNORE_INDEX
        else:
            target[:] = IGNORE_INDEX
            print(
                f"WARNING: Could not find assistant response pattern in OLMo conversation"
            )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.OLMO2
    ):
        return preprocess_olmo2(sources, tokenizer, has_speech=has_speech)
    raise NotImplementedError
