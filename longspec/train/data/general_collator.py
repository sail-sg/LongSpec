import collections
import copy
import json
import os.path
import random
from glob import glob
from typing import List, Dict, Tuple, Union, Any, Callable, Optional
import ast

import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from data.math import decompose_deepseek_math_cot_v2
from torch.nn.utils.rnn import pad_sequence
from fastchat.model.model_adapter import get_conversation_template

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class DPOCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int, padding: str = "longest"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding = padding

    def __call__(self, batch):
        chosen = [item["chosen"] for item in batch]
        reject = [item["reject"] for item in batch]
        indices = [item["index"] for item in batch]
        text_inputs = chosen + reject

        text_prompts = []
        for item in batch:
            if "chosen_prompt" in item:
                text_prompts.append(item["chosen_prompt"])
            else:
                text_prompts.append(item["prompt"])
        for item in batch:
            if "reject_prompt" in item:
                text_prompts.append(item["reject_prompt"])
            else:
                text_prompts.append(item["prompt"])
        # prompt = [item["prompt"] for item in batch]
        # text_prompts = prompt + prompt

        encoded_prompts = self.tokenizer(text_prompts, padding=self.padding, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        input_lens = torch.sum(encoded_prompts["attention_mask"], dim=-1)

        encoded_inputs = self.tokenizer(text_inputs, padding=self.padding, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if self.tokenizer.padding_side == "left":
            padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
            input_lens = input_lens + padding_len

        labels = encoded_inputs["input_ids"].clone()
        prompt_mask = torch.arange(encoded_inputs["input_ids"].size(1))[None, :] < input_lens[:, None]
        if prompt_mask.sum() == labels.numel():  # FIXME: This could also induce NAN loss during DPO with SFT loss. @2024/08/09
            logger.warning(f"Prompt mask is all True. Indices: {indices}")
            prompt_mask[0, -1] = False

        labels[prompt_mask] = self.tokenizer.pad_token_id

        encoded_inputs["labels"] = labels
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": text_prompts,
            "chosen": chosen,
            "reject": reject,
        }
        return encoded_inputs


class DPODataSFTCollator:
    """
    Note that when you are using the DPO pair dataset, you may overlook the oversampling of chosen samples.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        prompt = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        indices = [item["index"] for item in batch]

        text_prompts = prompt
        text_inputs = chosen

        encoded_prompts = self.tokenizer(text_prompts, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        input_lens = torch.sum(encoded_prompts["attention_mask"], dim=-1)

        encoded_inputs = self.tokenizer(text_inputs, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if self.tokenizer.padding_side == "left":
            padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
            input_lens = input_lens + padding_len

        labels = encoded_inputs["input_ids"].clone()
        prompt_mask = torch.arange(encoded_inputs["input_ids"].size(1))[None, :] < input_lens[:, None]
        if prompt_mask.sum() == labels.numel():
            logger.warning(f"Prompt mask is all True. Indices: {indices}")
            prompt_mask[0, -1] = False

        labels[prompt_mask] = self.tokenizer.pad_token_id

        encoded_inputs["labels"] = labels
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": prompt,
            "chosen": chosen,
            "response": chosen,
        }
        if "label" in batch[0]:
            encoded_inputs["meta_data"]["label"] = [item["label"] for item in batch]
        return encoded_inputs


class SFTCollator:

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_len = max_seq_length

    def __call__(self, batch):
        prompt = [item["prompt"] for item in batch]
        answer = [item["chosen"] for item in batch]
        indices = [item["index"] for item in batch]

        input_ids = torch.ones((len(prompt), self.max_len)).long().fill_(self.tokenizer.pad_token_id)
        labels = input_ids.clone()

        realMaxLen = 0
        for i in range(len(prompt)):
            format_prompt = (
                "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
                f"<|im_start|>user\n{prompt[i]}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            prompt_id = self.tokenizer(format_prompt, padding="longest", truncation=True, max_length=int(self.max_seq_length * 0.8), return_tensors="pt").input_ids
            answer_id = self.tokenizer(answer[i], padding="longest", truncation=True, max_length=int(self.max_seq_length * 0.8), return_tensors="pt").input_ids
            input_ids[i, :prompt_id.size(1)] = prompt_id[0]
            # labels[i, :prompt_id.size(1)] = -100
            answer_len = min(prompt_id.size(1) + answer_id.size(1), self.max_len)
            input_ids[i, prompt_id.size(1): answer_len] = answer_id[0, :answer_len - prompt_id.size(1)]
            labels[i, prompt_id.size(1): answer_len] = answer_id[0, :answer_len - prompt_id.size(1)]
            realMaxLen = max(realMaxLen, answer_len)

        encoded_inputs = {}
        encoded_inputs["input_ids"] = input_ids[:, :realMaxLen]
        encoded_inputs["labels"] = labels[:, :realMaxLen]
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": prompt,
        }
        return encoded_inputs


class ShareGPTDataSFTCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __call__(self, batch):
        indices = [item["index"] for item in batch]

        input_ids = torch.ones((len(batch), self.max_seq_length)).long().fill_(self.tokenizer.pad_token_id)
        labels = input_ids.clone()

        realMaxLen = 0
        system_prompt = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
        system_prompt_id = self.tokenizer(system_prompt, padding="longest", truncation=True, 
                                          max_length=self.max_seq_length, return_tensors="pt").input_ids
        input_ids[:, :system_prompt_id.size(1)] = system_prompt_id[0].unsqueeze(0)

        for idx, item in enumerate(batch):
            content_len = system_prompt_id.size(1)
            for turn in item["conversations"]:
                if content_len >= self.max_seq_length:
                    break
                if turn["role"] == "user":
                    user_prompt = f"<|im_start|>user\n{turn["content"]}<|im_end|>\n"
                    prompt_id = self.tokenizer(user_prompt, padding="longest", truncation=True, 
                                               max_length=self.max_seq_length, return_tensors="pt").input_ids
                    previous_content_len = content_len
                    content_len = min(previous_content_len + prompt_id.size(1), self.max_seq_length)
                    input_ids[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]
                elif turn["role"] == "assistant":
                    assistant_answer = f"<|im_start|>assistant\n{turn["content"]}<|im_end|>\n"
                    answer_id = self.tokenizer(assistant_answer, padding="longest", truncation=True, 
                                               max_length=self.max_seq_length, return_tensors="pt").input_ids
                    previous_content_len = content_len
                    content_len = min(previous_content_len + answer_id.size(1), self.max_seq_length)
                    input_ids[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                    labels[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                else:
                    pass
                realMaxLen = max(realMaxLen, content_len)
        encoded_inputs = {}
        encoded_inputs["input_ids"] = input_ids[:, :realMaxLen]
        encoded_inputs["labels"] = labels[:, :realMaxLen]
        encoded_inputs["meta_data"] = {
            "index": indices,
        }
        return encoded_inputs


class LongDataSFTCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __call__(self, batch):
        indices = [item["index"] for item in batch]

        input_ids = torch.ones((len(batch), self.max_seq_length)).long().fill_(self.tokenizer.pad_token_id)
        labels = input_ids.clone()

        realMaxLen = 0
        system_prompt = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
        system_prompt_id = self.tokenizer(system_prompt, padding="longest", truncation=True, 
                                          max_length=self.max_seq_length, return_tensors="pt").input_ids
        
        for idx, item in enumerate(batch):
            if item["source"] == "code" or item["source"] == "book":
                encoded_inputs = self.tokenizer(item["text"], padding="longest", truncation=True, 
                                                max_length=self.max_seq_length, return_tensors="pt").input_ids
                labels = encoded_inputs.clone()
                input_ids[idx] = encoded_inputs[0]
                labels[idx] = encoded_inputs[0]
                realMaxLen = max(realMaxLen, len(encoded_inputs[0]))
            
            elif item["source"] == "arxiv":
                input_ids[idx, :system_prompt_id.size(1)] = system_prompt_id[0]
                content_len = system_prompt_id.size(1)

                if isinstance(item["article"], list):
                    articles = item["article"]
                    abstracts = item["abstract"]
                elif isinstance(item["article"], str):
                    articles = [item["article"]]
                    abstracts = [item["abstract"]]
                else:
                    raise ValueError("item['article'] must be a list or a string")
                
                for article, abstract in zip(articles, abstracts):
                    if content_len >= self.max_seq_length:
                        break

                    prompt = f"<|im_start|>user\nPlease summarize the following article: {article}<|im_end|>\n"
                    prompt_id = self.tokenizer(prompt, padding="longest", truncation=True, 
                                               max_length=self.max_seq_length, return_tensors="pt").input_ids
                    previous_content_len = content_len
                    content_len = min(previous_content_len + prompt_id.size(1), self.max_seq_length)
                    input_ids[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]

                    if content_len < self.max_seq_length:
                        answer = f"<|im_start|>assistant\n{abstract}<|im_end|>\n"
                        answer_id = self.tokenizer(answer, padding="longest", truncation=True, 
                                                max_length=self.max_seq_length, return_tensors="pt").input_ids
                        previous_content_len = content_len
                        content_len = min(previous_content_len + answer_id.size(1), self.max_seq_length)
                        input_ids[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                        labels[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                    
                    realMaxLen = max(realMaxLen, content_len)
            
            elif item["source"] == "tulu-v2":
                input_ids[idx, :system_prompt_id.size(1)] = system_prompt_id[0]
                content_len = system_prompt_id.size(1)
                for turn in item["conversation"]:
                    if content_len >= self.max_seq_length:
                        break
                    if turn["role"] == "user":
                        user_prompt = f"<|im_start|>user\n{turn["content"]}<|im_end|>\n"
                        prompt_id = self.tokenizer(user_prompt, padding="longest", truncation=True, 
                                                   max_length=self.max_seq_length, return_tensors="pt").input_ids
                        previous_content_len = content_len
                        content_len = min(previous_content_len + prompt_id.size(1), self.max_seq_length)
                        input_ids[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]
                    elif turn["role"] == "assistant":
                        assistant_answer = f"<|im_start|>assistant\n{turn["content"]}<|im_end|>\n"
                        answer_id = self.tokenizer(assistant_answer, padding="longest", truncation=True, 
                                                   max_length=self.max_seq_length, return_tensors="pt").input_ids
                        previous_content_len = content_len
                        content_len = min(previous_content_len + answer_id.size(1), self.max_seq_length)
                        input_ids[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                        labels[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                    else:
                        pass
                    realMaxLen = max(realMaxLen, content_len)
            else:
                raise ValueError(f"Unknown data source {item["source"]}")
            
        encoded_inputs = {}
        encoded_inputs["input_ids"] = input_ids[:, :realMaxLen]
        encoded_inputs["labels"] = labels[:, :realMaxLen]
        encoded_inputs["meta_data"] = {
            "index": indices,
        }
        return encoded_inputs


class LongDataNoMaskSFTCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __call__(self, batch):
        indices = [item["index"] for item in batch]

        input_ids = torch.ones((len(batch), self.max_seq_length)).long().fill_(self.tokenizer.pad_token_id)
        labels = input_ids.clone()

        realMaxLen = 0
        system_prompt = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
        system_prompt_id = self.tokenizer(system_prompt, padding="longest", truncation=True, 
                                          max_length=self.max_seq_length, return_tensors="pt").input_ids
        
        for idx, item in enumerate(batch):
            if item["source"] == "code" or item["source"] == "book":
                encoded_inputs = self.tokenizer(item["text"], padding="longest", truncation=True, 
                                                max_length=self.max_seq_length, return_tensors="pt").input_ids
                labels = encoded_inputs.clone()
                input_ids[idx] = encoded_inputs[0]
                labels[idx] = encoded_inputs[0]
                realMaxLen = max(realMaxLen, len(encoded_inputs[0]))
            
            elif item["source"] == "arxiv":
                input_ids[idx, :system_prompt_id.size(1)] = system_prompt_id[0]
                content_len = system_prompt_id.size(1)

                if isinstance(item["article"], list):
                    articles = item["article"]
                    abstracts = item["abstract"]
                elif isinstance(item["article"], str):
                    articles = [item["article"]]
                    abstracts = [item["abstract"]]
                else:
                    raise ValueError("item['article'] must be a list or a string")
                
                for article, abstract in zip(articles, abstracts):
                    if content_len >= self.max_seq_length:
                        break
                    prompt = f"<|im_start|>user\nPlease summarize the following article: {article}<|im_end|>\n"
                    answer = f"<|im_start|>assistant\n{abstract}<|im_end|>\n"
                    text = prompt + answer
                    text_id = self.tokenizer(text, padding="longest", truncation=True,
                                             max_length=self.max_seq_length, return_tensors="pt").input_ids
                    previous_content_len = content_len
                    content_len = min(previous_content_len + text_id.size(1), self.max_seq_length)
                    input_ids[idx, previous_content_len : content_len] = text_id[0, : content_len - previous_content_len]
                    labels[idx, previous_content_len : content_len] = text_id[0, : content_len - previous_content_len]
                    realMaxLen = max(realMaxLen, content_len)
            
            elif item["source"] == "tulu-v2":
                input_ids[idx, :system_prompt_id.size(1)] = system_prompt_id[0]
                content_len = system_prompt_id.size(1)
                for turn in item["conversation"]:
                    if content_len >= self.max_seq_length:
                        break
                    if turn["role"] == "user":
                        user_prompt = f"<|im_start|>user\n{turn["content"]}<|im_end|>\n"
                        prompt_id = self.tokenizer(user_prompt, padding="longest", truncation=True, 
                                                   max_length=self.max_seq_length, return_tensors="pt").input_ids
                        previous_content_len = content_len
                        content_len = min(previous_content_len + prompt_id.size(1), self.max_seq_length)
                        input_ids[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]
                        labels[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]
                    elif turn["role"] == "assistant":
                        assistant_answer = f"<|im_start|>assistant\n{turn["content"]}<|im_end|>\n"
                        answer_id = self.tokenizer(assistant_answer, padding="longest", truncation=True, 
                                                   max_length=self.max_seq_length, return_tensors="pt").input_ids
                        previous_content_len = content_len
                        content_len = min(previous_content_len + answer_id.size(1), self.max_seq_length)
                        input_ids[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                        labels[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                    else:
                        pass
                    realMaxLen = max(realMaxLen, content_len)
            else:
                raise ValueError(f"Unknown data source {item["source"]}")
            
        encoded_inputs = {}
        encoded_inputs["input_ids"] = input_ids[:, :realMaxLen]
        encoded_inputs["labels"] = labels[:, :realMaxLen]
        encoded_inputs["meta_data"] = {
            "index": indices,
        }
        return encoded_inputs


class LongSFTDataSFTCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __call__(self, batch):
        indices = [item["index"] for item in batch]

        input_ids = torch.ones((len(batch), self.max_seq_length)).long().fill_(self.tokenizer.pad_token_id)
        labels = input_ids.clone()

        realMaxLen = 0
        system_prompt = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
        system_prompt_id = self.tokenizer(system_prompt, padding="longest", truncation=True, 
                                          max_length=self.max_seq_length, return_tensors="pt").input_ids
        
        for idx, item in enumerate(batch):
            if item["source"] == "gov-report":
                input_ids[idx, :system_prompt_id.size(1)] = system_prompt_id[0]
                content_len = system_prompt_id.size(1)

                prompt = f"<|im_start|>user\nYou are given a report by a government agency. Write a one-page summary of the report.\n\n" \
                         f"Report:\n{item["report"]}\n\nNow, write a one-page summary of the report.<|im_end|>\n"
                prompt_id = self.tokenizer(prompt, padding="longest", truncation=True, 
                                           max_length=self.max_seq_length - 1024, return_tensors="pt").input_ids
                previous_content_len = content_len
                content_len = min(previous_content_len + prompt_id.size(1), self.max_seq_length)
                input_ids[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]

                answer = f"<|im_start|>assistant\nSummary:{item["summary"]}<|im_end|>\n"
                answer_id = self.tokenizer(answer, padding="longest", truncation=True, 
                                           max_length=self.max_seq_length, return_tensors="pt").input_ids
                previous_content_len = content_len
                content_len = min(previous_content_len + answer_id.size(1), self.max_seq_length)
                input_ids[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                labels[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                
                realMaxLen = max(realMaxLen, content_len)
            
            elif item["source"] == "multi-news":
                input_ids[idx, :system_prompt_id.size(1)] = system_prompt_id[0]
                content_len = system_prompt_id.size(1)

                prompt = f"<|im_start|>user\nYou are given several news passages. Write a one-page summary of all news. \n\n" \
                         f"News:\n{item["document"]}\n\nNow, write a one-page summary of all the news.<|im_end|>\n"
                prompt_id = self.tokenizer(prompt, padding="longest", truncation=True, 
                                           max_length=self.max_seq_length - 1024, return_tensors="pt").input_ids
                previous_content_len = content_len
                content_len = min(previous_content_len + prompt_id.size(1), self.max_seq_length)
                input_ids[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]

                answer = f"<|im_start|>assistant\nSummary:{item["summary"]}<|im_end|>\n"
                answer_id = self.tokenizer(answer, padding="longest", truncation=True, 
                                           max_length=self.max_seq_length, return_tensors="pt").input_ids
                previous_content_len = content_len
                content_len = min(previous_content_len + answer_id.size(1), self.max_seq_length)
                input_ids[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                labels[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                
                realMaxLen = max(realMaxLen, content_len)
            
            elif item["source"] == "meetingbank":
                input_ids[idx, :system_prompt_id.size(1)] = system_prompt_id[0]
                content_len = system_prompt_id.size(1)

                prompt = f"<|im_start|>user\nYou are given a meeting transcript. Write a summary of the transcript. \n\n" \
                         f"Transcript:\n{item["transcript"]}\n\nNow, write a summary of the transcript.<|im_end|>\n"
                prompt_id = self.tokenizer(prompt, padding="longest", truncation=True, 
                                           max_length=self.max_seq_length - 1024, return_tensors="pt").input_ids
                previous_content_len = content_len
                content_len = min(previous_content_len + prompt_id.size(1), self.max_seq_length)
                input_ids[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]

                answer = f"<|im_start|>assistant\nSummary:{item["summary"]}<|im_end|>\n"
                answer_id = self.tokenizer(answer, padding="longest", truncation=True, 
                                           max_length=self.max_seq_length, return_tensors="pt").input_ids
                previous_content_len = content_len
                content_len = min(previous_content_len + answer_id.size(1), self.max_seq_length)
                input_ids[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                labels[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                
                realMaxLen = max(realMaxLen, content_len)
            
            elif item["source"] == "code":
                input_ids[idx, :system_prompt_id.size(1)] = system_prompt_id[0]
                content_len = system_prompt_id.size(1)

                prompt = f"<|im_start|>user\nPlease complete the code given below. \n\n" \
                         f"Code:\n{item["prefix"]}\n\nNow, complete the code given.<|im_end|>\n"
                prompt_id = self.tokenizer(prompt, padding="longest", truncation=True, 
                                           max_length=self.max_seq_length - 1024, return_tensors="pt").input_ids
                previous_content_len = content_len
                content_len = min(previous_content_len + prompt_id.size(1), self.max_seq_length)
                input_ids[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]

                answer = f"<|im_start|>assistant\n{item["suffix"]}<|im_end|>\n"
                answer_id = self.tokenizer(answer, padding="longest", truncation=True, 
                                           max_length=self.max_seq_length, return_tensors="pt").input_ids
                previous_content_len = content_len
                content_len = min(previous_content_len + answer_id.size(1), self.max_seq_length)
                input_ids[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                labels[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
                
                realMaxLen = max(realMaxLen, content_len)
            
            else:
                raise ValueError(f"Unknown data source {item["source"]}")
            
        encoded_inputs = {}
        encoded_inputs["input_ids"] = input_ids[:, :realMaxLen]
        encoded_inputs["labels"] = labels[:, :realMaxLen]
        encoded_inputs["meta_data"] = {
            "index": indices,
        }
        return encoded_inputs


class LongCoTDataSFTCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __call__(self, batch):
        indices = [item["index"] for item in batch]

        input_ids = torch.ones((len(batch), self.max_seq_length)).long().fill_(self.tokenizer.pad_token_id)
        labels = input_ids.clone()

        realMaxLen = 0
        system_prompt = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
        system_prompt_id = self.tokenizer(system_prompt, padding="longest", truncation=True, 
                                          max_length=self.max_seq_length, return_tensors="pt").input_ids
        
        for idx, item in enumerate(batch):
            
            input_ids[idx, :system_prompt_id.size(1)] = system_prompt_id[0]
            content_len = system_prompt_id.size(1)

            prompt = f"<|im_start|>user\n{item["problem"]}<|im_end|>\n"
            prompt_id = self.tokenizer(prompt, padding="longest", truncation=True, 
                                        max_length=self.max_seq_length - 1024, return_tensors="pt").input_ids
            previous_content_len = content_len
            content_len = min(previous_content_len + prompt_id.size(1), self.max_seq_length)
            input_ids[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]

            answer = f"<|im_start|>assistant\n{item["qwq"]}<|im_end|>\n"
            answer_id = self.tokenizer(answer, padding="longest", truncation=True, 
                                        max_length=self.max_seq_length, return_tensors="pt").input_ids
            previous_content_len = content_len
            content_len = min(previous_content_len + answer_id.size(1), self.max_seq_length)
            input_ids[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
            labels[idx, previous_content_len : content_len] = answer_id[0, : content_len - previous_content_len]
            
            realMaxLen = max(realMaxLen, content_len)

        encoded_inputs = {}
        encoded_inputs["input_ids"] = input_ids[:, :realMaxLen]
        encoded_inputs["labels"] = labels[:, :realMaxLen]
        encoded_inputs["meta_data"] = {
            "index": indices,
        }
        return encoded_inputs


class WeightDataSFTCollator:
    """
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        probs = [torch.tensor(ast.literal_eval(item["prompt"]) + [int(item['eos_id'])]) for item in batch]
        chosen = [torch.tensor(ast.literal_eval(item["chosen"]) + [int(item['eos_id'])]) for item in batch]
        indices = [item["index"] for item in batch]

        encoded_inputs = {}
        text_inputs = chosen
        probs = pad_sequence(probs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        text_inputs = pad_sequence(text_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        text_inputs = text_inputs[:, :self.max_seq_length]
        labels = text_inputs[:, :self.max_seq_length]
        # probs is less than input ids and labels with 2
        encoded_inputs['probs'] = probs[:, :labels.size(1) - 2]
        encoded_inputs['input_ids'] = text_inputs
        encoded_inputs['labels'] = labels[:, :self.max_seq_length]
        encoded_inputs['attention_mask'] = labels.ne(self.tokenizer.pad_token_id)

        if self.tokenizer.padding_side == "left":
            logger.warning(f"you are using left side padding, which is dangerous!")

        return encoded_inputs


class Trajectory2ValueCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        prompt = [item["prompt"] for item in batch]
        inputs = [item["input"] for item in batch]
        indices = [item["index"] for item in batch]
        values = [item["value"] for item in batch]

        text_prompts = prompt
        text_inputs = inputs

        encoded_prompts = self.tokenizer(text_prompts, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        input_lens = torch.sum(encoded_prompts["attention_mask"], dim=-1)

        encoded_inputs = self.tokenizer(text_inputs, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if self.tokenizer.padding_side == "left":
            padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
            input_lens = input_lens + padding_len

        labels = encoded_inputs["input_ids"].clone()
        prompt_mask = torch.arange(encoded_inputs["input_ids"].size(1))[None, :] < input_lens[:, None]
        if prompt_mask.sum() == labels.numel():
            logger.warning(f"Prompt mask is all True. Indices: {indices}")
            prompt_mask[0, -1] = False

        labels[prompt_mask] = self.tokenizer.pad_token_id

        encoded_inputs["labels"] = labels
        encoded_inputs["values"] = torch.tensor(values, dtype=torch.long)
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": prompt,
            "input": inputs,
            "response": inputs,
            "label": values,
        }
        return encoded_inputs


class StepEndingsCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        prompt = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        indices = [item["index"] for item in batch]

        text_prompts = prompt
        text_inputs = chosen

        encoded_prompts = self.tokenizer(text_prompts, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        input_lens = torch.sum(encoded_prompts["attention_mask"], dim=-1)

        encoded_inputs = self.tokenizer(text_inputs, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if self.tokenizer.padding_side == "left":
            padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
            input_lens = input_lens + padding_len
        else:
            padding_len = torch.zeros(len(batch), dtype=torch.long)

        labels = encoded_inputs["input_ids"].clone()
        prompt_mask = torch.arange(encoded_inputs["input_ids"].size(1))[None, :] < input_lens[:, None]
        if prompt_mask.sum() == labels.numel():
            logger.warning(f"Prompt mask is all True. Indices: {indices}")
            prompt_mask[0, -1] = False

        labels[prompt_mask] = self.tokenizer.pad_token_id

        endings = []
        for b, item in enumerate(batch):
            ending = decompose_deepseek_math_cot_v2(item["prompt"], item["response"], self.max_seq_length, self.tokenizer)
            ending = [e + padding_len[b].item() for e in ending]
            endings.append(ending)

        encoded_inputs["labels"] = labels
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": prompt,
            "chosen": chosen,
            "response": [item["response"] for item in batch],
            "ending": endings,
            "type": [None] * len(endings),
        }
        if "label" in batch[0]:
            encoded_inputs["meta_data"]["label"] = [item["label"] for item in batch]
        return encoded_inputs
