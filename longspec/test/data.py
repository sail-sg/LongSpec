# All the data files can be found in https://huggingface.co/datasets/sail/longspec-data

import torch
from transformers import PreTrainedTokenizer


class LlamaLongDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __call__(self, batch):
        indices = [item["index"] for item in batch]

        input_ids = torch.ones((len(batch), self.max_seq_length)).long().fill_(self.tokenizer.pad_token_id)
        labels = input_ids.clone()

        realMaxLen = 0
        system_prompt = "<s>system\nYou are a helpful assistant</s>\n"
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
                    prompt = f"<s>user\nPlease summarize the following article: {article}</s>\n"
                    answer = f"<s>assistant\n{abstract}</s>\n"
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
                        user_prompt = f"<s>user\n{turn["content"]}</s>\n"
                        prompt_id = self.tokenizer(user_prompt, padding="longest", truncation=True, 
                                                   max_length=self.max_seq_length, return_tensors="pt").input_ids
                        previous_content_len = content_len
                        content_len = min(previous_content_len + prompt_id.size(1), self.max_seq_length)
                        input_ids[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]
                        labels[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]
                    elif turn["role"] == "assistant":
                        assistant_answer = f"<s>assistant\n{turn["content"]}</s>\n"
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


class LlamaLongSFTDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __call__(self, batch):
        indices = [item["index"] for item in batch]

        input_ids = torch.ones((len(batch), self.max_seq_length)).long().fill_(self.tokenizer.pad_token_id)
        labels = input_ids.clone()

        realMaxLen = 0
        system_prompt = "<s>system\nYou are a helpful assistant</s>\n"
        system_prompt_id = self.tokenizer(system_prompt, padding="longest", truncation=True, 
                                          max_length=self.max_seq_length, return_tensors="pt").input_ids
        
        for idx, item in enumerate(batch):
            if item["source"] == "gov-report":
                input_ids[idx, :system_prompt_id.size(1)] = system_prompt_id[0]
                content_len = system_prompt_id.size(1)

                prompt = f"<s>user\nYou are given a report by a government agency. Write a one-page summary of the report.\n\n" \
                         f"Report:\n{item["report"]}\n\nNow, write a one-page summary of the report.</s>\n"
                prompt_id = self.tokenizer(prompt, padding="longest", truncation=True, 
                                           max_length=self.max_seq_length - 1024, return_tensors="pt").input_ids
                previous_content_len = content_len
                content_len = min(previous_content_len + prompt_id.size(1), self.max_seq_length)
                input_ids[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]

                answer = f"<s>assistant\nSummary:{item["summary"]}</s>\n"
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

                prompt = f"<s>user\nYou are given several news passages. Write a one-page summary of all news. \n\n" \
                         f"News:\n{item["document"]}\n\nNow, write a one-page summary of all the news.</s>\n"
                prompt_id = self.tokenizer(prompt, padding="longest", truncation=True, 
                                           max_length=self.max_seq_length - 1024, return_tensors="pt").input_ids
                previous_content_len = content_len
                content_len = min(previous_content_len + prompt_id.size(1), self.max_seq_length)
                input_ids[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]

                answer = f"<s>assistant\nSummary:{item["summary"]}</s>\n"
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

                prompt = f"<s>user\nYou are given a meeting transcript. Write a summary of the transcript. \n\n" \
                         f"Transcript:\n{item["transcript"]}\n\nNow, write a summary of the transcript.</s>\n"
                prompt_id = self.tokenizer(prompt, padding="longest", truncation=True, 
                                           max_length=self.max_seq_length - 1024, return_tensors="pt").input_ids
                previous_content_len = content_len
                content_len = min(previous_content_len + prompt_id.size(1), self.max_seq_length)
                input_ids[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]

                answer = f"<s>assistant\nSummary:{item["summary"]}</s>\n"
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

                prompt = f"<s>user\nPlease complete the code given below. \n\n" \
                         f"Code:\n{item["prefix"]}\n\nNow, complete the code given.</s>\n"
                prompt_id = self.tokenizer(prompt, padding="longest", truncation=True, 
                                           max_length=self.max_seq_length - 1024, return_tensors="pt").input_ids
                previous_content_len = content_len
                content_len = min(previous_content_len + prompt_id.size(1), self.max_seq_length)
                input_ids[idx, previous_content_len : content_len] = prompt_id[0, : content_len - previous_content_len]

                answer = f"<s>assistant\n{item["suffix"]}</s>\n"
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


class QwenLongDataCollator:
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


class QwenLongCoTDataSFTCollator:
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
