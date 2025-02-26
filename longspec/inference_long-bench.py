from llama_glide import LlamaGlide
from transformers import AutoTokenizer, AutoConfig
import torch
import os
import json
import argparse

dataset2prompt = {
    "gov_report": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nYou are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\nNow, write a one-page summary of the report.</s>\n"
        "<s>assistant\nSummary:"
    ),
    "qmsum": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nYou are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}</s>\n"
        "<s>assistant\nAnswer:"
    ),
    "multi_news": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nYou are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.</s>\n"
        "<s>assistant\nSummary:"
    ),
    "lcc": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nPlease complete the code given below. \n{context}Now, complete the code given.</s>\n"
        "<s>assistant\n"
    ),
    "repobench-p": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nPlease complete the code given below. \n{context}Now, complete the code given.</s>\n"
        "<s>assistant\n"
    ),
}

model_names = {
    "vicuna7b": {
        "target": "lmsys/vicuna-7b-v1.5-16k",
        "draft": "sail/longspec-vicuna-7b-v1.5-16k"
    },
    "vicuna13b": {
        "target": "lmsys/vicuna-13b-v1.5-16k",
        "draft": "sail/longspec-vicuna-13b-v1.5-16k"
    },
    "longchat7b": {
        "target": "lmsys/longchat-7b-v1.5-32k",
        "draft": "sail/longspec-longchat-7b-v1.5-32k"
    },
    "longchat13b": {
        "target": "lmsys/longchat-13b-16k",
        "draft": "sail/longspec-longchat-13b-16k"
    },
    "llama8b": {
        "target": "gradientai/Llama-3-8B-Instruct-262k",
        "draft": "sail/longspec-Llama-3-8B-Instruct-262k"
    },
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="vicuna7b", type=str)
    parser.add_argument("--method", default="tree", type=str, choices=["vanilla", "vanilla_torch", "seq", "tree", "magicdec"])
    parser.add_argument("--task", default="lcc", type=str, choices=[
        "gov_report", "qmsum", "multi_news", "lcc", "repobench-p"
    ])
    parser.add_argument("--data_path_prefix", type=str)
    parser.add_argument("--test_length", default=1, type=int)
    parser.add_argument("--max_gen_len", default=1024, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--gamma", default=4, type=int, help='sequence length')
    parser.add_argument('--tree_shape', nargs='+', type=int, default=[4, 16, 16, 16, 16], help='A list of tree size (default: [4, 16, 16, 16, 16])')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    log_path = "./long-bench_results"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, f"output_{args.task}.txt")

    record_path = f"{log_path}/{args.task}"
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    prompt_format = dataset2prompt[args.task]

    context_length = {
        "longchat7b": 32768,
        "longchat13b": 16384,
        "vicuna7b": 16384,
        "vicuna13b": 16384,
        "llama8b": 262000,
    }
    context_length = context_length[args.model_name] - 2000
    draft_model_name = model_names[args.model_name]["draft"]
    target_model_name = model_names[args.model_name]["target"]

    config = AutoConfig.from_pretrained(target_model_name)
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)

    if args.model_name == "llama8b":
        config.pad_token_id = 128001
        config.eos_token_id = 128009
    llama_glide = LlamaGlide(config, target_model_name, draft_model_name)
    
    meta_prompts = []
    data_path = os.path.join(args.data_path_prefix, f"{args.task}.jsonl")
    data_task = [json.loads(line) for line in open(data_path).readlines()]
    for raw_data in data_task:
        prompt = prompt_format.format(**raw_data)
        a = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="right")
        a_cuda = a['input_ids'].cuda()
        len_cuda = a['attention_mask'].sum(dim=-1).cuda()
        if 1200 < len_cuda <= context_length:
            meta_prompts.append({
                "prompt": prompt,
                "length": a['attention_mask'].sum(dim=-1).cuda(),
                "input_ids": a_cuda,
                "task": args.task,
            })
    if args.test_length == 0:
        args.test_length = len(meta_prompts)
    
    if args.method == "vanilla":
        vanilla_time = .0
        nums = 0
        with torch.inference_mode():
            for i in range(args.test_length):
                meta_prompt = meta_prompts[i]
                output_ids, num, elapsed_time = llama_glide.vanilla_generate(
                    meta_prompt["input_ids"], 
                    prompt_length=meta_prompt["length"], 
                    max_gen_len=args.max_gen_len,
                )
                print(num, elapsed_time)
                print(tokenizer.decode(output_ids[0]))
                vanilla_time += elapsed_time
                nums += num
        
        print("vanilla\n")
        print(f"{draft_model_name}\n")
        print(f"{args.model_name} {args.method} {nums}\n")
        print(f"{args.model_name} {args.method} {vanilla_time} {nums / vanilla_time}\n")


    elif args.method == 'vanilla_torch':
        vanilla_time = .0
        nums = 0
        with torch.inference_mode():
            for i in range(args.test_length):
                meta_prompt = meta_prompts[i]
                output_ids, num, elapsed_time = llama_glide.vanilla_torch_generate(
                    meta_prompt["input_ids"], 
                    prompt_length=meta_prompt["length"], 
                    max_gen_len=args.max_gen_len
                )
                print(num, elapsed_time)
                print(tokenizer.decode(output_ids[0]))
                vanilla_time += elapsed_time
                nums += num
        
        print("vanilla torch\n")
        print(f"{draft_model_name}\n")
        print(f"{args.model_name} {args.method} {nums}\n")
        print(f"{args.model_name} {args.method} {vanilla_time} {nums / vanilla_time}\n")
        

    elif args.method == "seq":
        counts = 0
        nums = 0
        glide_time = .0
        with torch.inference_mode():
            for i in range(args.test_length):
                meta_prompt = meta_prompts[i]
                output_ids, count, num, elapsed_time, spec_mask = llama_glide.spec_generate(
                    meta_prompt["input_ids"], 
                    prompt_length=meta_prompt["length"], 
                    max_gen_len=args.max_gen_len, 
                    gamma=args.gamma,
                    temperature=args.temperature,
                )
                print(count, num, elapsed_time)
                print(tokenizer.decode(output_ids[0]))
                glide_time += elapsed_time
                counts += count
                nums += num

        print(f"{draft_model_name}\n")
        print(f"{args.method} gamma={args.gamma}\n")
        print(f"{args.model_name} {args.method} {args.temperature} {counts} {nums} {(counts + nums) / nums}\n")
        print(f"{args.model_name} {args.method} {args.temperature} {glide_time} {(counts + nums) / glide_time}\n")


    elif args.method == "magicdec":
        counts = 0
        nums = 0
        glide_time = .0
        with torch.inference_mode():
            for i in range(args.test_length):
                meta_prompt = meta_prompts[i]
                output_ids, count, num, elapsed_time, spec_mask = llama_glide.magicdec_generate(
                    meta_prompt["input_ids"], 
                    prompt_length=meta_prompt["length"], 
                    max_gen_len=args.max_gen_len, 
                    gamma=args.gamma,
                    temperature=args.temperature,
                )
                print(count, num, elapsed_time)
                print(tokenizer.decode(output_ids[0]))
                glide_time += elapsed_time
                counts += count
                nums += num

        print(f"{draft_model_name}\n")
        print(f"{args.method} gamma={args.gamma}\n")
        print(f"{args.model_name} {args.method} {args.temperature} {counts} {nums} {(counts + nums) / nums}\n")
        print(f"{args.model_name} {args.method} {args.temperature} {glide_time} {(counts + nums) / glide_time}\n")


    elif args.method == "tree":
        counts = 0
        nums = 0
        glide_time = .0
        with torch.inference_mode():
            # warm up
            output_ids, count, num, elapsed_time, spec_mask = llama_glide.tree_spec_generate(
                meta_prompts[0]["input_ids"], 
                prompt_length=meta_prompts[0]["length"], 
                max_gen_len=args.max_gen_len, 
                tree_shape=args.tree_shape,
                temperature=args.temperature,
            )
            # real run
            for i in range(args.test_length):
                meta_prompt = meta_prompts[i]
                output_ids, count, num, elapsed_time, spec_mask = llama_glide.tree_spec_generate(
                    meta_prompt["input_ids"], 
                    prompt_length=meta_prompt["length"], 
                    max_gen_len=args.max_gen_len, 
                    tree_shape=args.tree_shape,
                    temperature=args.temperature,
                )
                print(count, num, elapsed_time)
                print(tokenizer.decode(output_ids[0]))
                glide_time += elapsed_time
                counts += count
                nums += num

        print(f"{draft_model_name}\n")
        print(f"{args.method} tree shape={args.tree_shape}\n")
        print(f"{args.model_name} {args.method} {args.temperature} {counts} {nums} {(counts + nums) / nums}\n")
        print(f"{args.model_name} {args.method} {args.temperature} {glide_time} {(counts + nums) / glide_time}\n")
