from qwen2_glide import Qwen2Glide
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import torch
from tqdm import tqdm
import os
import argparse

model_names = {
    "qwq": {
        "target": "Qwen/QwQ-32B-Preview",
        "draft": "sail/longspec-QwQ-32B-Preview"
    }
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="qwq", type=str)
    parser.add_argument("--method", default="tree", type=str, choices=["vanilla", "seq", "tree"])
    parser.add_argument("--id_min", default=60, type=int)
    parser.add_argument("--id_max", default=60, type=int)
    parser.add_argument("--max_gen_len", default=200, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--gamma", default=4, type=int, help='sequence length')
    parser.add_argument('--tree_shape', nargs='+', type=int, default=[4, 16, 16, 16, 16], help='A list of tree size (default: [4, 16, 16, 16, 16])')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    torch.set_printoptions(threshold=200)

    log_path = "./long-bench_results"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, f"output_aime.txt")

    draft_model_name = model_names[args.model_name]["draft"]
    target_model_name = model_names[args.model_name]["target"]
    config = AutoConfig.from_pretrained(target_model_name)
    if args.model_name == "qwq":
        config.pad_token_id = 151643
        config.eos_token_id = 151645
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    qwen2_glide = Qwen2Glide(config, target_model_name, draft_model_name)

    dataset = load_dataset("AI-MO/aimo-validation-aime")
    prompts = []
    test = dataset["train"]
    for doc in tqdm(test):
        if args.id_min <= int(doc["id"]) <= args.id_max:
            question = doc["problem"]
            prompts.append(question)

    meta_prompts = []
    for raw_data in prompts:
        prompt = f"<|im_start|>system\nYou are a helpful assistant<|im_end|>\n" \
                 f"<|im_start|>user\n{raw_data}<|im_end|>\n<|im_start|>assistant\n"
        a = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="right")
        a_cuda = a['input_ids'].cuda()
        len_cuda = a['attention_mask'].sum(dim=-1).cuda()
        meta_prompts.append({
            "prompt": prompt,
            "length": a['attention_mask'].sum(dim=-1).cuda(),
            "input_ids": a_cuda,
        })
    args.test_length = len(meta_prompts)
    

    if args.method == "vanilla":
        vanilla_time = .0
        nums = 0
        with torch.inference_mode():
            for i in range(args.test_length):
                meta_prompt = meta_prompts[i]
                output_ids, num, elapsed_time = qwen2_glide.vanilla_generate(
                    meta_prompt["input_ids"], 
                    prompt_length=meta_prompt["length"], 
                    max_gen_len=args.max_gen_len,
                )
                print(num, elapsed_time)
                print(tokenizer.decode(output_ids[0]))
                vanilla_time += elapsed_time
                nums += num
        
        with open(log_file, "a") as f:
            f.write("vanilla\n")
            f.write(f"{draft_model_name} {args.id_min}~{args.id_max}\n")
            f.write(f"{args.model_name} {args.method} {nums}\n")
            f.write(f"{args.model_name} {args.method} {vanilla_time} {nums / vanilla_time}\n")


    elif args.method == "seq":
        counts = 0
        nums = 0
        glide_time = .0
        with torch.inference_mode():
            for i in range(args.test_length):
                meta_prompt = meta_prompts[i]
                output_ids, count, num, elapsed_time, spec_mask = qwen2_glide.spec_generate(
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
            output_ids, count, num, elapsed_time, spec_mask = qwen2_glide.tree_spec_generate(
                meta_prompts[0]["input_ids"], 
                prompt_length=meta_prompts[0]["length"], 
                max_gen_len=args.max_gen_len, 
                tree_shape=args.tree_shape,
                temperature=args.temperature,
            )
            # real run
            for i in range(args.test_length):
                meta_prompt = meta_prompts[i]
                output_ids, count, num, elapsed_time, spec_mask = qwen2_glide.tree_spec_generate(
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

        with open(log_file, "a") as f:
            f.write(f"{draft_model_name}\n")
            f.write(f"{args.method} tree shape={args.tree_shape} {args.id_min}~{args.id_max}\n")
            f.write(f"{args.model_name} {args.method} {args.temperature} {counts} {nums} {(counts + nums) / nums}\n")
            f.write(f"{args.model_name} {args.method} {args.temperature} {glide_time} {(counts + nums) / glide_time}\n")
