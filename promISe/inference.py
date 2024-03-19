import argparse
import json
import os
import time

import pandas as pd
import tensor_parallel as tp
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from config import OPENAI_KEY, PROXY, TASK, DATA_PATH, AGIEVAL_FEWSHOT, manual_prompt
from utils import format_task, read_jsonl, agieval_prompt_construct, mmlu_prompt_construct, compute_metric

choices = ["A", "B", "C", "D"]

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    input_tokens = {k:input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]}
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def load(ckpt_dir: str, model_type: str):
    n_gpus = torch.cuda.device_count()

    if model_type in ['llama','llama2']:
        # we use tensor parallel for loading llama
        tokenizer = LlamaTokenizer.from_pretrained(ckpt_dir, use_fast=False, padding_side="left")
        
        model = LlamaForCausalLM.from_pretrained(ckpt_dir, low_cpu_mem_usage = True, torch_dtype=torch.float16)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])

        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
    
    elif model_type in ['falcon']:
        # however, tensor parallel for running falcon will occur bugs
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=False, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map = 'balanced_low_0', torch_dtype=torch.bfloat16, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0
                
    elif model_type in ['baichuan','moss']:
        # we use tensor parallel for loading baichuan and moss
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=False, add_bos_token=False, padding_side="left", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
    model.eval()
    return model, tokenizer

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_infer(model, tokenizer, prompts, batch_size=8):
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)
        answers.extend(tokenizer.batch_decode(outputs[:, encode_inputs['input_ids'].shape[1]:], skip_special_tokens=True))
    answers = [answer.strip() for answer in answers]
    return answers

def run_single_task(model, tokenizer, benchmark, training, task, instructions, fewshot_num, batchsize):
    run_results = {}
    if benchmark == 'mmlu':
        dev_df = pd.read_csv(os.path.join(DATA_PATH[benchmark], "dev", task + "_dev.csv"), header=None)[:args.ntrain]
        if training:
            test_df = pd.read_csv(os.path.join(DATA_PATH[benchmark], "train", task + "_train.csv"), header=None)
        else:
            test_df = pd.read_csv(os.path.join(DATA_PATH[benchmark], "test", task + "_test.csv"), header=None)[args.ntrain:]

    elif benchmark == 'agieval':
        en = False if task in TASK['agieval_zh'] else True
        skip_passage = False
        if task == 'sat-en-without-passage':
            skip_passage = True
            task = "sat-en" 
        test_json = read_jsonl(os.path.join(DATA_PATH[benchmark], "train", task + ".jsonl"))
        context_row = [0, 1, 3, 5, 7, 9]
        prompt_df = pd.read_csv(AGIEVAL_FEWSHOT, header=0, skiprows=lambda x: x not in context_row,
                                        keep_default_na=False)
    for num, instruction in enumerate(instructions):
        if benchmark == 'mmlu':
            records = mmlu_prompt_construct(num, instruction, tokenizer, fewshot_num, dev_df, test_df)
        elif benchmark == 'agieval':
            records = agieval_prompt_construct(num, instruction, tokenizer, prompt_df, task, en, test_json, skip_passage)
        
        pred_answers = batch_infer(model, tokenizer, [record['prompt'] for record in records], batchsize)
        gold_answers = [record['answer'] for record in records]
        run_results[f'prompt{num}'] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
    return run_results

def main(args:argparse.Namespace):
    filename = os.path.join('outputs', f'{args.benchmark}', f'{args.model_type}_{args.param_size}b', f'round_{args.round}')
    if not os.path.exists(filename):
        os.makedirs(filename)
    model, tokenizer = load(args.ckpt_dir, args.model_type)
    start_time = time.time()
    for task in TASK[args.benchmark]:
        print("=" * 100)
        print(f"run task: {task}")
        output_filename = os.path.join(filename, f'run_{args.benchmark}_{task}.json')
        result_filename = os.path.join(filename, f'acc_{args.benchmark}_{task}.txt')

        task_start_time = time.time()
        #read task prompts
        if args.round == 'manual':
            if args.benchmark == 'agieval' and task in TASK['agieval_en']:
                instructions = manual_prompt[args.benchmark]['en']
            elif args.benchmark == 'agieval' and task in TASK['agieval_zh']:
                instructions = manual_prompt[args.benchmark]['zh']
            else:
                instructions = manual_prompt[args.benchmark]
                instructions = [x.replace('<task>', format_task(task)) for x in instructions]
        elif args.round == '0':
            instruction_filename = os.path.join('instruction', f'{args.benchmark}', f'round_{args.round}', f'{task}.json')
        else:
            instruction_filename = os.path.join('instruction', f'{args.benchmark}', f'round_{args.round}', f'{args.model_type}_{args.param_size}b', f'{task}.json')
         #Read instrunctions JSON
        with open(instruction_filename, 'r') as file:
            json_data = file.read() 
        # Parse the JSON data
        if not instructions:
            instructions = json.loads(json_data)
        run_results = run_single_task(model, tokenizer, args.benchmark, args.training, task, instructions, args.ntrain, args.batch_size)
                        
        with open(output_filename, 'w', encoding='utf8') as f:
            json.dump(run_results, f, ensure_ascii=False, indent=2)

        result = compute_metric(output_filename)
        end_time = time.time()
        result += "task run time %.2f\n" % (end_time - task_start_time)
        result += "total run time %.2f\n" % (end_time - start_time)
        print("task run time %.2f" % (end_time - task_start_time))
        print("total run time %.2f" % (end_time - start_time))
        with open(result_filename, 'w') as f:
            f.write(result)        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='/u01/wangminzheng/checkpt/llama2-7b-hf', help='model checkpoint directory')
    parser.add_argument('--param_size', type=str, default='7')
    parser.add_argument('--model_type', type=str, default='llama2')
    parser.add_argument('--round', type=str, default='0', help='the current round of prompt search')
    parser.add_argument('--ntrain', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--benchmark', type=str, default='agieval', help='benchmark name')
    parser.add_argument('--training', type=bool, default=True, help='training or not')
    args = parser.parse_args()
    print(args)
    main(args)
