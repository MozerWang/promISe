import argparse
import pandas as pd
import os
import json
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='falcon')
    parser.add_argument('--param_size', type=str, default='7')
    parser.add_argument('--round', type=str, default='0')
    parser.add_argument('--benchmark', type=str, default='mmlu')
    args = parser.parse_args()

    model_name = f"{args.model_type}_{args.param_size}b" 
    prompt_datapath = os.path.join('outputs',f'{args.benchmark}', model_name, f'round_{args.round}')
    best_prompt_path = os.path.join('instruction', f'{args.benchmark}', 'round_best', model_name, 'best_prompt_data.json')
    with open(best_prompt_path, 'r') as json_file:
        prompt_dics = json.load(json_file)

    files_list = glob.glob(prompt_datapath + '/*.txt')
    for file in files_list:
        result_lists = []
        subject = file.split(f'acc_{args.benchmark}_')[1].replace('.txt',"")
        with open(file, 'r') as f1:
            lines = f1.readlines()
        
        if subject == 'sat-en-without-passage' and args.benchmark == 'agieval':
            subject_name = 'sat-en'
        else:
            subject_name = subject
            
        if args.round == '0':
            instruction_filename = os.path.join('instruction', f'{args.benchmark}', f'round_{args.round}', f'{subject_name}.json')
        else:
            instruction_filename = os.path.join('instruction', f'{args.benchmark}', f'round_{args.round}', model_name, f'{subject_name}.json')

        with open(instruction_filename, 'r') as file:
            json_data = file.read()
        # Parse the JSON data
        instructions = json.loads(json_data)
        
        for line in lines:
            if "ACC-max: " in line:
                break
            line = line.replace('\n',"")
            idx, acc = line.split('ACC-prompt')[1].split(': ')
            idx = int(idx)
            prompt = instructions[idx]
            result_list = [prompt, float(acc), idx]
            result_lists.append(result_list)
        best_prompt = prompt_dics[subject]['prompt']
        best_prompt_idx = int(prompt_dics[subject]['max_idx'])
        best_prompt_acc = float(prompt_dics[subject]['max_acc'])
        best_prompt_list = [best_prompt, best_prompt_acc, best_prompt_idx]
        result_lists.append(best_prompt_list)
        
        df = pd.DataFrame(result_lists, columns=["prompt","acc","idx"]).reset_index(drop=True)
        df = df.sort_values('acc',ascending=False).reset_index(drop=True)
        prompts_path = os.path.join('outputs', f'{args.benchmark}', model_name, f'round_{args.round}', 'prompts')
        if not os.path.exists(prompts_path):
            os.makedirs(prompts_path)
        df.to_excel(os.path.join('outputs', f'{args.benchmark}', model_name, f'round_{args.round}', 'prompts', f'{subject}.xlsx')) 
            
        
 
    
    