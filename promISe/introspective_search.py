import argparse
from tqdm import tqdm
import openai
import json
import os
from tenacity import (
    retry,
    wait_random_exponential,
)  # for exponential backoff
import os
import pandas as pd
from config import OPENAI_KEY, PROXY, TASK, search_instructions


os.environ["http_proxy"] = PROXY
os.environ["https_proxy"] = PROXY
# Apply the API key
openai.api_key = OPENAI_KEY

def format_task(subject: str, benchmark: str):
    if benchmark == 'mmlu':
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
    elif benchmark == 'agieval':
        s = TASK['agieval_name'][subject]
    return s

def format_instruction_results(instructions: str, scores: str, k: int=5):
    selected_instruction = instructions[:k] + instructions[-k:]
    selected_scores = scores[:k] + scores[-k:]
    formated = {}
    for _ins, _score in zip(selected_instruction, selected_scores):
        formated[_ins] = _score
    formated = json.dumps(formated)
    return formated

def initilize_instruction(task: str, step: str, args: argparse.Namespace, language: str):
    instruction = search_instructions[args.benchmark][language][step]
    if language == 'en':
        instruction = instruction.replace('<task>', format_task(task, args.benchmark))
    elif language == 'zh':
        instruction = instruction.replace('《任务》', format_task(task, args.benchmark))
    return instruction

def format_refine_instruction(instructions: list, analysis: str):
    refine_instruction = '#Initial Instructions: \n{}\n\n\n#Analysis: \n{} \n\n\n#Refined Instruction:'.format(instructions, analysis)
    return refine_instruction

def gpt_response(instruction: str, system_content: str):
    '''
    :param instruction: input instuction
    :return: returned content of gpt
    '''
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": instruction}
        ],
    )
    return response['choices'][0]['message']['content']

@retry(wait=wait_random_exponential(min=1, max=60))
def completions_with_backoff(instruction, system_content):
    return gpt_response(instruction, system_content)

def save_json(task: str, instruction: list, args: argparse.Namespace):
    save_dir = os.path.join('instruction', f'{args.benchmark}', f'round_{args.round}', f"{args.model_type}_{args.param_size}b")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, f'{task}.json')

    json_data = json.dumps(instruction, ensure_ascii=False)
    with open(file_path, 'w') as file:
        file.write(json_data)

def get_evaluation_results(benchmark: str,model_name: str, task: str, round: int):
    data = pd.read_excel(os.path.join('outputs',f'{benchmark}',f'{model_name}',f'round_{round}','prompts',f'{task}.xlsx'))
    prompt = data['prompt'].tolist()
    score = data['acc'].tolist()
    return prompt, score

def legal_instruction(instruction:str, benchmark:str, language:str):
    if benchmark == 'mmlu':
        placeholder_symbol = ['<question>','<options>','<answer>']
    elif benchmark == 'agieval':
        placeholder_symbol_en = ['<id>','<passage>','<question>','<options>','<answer>']
        placeholder_symbol_cn = ['<序号>','<材料>','<问题>','<选项>','<答案>']
        placeholder_symbol = placeholder_symbol_en if language == 'en' else placeholder_symbol_cn
   
    for item in placeholder_symbol:
        if item not in instruction:
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='mmlu', help='benchmark name')
    parser.add_argument('--augment_size', type=int, default=15, help='augment size')
    parser.add_argument('--round', type=str, default='1', help='the current round of prompt search')
    parser.add_argument('--param_size', type=str, default='7')
    parser.add_argument('--model_type', type=str, default='llama2')
    args = parser.parse_args()
    model_name = f"{args.model_type}_{args.param_size}b"
    gpt_analysis = {}
    for task in TASK[args.benchmark]:
        print(task)
        en = False if args.benchmark == 'agieval' and task in TASK['agieval_zh'] else True
        language = 'en' if en else 'zh' 
        instructions, scores = get_evaluation_results(args.benchmark, model_name, task, int(args.round)-1)
        eval_input = format_instruction_results(instructions, scores)
        system_content_evaluation = initilize_instruction(task, 'system_content_introspect', args, language)
        print(eval_input)
        
        analysis_results = completions_with_backoff(eval_input, system_content_evaluation)
        gpt_analysis[task] = analysis_results
        refine_instruction = format_refine_instruction(instructions[:3], analysis_results)
        print(analysis_results)
        print(refine_instruction)
        system_content_iteration = initilize_instruction(task, 'system_content_refine', args, language)
    
        responses = []
        for _ in tqdm(range(args.augment_size)):
            _response = ''
            while not legal_instruction(_response, args.benchmark, language):
                _response = completions_with_backoff(refine_instruction, system_content_iteration)
                print(_response)
            responses.append(_response)
        save_json(task, responses, args)