import argparse
from tqdm import tqdm
import openai
import json
import os
import time
import backoff
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
        ]
    )
    return response['choices'][0]['message']['content']


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError,
                       openai.error.ServiceUnavailableError))
def completions_with_backoff(instruction, system_content):
    return gpt_response(instruction, system_content)

def initilize_instruction(task: str, step: str, args: argparse.Namespace, language: str):
    instruction = search_instructions[args.benchmark][language][step]
    if language == 'en':
        instruction = instruction.replace('<task>', format_task(task, args.benchmark))
    elif language == 'zh':
        instruction = instruction.replace('《任务》', format_task(task, args.benchmark))
    return instruction


def save_json(task: str, instruction: list, args: argparse.Namespace):
    save_dir = os.path.join('instruction', f'{args.benchmark}', f'round_{args.round}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, f'{task}.json')
        
    json_data = json.dumps(instruction, ensure_ascii=False)
    with open(file_path, 'w') as file:
        file.write(json_data)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='mmlu', help='benchmark name')
    parser.add_argument('--augment_size', type=int, default=50, help='augment size')
    parser.add_argument('--round', type=str, default='0', help='the current round of prompt search')
    args = parser.parse_args()
    for task in TASK[args.benchmark]:
        print(task)
        if task == 'sat-en-without-passage': 
            save_json(task, temp, args)
            continue
        en = False if args.benchmark == 'agieval' and task in TASK['agieval_zh'] else True
        language = 'en' if en else 'zh'         
        system_content = search_instructions[args.benchmark][language]['system_content_generate']
        instruction = initilize_instruction(task, 'extensive', args, language)
        responses = []
        for _ in tqdm(range(args.augment_size)):
            _response = ''
            while not legal_instruction(_response, args.benchmark, language):
                _response = completions_with_backoff(instruction, system_content)
            responses.append(_response)
        save_json(task, responses, args)
        if task == 'sat-en': temp = responses
    
    