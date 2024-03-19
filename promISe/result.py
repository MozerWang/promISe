import argparse
import pandas as pd
import os
import json
import glob
from config import categories, subcategories, manual_prompt, TASK

def format_task(task):
    l = task.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def compute_acc(run_results):
    acc_result_list = []
    for prompt in run_results:
        acc = 0
        pred_answers = run_results[prompt]['pred_answers']
        gold_answers = run_results[prompt]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        #result of every prompt was in list
        acc_result_list.append(acc)
    return acc_result_list    

def get_prompt(benchmark, prompt_dics, model_name):
    prompt_output_path = os.path.join('instruction', f'{benchmark}', 'round_best', model_name)
    if not os.path.exists(prompt_output_path):
        os.makedirs(prompt_output_path)

    for task in prompt_dics:
        prompt_list = []
        round_idx = prompt_dics[task]['round_idx']
        max_idx = prompt_dics[task]['max_idx']
        #find best prompt
        if round_idx == 'maunal':
            if benchmark == 'agieval' and task in TASK['agieval_en']:
                prompt = manual_prompt['agieval']['en'][0]
            elif args.benchmark == 'agieval' and task in TASK['agieval_zh']:
                prompt = manual_prompt['agieval']['zh'][0]
            else:
                prompt = manual_prompt['mmlu'][0].replace('<task>', format_task(task)) 

        else:
            if round_idx == '0':
                prompt_input_path = os.path.join('instruction', f'{benchmark}', f'round_{0}', f'{task}.json')
                
            else:
                prompt_input_path = os.path.join('instruction', f'{benchmark}', f'round_{round_idx}', model_name, f"{task}.json")
            with open(prompt_input_path, 'r') as file:
                json_data = file.read() 
            # Parse the JSON data
            instructions = json.loads(json_data)
            prompt = instructions[max_idx]   
        prompt_list.append(prompt)
        json_data = json.dumps(prompt_list, ensure_ascii=False)
        with open(os.path.join(prompt_output_path, f'{task}.json'), 'w') as file:
            file.write(json_data)
        prompt_dics[task]['prompt'] = prompt
    with open(os.path.join(prompt_output_path, 'best_prompt_data.json'), 'w', encoding='utf-8') as file:
        json.dump(prompt_dics, file, ensure_ascii=False, indent=2) 
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='moss')
    parser.add_argument('--param_size', type=str, default='7')
    parser.add_argument('--round', type=str, default='1', help='The result corresponding to the current round of prompt')
    parser.add_argument('--benchmark', type=str, default='agieval')
    args = parser.parse_args()
    model_name = f"{args.model_type}_{args.param_size}b" 
    
    history_lists = []
    for i in range(int(args.round) + 1):
        history_lists.append(glob.glob(os.path.join('outputs', f'{args.benchmark}', model_name, f'round_{i}') + '/*.json'))
  
    result_lists = []
    total_max = 0
    total_mean = 0
    total_min = 0
    total_num = 0
    manual_total_acc = 0
    manual_total_num = 0
    prompt_dics = {}
    task_results = {}

    for num, file in enumerate(history_lists[0]):
        prompt_dic = {}
        task_result = {}
        #calculate manual prompt acc value
        manual_task_acc = 0
        subject = file.split(f'run_{args.benchmark}_')[1].replace('.json',"")
        manual_file = os.path.join('outputs', f'{args.benchmark}', model_name, f'round_manual', f'run_{args.benchmark}_{subject}.json')
        with open(manual_file, 'r') as f:
            manual_results = json.load(f) 
        pred_answers = manual_results['prompt0']['pred_answers']
        gold_answers = manual_results['prompt0']['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: manual_task_acc += 1
        manual_total_acc += manual_task_acc
        manual_total_num += len(gold_answers)
        task_num = len(gold_answers)
        manual = float('%.4f' % (manual_task_acc / task_num))
    
        acc_result_list = []
        length_list = []
        #calculate round_n task prompt acc value    
        for _, history_list in enumerate(history_lists):
            with open(history_list[num], 'r') as f:
                run_results = json.load(f)
                temp_list = compute_acc(run_results)
                length_list.append(len(temp_list))
                acc_result_list = acc_result_list + temp_list
        list_length = len(acc_result_list)

        #find the idx of the max acc value
        task_max = max(acc_result_list)
        max_idx_list = [index for index, result in enumerate(acc_result_list) if result == task_max]
        max_idx = max_idx_list[-1]
        task_min = min(acc_result_list)
        task_mean = sum(acc_result_list)/len(acc_result_list)
        #find the round_idx of the max acc value to indentify the best prompt
        for id, length in enumerate(length_list):
            if max_idx < length:
                round_idx = str(id)
                break
            else:
                max_idx = max_idx - length
                
        task_acc_max = float('%.4f' % (task_max/task_num))
        task_acc_mean = float('%.4f' % (task_mean/task_num))
        task_acc_min = float('%.4f' % (task_min/task_num))
        delta = task_acc_max-manual
        if delta < 0:
            delta = 0
            task_max = manual_task_acc
            task_acc_max = manual
            round_idx = 'maunal'
            max_idx = 0
        
        total_max += task_max
        total_min += task_min
        total_mean += task_mean
        total_num += task_num
        
        task_result['max'] = task_max
        task_result['min'] = task_min
        task_result['mean'] = task_mean
        task_result['num'] = task_num
        task_result['manual'] = manual_task_acc
        task_results[subject] = task_result
        
        #Calculate numerical indicators for each subtask
        result_list = [subject, task_acc_max, task_acc_min,
                task_acc_mean, manual, delta]
        result_lists.append(result_list)
        prompt_dic['round_idx'] = round_idx
        prompt_dic['max_idx'] = max_idx
        prompt_dic['acc'] = task_acc_max
        prompt_dics[subject] = prompt_dic
        
    df = pd.DataFrame(result_lists, columns=["task", "max", "min","mean","manual","delta"]).reset_index(drop=True)
    df = df.sort_values('delta',ascending=False).reset_index(drop=True)
    total_acc_max = float('%.4f' % (total_max/total_num))
    total_acc_mean = float('%.4f' % (total_mean/total_num))
    total_acc_min = float('%.4f' % (total_min/total_num))
    manual = float('%.4f' % (manual_total_acc/manual_total_num))
    
    df.loc[len(df)]=['avg', total_acc_max, total_acc_min, total_acc_mean, manual, total_acc_max-manual]
    print(df)
     
    if args.benchmark == 'mmlu':
        subcategories_dic = {}
        for subject in subcategories:
            subcategory = subcategories[subject][0] 
            try:
                subcategories_dic[subcategory].append(subject)
            except:
                subcategories_dic[subcategory]=[]
                subcategories_dic[subcategory].append(subject)
        
        for subcategory in subcategories_dic:
            temp_dic = {'max':0,'min':0,'mean':0,'num':0,'manual':0}
            for subject in subcategories_dic[subcategory]:
                temp_dic['max'] += task_results[subject]['max']
                temp_dic['min'] += task_results[subject]['min']    
                temp_dic['mean'] += task_results[subject]['mean']    
                temp_dic['num'] += task_results[subject]['num']    
                temp_dic['manual'] += task_results[subject]['manual']
            subcategories_dic[subcategory] = temp_dic
        
        categories_dic = {}
        for category in categories[args.benchmark]:
            temp_dic = {'max':0,'min':0,'mean':0,'num':0,'manual':0}
            for subcategory in categories[args.benchmark][category]:
                temp_dic['max'] += subcategories_dic[subcategory]['max']
                temp_dic['min'] += subcategories_dic[subcategory]['min']    
                temp_dic['mean'] += subcategories_dic[subcategory]['mean']    
                temp_dic['num'] += subcategories_dic[subcategory]['num']    
                temp_dic['manual'] += subcategories_dic[subcategory]['manual']
            categories_dic[category] = temp_dic
        df1 = pd.DataFrame(categories_dic).T.reset_index()

        df1.columns = ['task', 'max', 'min', 'mean', 'num', 'manual']
        df1['max'] = df1['max'] / df1['num']
        df1['min'] = df1['min'] / df1['num']
        df1['mean'] = df1['mean'] / df1['num']
        df1['manual'] = df1['manual'] / df1['num']
        df1['delta'] = df1['max'] - df1['manual']
        df1 = df1.sort_values('delta',ascending=False).reset_index(drop=True)
        print(df1)
        
    elif args.benchmark == 'agieval':
        categories_dic = {}
        for category in categories[args.benchmark]:
            temp_dic = {'max':0,'min':0,'mean':0,'num':0,'manual':0}
            for subject in categories[args.benchmark][category]:
                temp_dic['max'] += task_results[subject]['max']
                temp_dic['min'] += task_results[subject]['min']    
                temp_dic['mean'] += task_results[subject]['mean']    
                temp_dic['num'] += task_results[subject]['num']    
                temp_dic['manual'] += task_results[subject]['manual']
            categories_dic[category] = temp_dic
        #print(subcategories_dic)        
        df1 = pd.DataFrame(categories_dic).T.reset_index()

        df1.columns = ['task', 'max', 'min', 'mean', 'num', 'manual']
        df1['max'] = df1['max'] / df1['num']
        df1['min'] = df1['min'] / df1['num']
        df1['mean'] = df1['mean'] / df1['num']
        df1['manual'] = df1['manual'] / df1['num']
        df1['delta'] = df1['max'] - df1['manual']
        df1 = df1.sort_values('delta',ascending=False).reset_index(drop=True)
        print(df1)
        #find and save the optimizing prompts
    get_prompt(args.benchmark, prompt_dics, model_name)
            
    