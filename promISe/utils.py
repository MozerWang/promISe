import json
import ast

choices = ["A", "B", "C", "D"]

def format_task(task):
    l = task.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def compute_metric(output_filename):
    with open(output_filename, 'r',encoding='utf8') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    acc_result = ""
    acc_result_list = []
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        acc_result += "ACC-%s: %.4f\n" % (task, acc/len(gold_answers))
        acc_result_list.append(acc/len(gold_answers))
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    acc_result += "ACC-max: %.4f\n" % (max(acc_result_list))
    acc_result += "ACC-min: %.4f\n" % (min(acc_result_list))
    acc_result += "ACC-all: %.4f\n" % (total_acc/total_num)
    print("ACC-max: %.4f" % (max(acc_result_list)))
    print("ACC-min: %.4f" % (min(acc_result_list)))
    print("ACC-all: %.4f" % (total_acc/total_num))
    return acc_result

def mmlu_format_example(df, idx, fewshot_part, include_answer=True):
    question = df.iloc[idx, 0]
    fewshot_part = fewshot_part.replace('<question>',question)
    
    options = ""
    k = df.shape[1] - 2
    for j in range(k):
        options += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    fewshot_part = fewshot_part.replace('<options>',options.lstrip('\n'))
    
    answer=''
    if include_answer:
        answer = "{}\n\n".format(df.iloc[idx, k + 1])
        fewshot_part = fewshot_part.replace('<answer>', answer)
    else:
        fewshot_part = fewshot_part.replace('<answer>', answer)
        if fewshot_part[-1] == ' ':
            fewshot_part = fewshot_part.strip()
    return fewshot_part

def agieval_format_example(con, idx, fewshot_part, en, skip_passage=False, include_answer=True):
    passage = con["passage"] if con["passage"] is not None and not skip_passage else ""
    question = con["question"]
    options = con["options"] if con["options"] is not None else ""
    label = con["label"] if con["label"] is not None else ""
    if en:
        
        fewshot_part = fewshot_part.replace('<id>',f"Problem {idx+1}   ")
        fewshot_part = fewshot_part.replace('<passage>',passage)
        fewshot_part = fewshot_part.replace('<question>',question)
        fewshot_part = fewshot_part.replace('<options>'," ".join(options))
        if include_answer:
            fewshot_part = fewshot_part.replace('<answer>',f"{label}")
        else:
            fewshot_part = fewshot_part.replace('<answer>',"").strip(" ")
    else :
        fewshot_part = fewshot_part.replace('<序号>',f"问题 {idx+1}   ")
        fewshot_part = fewshot_part.replace('<材料>',passage)
        fewshot_part = fewshot_part.replace('<问题>',question)
        fewshot_part = fewshot_part.replace('<选项>'," ".join(options))
        if include_answer:
            fewshot_part = fewshot_part.replace('<答案>',f"{label}")
        else:
            fewshot_part = fewshot_part.replace('<答案>',"").strip(" ")
    return fewshot_part

def mmlu_gen_prompt(train_df, prompt, k=-1):
    fewshot_prompt = ''
    # Split the prompt
    start_part = prompt.split('<question>')[0]
    end_part = prompt.split('<answer>')[1]
    if bool(end_part):
        fewshot_part = prompt.split(start_part)[1].split(end_part)[0]
    else:
        fewshot_part = prompt.split(start_part)[1]
    # Construct fewshot part of prompt
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        fewshot_prompt += mmlu_format_example(train_df, i, fewshot_part)
    prompt = start_part + fewshot_prompt.rstrip('\n') + end_part
    return prompt, fewshot_part

def agieval_gen_prompt(instruction, tokenizer, train_df, task_name, en, skip_passage, end_of_example="<END>\n"):
    # Split the prompt
    if en:
        start_part = instruction.split('<id>')[0]
        end_part = instruction.split('<answer>')[1]
        if bool(end_part):
            fewshot_part = ('<id>' + "<id>".join(instruction.split('<id>')[1:])).split('<answer>')[0] + '<answer>'
        else:
            fewshot_part = '<id>' + "<id>".join(instruction.split('<id>')[1:]) 
    else:
        start_part = instruction.split('<序号>')[0]
        end_part = instruction.split('<答案>')[1]
        if bool(end_part):
            fewshot_part = ('<序号>' + "<序号>".join(instruction.split('<序号>')[1:])).split('<答案>')[0] + '<答案>'
        else:
            fewshot_part = '<序号>' + "<序号>".join(instruction.split('<序号>')[1:]) 
    
    prompt = start_part
    demostrations = []
    contexts = [] 
    prompt_num = 0
    for line in list(train_df[task_name]):
        if line:
            # print(line)
            contexts.append(ast.literal_eval(line))
    for idx, con in enumerate(contexts):
        demostrations.append(agieval_format_example(con, idx, fewshot_part, en, skip_passage)) 
    for i in range(len(demostrations)):
        prompt += demostrations[i] + '\n' + end_of_example
    prompt_num = len(demostrations)
    return prompt, prompt_num, fewshot_part

def mmlu_prompt_construct(num, instruction, tokenizer, fewshot_num, dev_df, test_df):
    print(f'prompt{num}')
    print("="*100)
    print(instruction)
    records = []
    train_prompt, fewshot_part = mmlu_gen_prompt(dev_df, instruction, fewshot_num)
    print("="*100)
    print(fewshot_part)
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        inference_prompt = mmlu_format_example(test_df, i, fewshot_part, include_answer=False)
        prompt = train_prompt + '\n\n' + inference_prompt
        
        while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
            prompt_split = prompt.split("\n\n")
            prompt_split.pop(1)
            prompt = '\n\n'.join(prompt_split)
        
        label = test_df.iloc[i, test_df.shape[1]-1]
        records.append({'prompt':prompt, 'answer':label})
    return records

def agieval_prompt_construct(num:int, instruction:str, tokenizer, prompt_df, task_name, en, test_json, skip_passage=False):
    print(f'prompt{num}')
    print("="*100)
    print(instruction)
    train_prompt, shot_num, fewshot_part = agieval_gen_prompt(instruction, tokenizer, prompt_df, task_name, en, skip_passage)
    print("="*100)
    print(fewshot_part)
    records = []
    for i,line in enumerate(test_json):
        # get prompt and make sure it fits
        prompt_end = agieval_format_example(line, shot_num, fewshot_part, en, skip_passage, include_answer=False)
        prompt = train_prompt
        while len(tokenizer.tokenize(prompt + prompt_end)) + 1> 2048: # bos token
            prompt_split = prompt.split("<END>\n")
            if len(prompt_split)==2:
                prompt = '<END>\n'.join(prompt_split)
                break
            else:
                prompt_split.pop(1)
                prompt = '<END>\n'.join(prompt_split)
                shot_num -= 1
        prompt += agieval_format_example(line, shot_num, fewshot_part,en, skip_passage, include_answer=False)
        #print(len(tokenizer.tokenize(prompt)))
        label = line["label"]
        records.append({'prompt':prompt, 'answer':label})
    return records


def read_jsonl(path):
    with open(path, encoding='utf8') as fh:
        results = []
        for line in fh:
            if line is None:
                continue
            try:
                results.append(json.loads(line) if line != "null" else line)
            except Exception as e:
                print(e)
                print(path)
                print(line)
                raise e
    return results

