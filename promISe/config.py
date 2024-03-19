OPENAI_KEY = "Your OpenAI API Key Here"

PROXY = "http://127.0.0.1:7890"

DATA_PATH = {'mmlu': "mmlu/data/",
             'agieval':'agieval/data/'}

AGIEVAL_FEWSHOT = "agieval/data/few_shot_prompts.csv"

TASK = {'mmlu':[
            'abstract_algebra',
            'anatomy',
            'astronomy',
            'business_ethics',
            'clinical_knowledge',
            'college_biology',
            'college_chemistry',
            'college_computer_science',
            'college_mathematics',
            'college_medicine',
            'college_physics',
            'computer_security',
            'conceptual_physics',
            'econometrics',
            'electrical_engineering',
            'elementary_mathematics',
            'formal_logic',
            'global_facts',
            'high_school_biology',
            'high_school_chemistry',
            'high_school_computer_science',
            'high_school_european_history',
            'high_school_geography',
            'high_school_government_and_politics',
            'high_school_macroeconomics',
            'high_school_mathematics',
            'high_school_microeconomics',
            'high_school_physics',
            'high_school_psychology',
            'high_school_statistics',
            'high_school_us_history',
            'high_school_world_history',
            'human_aging',
            'human_sexuality',
            'international_law',
            'jurisprudence',
            'logical_fallacies',
            'machine_learning',
            'management',
            'marketing',
            'medical_genetics',
            'miscellaneous',
            'moral_disputes',
            'moral_scenarios',
            'nutrition',
            'philosophy',
            'prehistory',
            'professional_accounting',
            'professional_law',
            'professional_medicine',
            'professional_psychology',
            'public_relations',
            'security_studies',
            'sociology',
            'us_foreign_policy',
            'virology',
            'world_religions'
        ],
        'agieval':[
            "lsat-ar", 
            "lsat-lr", 
            "lsat-rc", 
            "logiqa-en", 
            "sat-math", 
            "sat-en", 
            "aqua-rat",
            "sat-en-without-passage", 
            "gaokao-english",
            "logiqa-zh", 
            "gaokao-chinese", 
            "gaokao-geography", 
            "gaokao-history",
            "gaokao-biology", 
            "gaokao-chemistry", 
            "gaokao-mathqa"
        ],
        'agieval_name': {
            "lsat-ar":"LSAT Analytical Reasoning", 
            "lsat-lr":"LSAT Logic Reasoning", 
            "lsat-rc":"LAST Reading Comprehension", 
            "logiqa-en":"Civil Service Exam", 
            "sat-math":"SAT Math", 
            "sat-en":"SAT", 
            "aqua-rat":"GRE & GMAT Math",
            "sat-en-without-passage":"SAT", 
            "gaokao-english": "GAOKAO English",
            "logiqa-zh":"公务员考试", 
            "gaokao-chinese":"高考语文", 
            "gaokao-geography":"高考地理", 
            "gaokao-history":"高考历史",
            "gaokao-biology":"高考生物", 
            "gaokao-chemistry":"高考化学", 
            "gaokao-mathqa":"高考数学"
        },
        'agieval_en':["lsat-ar", "lsat-lr", "lsat-rc", "logiqa-en", "sat-math", "sat-en", "aqua-rat",
                    "sat-en-without-passage", "gaokao-english"],
        'agieval_zh':["logiqa-zh", "jec-qa-kd", "jec-qa-ca", "gaokao-chinese", "gaokao-geography", "gaokao-history",
                    "gaokao-biology", "gaokao-chemistry", "gaokao-physics", "gaokao-mathqa"]

        }

search_instructions = {
    'mmlu': {'en':{
        'system_content_generate': "You are a professional Instruction Editor. \n Your objective is to rewrite the given instruction into a more appropriate version to make those famous large language models have better performance at few-shot inference. \nYou SHOULD supply more background information about the given subject. You SHOULD try your best to optimize the formation of given instruction. You SHOULD NOT change <question> <options> and <answer>. You SHOULD NOT extend <question> <options> and <answer>.",
        
        'system_content_introspect': "You are a professional Instruction Editor. \nBelow are the results of the different instructions for completing multiple-choice questions in [task]. The results are in JSON format, where the key is the instruction and the value is its corresponding score. The better the instruction, the higher the score.\nYou SHOULD analyze the results (i.e. what is good and what is bad) in order to guide further refine of instruction. Your analysis SHOULD focus on grouped characteristics and avoid repeating the results. Your analysis SHOULD be brief and precise. You SHOULD NOT write a refined or reversed instruction. ",
        
        'system_content_refine': "You are a professional Instruction Editor. \nBased on analysis and three initial instructions, write one instruction for completing multiple-choice questions in [task]. \nYou MUST keep <question> <options> and <answer> in the refined instruction.",
        
        'extensive': "#Given Instruction:\n The following are multiple choice questions (with answers) about [task].\n\n<question>\n<options>\nAnswer: <answer>\n\n#Rewrite Instruction:"}
    },
 'agieval': {
        'en':{
            'system_content_generate': "You are a professional Instruction Editor.\nYour objective is to rewrite the given instruction into a more appropriate version to make those famous large language models have better performance at few-shot inference.\nYou SHOULD supply more background information about the given subject. You SHOULD try your best to optimize the formation of given instruction. You SHOULD NOT change and extend <id> <passage> <question> <options> and <answer>.",
            
            'system_content_introspect': "Your are a professional Instruction Editor. \nBelow are the results of the different instructions for completing multiple-choice questions in [task]. The results are in JSON format, where the key is the instruction and the value is its corresponding score. The better the instruction, the higher the score.\nYou SHOULD analyze the results (i.e. what is good and what is bad) in order to guide further refine of instruction. Your analysis SHOULD focus on grouped characteristics and avoid repeating the results. Your analysis SHOULD be brief and precise. You SHOULD NOT write a refined or reversed instruction. ",
            
            'system_content_refine': "Your are a professional Instruction Editor. \nBased on analysis and three initial instructions, write one instruction for completing multiple-choice questions in [task]. \nYou MUST keep <id> <passage> <question> <options> and <answer> in the refined instruction.",
                        
            'extensive': "#Given Instruction:\nHere are the answers for the problems in the exam.\n<id>.   <passage> <question>\nChoose from the following options:    <options>\nThe answer is therefore <answer>\n#Subject:\n<task>\n#Rewrite Instruction:"},
        'zh':{
            'system_content_generate': "你是一位资深的指令编写专家。\n你的目标是将初始的指令重写为更合适的指令，使大规模预训练语言模型在推理任务中有更好的表现。\n你需要给出考试科目相关的背景信息。你需要改进指令的格式。你不可以修改或扩写这五个占位符： <序号> <材料> <问题> <选项> <答案>。",
            
            'system_content_introspect': "你是一位资深的指令编写专家。\n以下是用于完成《任务》多项选择题的不同指令的结果。这份结果是JSON格式，其中键key是指令，值value是该指令对应的分数。一个指令越好，它对应的得分也就越高。\n为了进一步优化这些指令，提升指令效果，你需要仔细分析结果（比如根据得分，分析这些指令的优点和缺点）。你对结果的分析应该侧重于分组特征并且要避免复述指令结果。你的分析要简短准确。你不需要写出改进后的或者意思相反的指令。",
            
            'system_content_refine': "你是一位资深的指令编写专家。 基于对指令的分析以及三条初始指令，你需要重新写一条指令用于完成《任务》\n。你要保留这五个占位符： <序号> <材料> <问题> <选项> <答案> ，对这五个占位符不做任何更改。",
            
            'extensive': "#初始的指令：\n以下是考试中各个问题的答案。\n<序号>.   <材料> <问题>\n从以下选项中选择:    <选项>\n答案是 <答案>\n#本次考试的科目：\n《任务》\n#重写后的指令："}
    }
} 

manual_prompt = {
    'mmlu':["The following are multiple choice questions (with answers) about <task>.\n\n<question>\n<options>\nAnswer: <answer>"],
    'agieval':{'en':["Here are the answers for the problems in the exam.\n<id>.   <passage> <question>\nChoose from the following options:    <options>\nThe answer is therefore <answer>"],
               'zh':["以下是考试中各个问题的答案。\n<序号>.   <材料> <问题>\n从以下选项中选择:    <选项>\n答案是 <答案>"]}
}

subcategories = {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"]
}

categories = {
    'mmlu':{
        "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
        "other (business, health, misc.)": ["other", "business", "health"] 
    },
    'agieval':{
        "LSAT": ["lsat-ar", "lsat-lr", "lsat-rc"],
        "GAOKAO&SAT": ["sat-math", "sat-en", "sat-en-without-passage", "gaokao-english", "gaokao-chinese", "gaokao-geography", "gaokao-history","gaokao-biology", "gaokao-chemistry", "gaokao-mathqa"],
        "CSE": ["logiqa-en", "logiqa-zh"],
        "GRE&GMAT": ["aqua-rat"],
    }
}
