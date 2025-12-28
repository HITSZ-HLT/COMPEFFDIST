import random
random.seed(42)
import os
import json
import re
import string
from utils import load_line_json, save_line_json
import numpy as np
from tqdm import tqdm, trange
import argparse
from collections import Counter
import re
import ast
import math
import pandas as pd


output_dir = './machine_generated_instr/'

# 对于task prompt 阶段二生成的具体任务描述和要求文本进行抽取
# 得到cluster_id -> [(task1 name, task1 descr, task1 example), ....  ]



ori_path = os.path.join(output_dir, "7_instrution_for_generating_task_prompt_stage2_affinity.json")

ori_data = load_line_json(ori_path)


path = os.path.join(output_dir, "result_7_instrution_for_generating_task_prompt_stage2_affinity.json")
data = load_line_json(path)


for item in ori_data:
    for item2 in data:
        if item['instruction'] == item2['instruction']:
            item['output'] = item2['response']
            break



def extract_pairs(text: str):


    pattern = re.compile(
        r'[\+\*\-]\s*Task Input:?\s*(?P<input>.*?)\s*'
        r'[\+\*\-]\s*Task Output:?\s*(?P<output>.*?)'
        r'(?=\n*\s*[\+\*\-]\s*Task Input:|\n*\s*\* Example|\n*\s*Example|\n*\s*\-\s*Example|\Z)',
        flags=re.DOTALL
    )


    results = []
    for m in pattern.finditer(text):
        inp = m.group('input').strip()
        raw_out = m.group('output').strip()
        results.append({'input': inp, 'output': raw_out})
    return results



def norm_text(text):
    text = text.replace('**', '')
    text = text.replace('```json', '')
    text = text.replace('```', '')
    text = text.strip()

    return text
    

def parse_instance(text):
    pattern = re.compile(
        r'(?s)'
        r'(.*?)'  # 组1：第一部分（标题前内容）
        r'\*\*1\. Task Name\*\*(.*?)(?=\*\*2\. Task Description\*\*)'  # 组2：Task Name 内容
        r'\*\*2\. Task Description\*\*(.*?)(?=\*\*(?:3\. )?Task Examples:?\*\*)'  # 组3：Task Description 内容
        r'\*\*(?:3\. )?Task Examples:?\*\*(.*)',  # 组4：Task Examples 内容
        flags=re.DOTALL
    )


    match = pattern.match(text)
    if not match:
        print(text)
        raise ValueError("文本格式不符合预期")

    part1, part2, part3, part4 = match.groups()

    return norm_text(part1), norm_text(part2), norm_text(part3), norm_text(part4)



from collections import defaultdict

cluster_id_2task_prompt = defaultdict(list)


for item in ori_data:
    cluster_id = item['cluster_id']
    _, task_name, task_description, part4 = parse_instance(item['output'])


    if 'text 1' in part4.lower() and 'text 2' in part4.lower():
        continue
    if 'text1' in part4.lower() and 'text2' in part4.lower():
        continue
    if 'text a' in part4.lower() and 'text b' in part4.lower():
        continue
    if 'plot 1' in part4.lower() and 'plot 2' in part4.lower():
        continue
    if 'review 1' in part4.lower() and 'review 2' in part4.lower():
        continue

    demos = []

    for item in extract_pairs(part4):
        # 处理input
        input = item['input']
        new_input = []
        for aa in input.split('\n'):
            if aa.strip().startswith('+'):
                aa = aa.replace('+', '').strip()
            new_input.append(aa)
        input = '\n'.join(new_input)
        input = input.replace('`','')

        new_input = []
        for aa in input.split('\n'):
            if aa.strip().startswith('-'):
                aa = aa.replace('-', '').strip()
            new_input.append(aa)
        input = '\n'.join(new_input)

        # 处理output
        output = item['output']
        
        if '+' in output:
            new_output = []
            for aa in output.split('\n'):
                if aa.strip().startswith('+'):
                    aa = aa.replace('+', '').strip()
                new_output.append(aa)
            try:
                types = type(ast.literal_eval('\n'.join(new_output)))
                if types in (dict ,list):
                    output = output.replace('+', '')
            except:
                output = '\n'.join(new_output)




        if output.startswith('"'):
            output = output[1:-1]
        output = output.replace('`','')

        demos.append({'input':input, 'output':output})

    cluster_id_2task_prompt[cluster_id].append((task_name, task_description, demos[:7]))




path = os.path.join(output_dir, "8_clusterid_2task_prompts_affinity.json")


with open(path, 'w', encoding='utf-8-sig') as f:
    json.dump(cluster_id_2task_prompt, f, indent = 4)