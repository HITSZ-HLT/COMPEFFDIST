import sys

from utils import load_line_json, save_line_json
import json
import os
import random
random.seed(42)
from tqdm import tqdm
import re




def extract_prompts(text):

    # 编译正则表达式：匹配以数字加句点开头的行，并提取其内容，直到遇到下一个 prompt 或分隔符或文本结束
    pattern = re.compile(r'(?ms)^\d+\.\s*(.*?)(?=^\d+\.|\n|$)')

    # 查找所有匹配项
    prompts = pattern.findall(text)

    # 输出提取的 prompt 内容（去除首尾空白字符）
    results = []
    for i, prompt in enumerate(prompts, start=1):
        if ':' in prompt:
            results.append(prompt.split(':')[1].strip())
        else:
            
            results.append(prompt.strip())

    return results




data_dir = "./machine_generated_instr/"

file_name = "5_prompt_generated_by_attr_cluster.json"

path = os.path.join(data_dir, file_name)

data = load_line_json(path)



for item in data:
    item['prompts'] = extract_prompts(item['output'])


a = random.sample(data, 10)

for item in a:
    print(item['attr_names'])
    for p in item['prompts']:
        print(p)
    print()


save_line_json(data, path.replace('.json', '_with_prompts.json'))