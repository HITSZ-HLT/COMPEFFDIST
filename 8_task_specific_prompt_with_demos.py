
from utils import save_line_json, load_line_json, mkdir_if_not_exist
import json

cluster_task_prompt_information_path = './machine_generated_instr/8_clusterid_2task_prompts_affinity.json'

with open(cluster_task_prompt_information_path, 'r', encoding='utf-8-sig') as f:
    cluster_task_prompt_information = json.load(f)

path = './machine_generated_instr/result_9_instrution_for_generating_task_prompt_stage3_affinity.json'

data = load_line_json(path)



def parse(text):

    pattern = r"(?:\*\*)?Input:(?:\*\*)?\s*(.*?)\s*(?:\*\*)?Output:(?:\*\*)?\s*(.*?)(?=(?:\s*(?:\*\*)?Input:(?:\*\*)?)|$)"

    matches = re.findall(pattern, text, re.DOTALL)

    results = []

    if len(matches) != 2:
        return []


    for idx, (inp, out) in enumerate(matches, start=1):
        input_text = inp.strip()
        # 如果输入内容被双引号包裹，则去除两侧的双引号
        if input_text.startswith('"') and input_text.endswith('"'):
            input_text = input_text[1:-1]
        output_text = out.strip()
        # 如果 Output 部分后附有额外说明（与主体之间有空行分隔），只取第一段
        output_text = output_text.split("\n\n")[0].strip()

        if input_text.startswith('The text is:'):
            input_text = input_text.split('The text is:')[1].strip()

        if input_text.startswith('Analyze the text'):
            input_text = input_text.split('Analyze the text')[1].strip()

            
        if output_text.startswith('{') and not output_text.endswith('}'):
            output_text = output_text + '\n}'
        if output_text.startswith('[') and not output_text.endswith(']'):
            output_text = output_text + '\n]'


        if not ('reference text' in output_text.lower() or 'reference text' in input_text.lower()):
            results.append({'input':input_text, 'output':output_text})

    return results


import re


for item in data:
    results = parse(item['response'])
    item['parsed_demos'] = results

    
    


from collections import defaultdict
cluster_prompt_demos = defaultdict(list)



i = 0
for cluster_id, value in cluster_task_prompt_information.items():
    for index in range(len(value)):
        part = []
        for item in data[i:i+16]:
            part.extend(item['parsed_demos'])
        # print(len(part))

        value[index].append(part)
        i += 16


output_path = './machine_generated_instr/10_clusterid_2task_prompts_with_demos_affinity.json'

with open(output_path, 'w', encoding='utf-8-sig') as f:
    json.dump(cluster_task_prompt_information, f, indent=4)