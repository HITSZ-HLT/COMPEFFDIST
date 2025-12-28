from utils import save_line_json, load_line_json, mkdir_if_not_exist
import json

instr_demo_gen_part1 = """**Generate two instances for the following task. The text part in the samples needs to refer to the style, vocabulary, and themes in the Reference Texts. Carefully read the task description to ensure the correct labeling in the generated samples.**

**Reference Texts:**

{}

{}

**Task Description:**

{}

**Give your respone in the following format: **
"""



instr_demo_gen_part2= """
Input: {}
Output: {}
"""

import random
random.seed(42)
path = "./machine_generated_instr/8_clusterid_2task_prompts_affinity.json"


with open(path, 'r', encoding='utf-8-sig') as f:
    clusterid_2task_prompts = json.load(f)


real_world_text_path = "./machine_generated_instr/real_world_text_45k.txt"

real_world_text = []
with open(real_world_text_path, 'r', encoding='utf-8') as f:
    for line in f:
        real_world_text.append(line.strip())

random.shuffle(real_world_text)



all_input_demo_gen = []

from tqdm import tqdm
for cluster_id, clutser_task_prompts in tqdm(clusterid_2task_prompts.items()):


    for i in range(len(clutser_task_prompts)):
        task_name, task_description, demos = clutser_task_prompts[i]
        
        for j in range(16):
            if len(demos) == 1:
                sample_demos = random.sample(demos, 1)
            else:
                sample_demos = random.sample(demos, 2)
            texts = random.sample(real_world_text, 2)
            input_demo_gen_part1 = instr_demo_gen_part1.format(*texts, task_description)
            input_demo_gen_part2 = ''
            for item in sample_demos:
                input_demo_gen_part2 += instr_demo_gen_part2.format(item['input'], item['output'])
        
            input_demo_gen = input_demo_gen_part1 + input_demo_gen_part2

            all_input_demo_gen.append({
                'cluster_id':cluster_id,
                'task_prompt_index':i,
                'sub_index':j,
                'instruction': input_demo_gen.strip()
            })


ouput_path = "./machine_generated_instr/9_instrution_for_generating_task_prompt_stage3_affinity.json"
save_line_json(all_input_demo_gen, ouput_path )