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
import ast
import pandas as pd


output_dir = './machine_generated_instr/'

# 任务特定prompt stage2 生成


import re


# 从stage1生成的回答提取任务描述对列表 stage1_response_per_cluster -> [(task name1, decs1), .... ]
def parse_task_prompt_stage1(text):

    try:
        pairs = ast.literal_eval(text)
       
    except:
        text1 = text[text.find('['):text.rfind(']') + 1]
        try:
            pairs = ast.literal_eval(text1)
        except:
            pairs = []

    assert len(pairs) != 0

    task_desc_pairs = []

    for item in pairs:
        task, description = item['task_name'], item['task_description']

        task_desc_pairs.append(
            [task.strip(), description.strip()]
        )
        # print(task.strip())
        # print(description.strip())
        # print("-----")

    return task_desc_pairs



instruction_stage2_v2 = """Please rewrite the task based on the task name and description, making the task definition more standardized and normalized.

Task Name: {}
Task Description: {}

Below are the specific requirements and guidelines: 
1. Avoiding Ambiguity: Ensure task description, requirement and constraint is precise, complete, and free of ambiguity. If the task contains two direction, specify one direction in the task description and requirments and you should NOT add any requirments in input. 

2. Ensure the rewritten task is consistent with the original task description.

3. Task Elements: Ensure that each task definition includes the following components:
    - Task Name: A concise title of the task.

    - Task Description: A detailed explanation of the task and should contain the following parts: 
        - Explicitly specifying the expected output format and requirements (e.g., classification label, numerical score, structured JSON, Python list).
        - If the task is a classification task or contains classification task as subtask, for closed-set classification, you should explicitly list all allowed labels. For open-set classification, you should instruct the model to infer the appropriate labels from the input.
        - If the task is a annotation/extraction task, you should specify whether the extracted or annotated text must exactly match the original text or if modifications are allowed.
        - If the task requires structured output, specify the exact structure (for example, a JSON schema or Python list format) and enumerate all required fields.
    - Task Examples: You should provide at least EIGHT concrete examples, each including:
        - Task Input: Formatted according to the input specifications.
        - Task Output: Formatted according to the output specifications.

You should output you rewritten task in the following format:
1. Task Name
2. Task Description
3. Task Examples
    - Example1
        - Task Input
        - Task Output"""



# task_name = "Emotion Intensity Classification"
# task_description = "Classify the emotional intensity of a given text into one of the following categories: Low, Moderate, High, or Extremely High."

# task_name = "Emotion Trigger Extraction"
# task_description = "Identify and extract the specific words or phrases that trigger the emotional intensity in a given text."

task_description = "Rewrite a text to strengthen or weaken the sentiment polarity while maintaining the original meaning. The goal is to amplify or diminish the emotional tone of the text."
task_name = "Sentiment Polarity Strengthening/Weakening"

print(instruction_stage2_v2.format(task_name, task_description))



path = "./machine_generated_instr/6_instrution_for_generating_task_prompt_stage1_affinity.json"

data = load_line_json(path)




path = os.path.join(output_dir, "result_6_instrution_for_generating_task_prompt_stage1_affinity.json")

results = load_line_json(path)

for item in data:
    for result in results:
        if item['instruction'] == result['instruction']:
            item['llm_response'] = result['response']
            break

all_input = []

all_pairs = []

for item in data:
    cluster_id = item['cluster_id']
    for item in parse_task_prompt_stage1(item['llm_response']):
        # if 'baseline' in item[1].lower():
        #     continue
        all_input.append(
            {
                "cluster_id":cluster_id,
                "instruction":instruction_stage2_v2.format(*item)
            }
        )

        all_pairs.append(item)

print(all_input[759]['instruction'])

print(len(all_input))


path = os.path.join(output_dir, "7_instrution_for_generating_task_prompt_stage2_affinity.json")

save_line_json(all_input, path)