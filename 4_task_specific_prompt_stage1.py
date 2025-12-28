import random
random.seed(42)
import os
import json
import re
import string
from utils import load_line_json, save_line_json
from multiprocessing import Pool
from functools import partial
import numpy as np
from tqdm import tqdm, trange
import torch
import argparse
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from sklearn.cluster import KMeans
import math
import pandas as pd


output_dir = './machine_generated_instr/'



attrs_to_id = {}


clustering_files_path = "./clustering/data/clustering/affinity_propagation_clusters_p=0.5.json"
with open(clustering_files_path, 'r', encoding='utf-8-sig') as f:
    data = json.load(f)



with open(os.path.join(output_dir, "2_attrs_infomation.json"), 'r', encoding='utf-8-sig') as f:
    attrs_with_most_three_descr = json.load(f)


attr_list_path = "./clustering/data/attr_list.json"

with open(attr_list_path, 'r', encoding='utf-8-sig') as f:
    attr_list = json.load(f)


from collections import defaultdict
attr_map = {}

for item in attr_list:
        attr_map[item['attr']] = item['attr_names']



for item in data:

    k_attrs = [ attr.split('|')[0] for attr in item['attr_names']]

    item['five_most_common_attr_decs'] = [ attrs_with_most_three_descr[attr_map[attrs][0]]['most_common_decs'] for attrs in k_attrs[:1]]



instruction_v2 = """I want you to focus on the following text attribute: **{}({})**, and systematically generate a diverse range of tasks that target a single text. Please make sure each task includes the following elements:
    - Task Name: a concise title that captures the core goal or theme of the task.
    - Task Description: an explanation of the problem this task aims to solve or the objective it aims to achieve.

The task types should be diverse, such as: 
1. Classification
    - Closed-set categories classification
    - Open-ended categories classification
2. Scoring or Rating
    - Quantitative scales
3. Information Extraction
    - Keywords, key sentences, triggers
    - Root causes, contextual dependencies, and more
4. Structured Output
    - JSON, tables, or other machine-readable formats
    - Potentially includes multiple fields (roles, attribute values, etc.)

When designing these tasks, please follow these guidelines:
    - Clarity: Each task's goal should be described methodically.
    - Diversity: Aim for a wide range of creative ideas across classification, scoring, extraction, and extended analyses.
    - All tasks must target a single text. Therefore, do not generate tasks involving comparisons between two texts.
Based on the above requirements, please list several diverse tasks focused on **{}**.
Present your output in the following structured JSON format, ensuring that it can be directly parsed.
[
    {{
        "task_name": "Task Name",
        "task_description": "Task Description"
    }}, 
    ...
]"""



for cluster in data:
    attrs = [ item.split('|')[0] for item in cluster['attr_names']]

    cluster['instruction'] =  instruction_v2.format(attrs[0], cluster['five_most_common_attr_decs'][0][:-1].lower(), attrs[0])


print(data[1]['instruction'])

save_line_json(data, os.path.join(output_dir, "6_instrution_for_generating_task_prompt_stage1_affinity.json"))
