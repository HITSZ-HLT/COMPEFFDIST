import random
random.seed(42)
import os
import json
import re
import string
from utils import load_line_json, save_line_json, load_json
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



# 找到最常出现的属性描述，如果属性描述出现次数一样多，随机选一个
def get_most_common_decs(decses, n=1):

    if n == 1:

        # 使用 Counter 统计每个文本的出现次数
        text_counter = Counter(decses)

        # 找到出现次数最多的次数
        max_count = max(text_counter.values())

        # 找到所有出现次数等于 max_count 的文本
        most_common_texts = [text for text, count in text_counter.items() if count == max_count]

        # 如果有多条文本出现次数相同，随机抽取一条
        most_common_text = random.choice(most_common_texts)


    else:

        # 使用 Counter 统计每个文本的出现次数
        text_counter = Counter(decses)

        # 找到出现次数最多的次数

        most_common_texts = text_counter.most_common(n) 

        most_common_text = [text for text, count in most_common_texts]

    return most_common_text


 
 
# 去除开头的标点符号，提取第一个英文字母或者数字开始的内容
def extract_text_content(text):
    pattern = r"[A-Za-z0-9].*"
    match = re.search(pattern, text, re.DOTALL)

    return match.group().strip()

# 提取的属性名可能是  attirubute: tone 或者是 tone, 需要处理前一种情况
def extract_attribute(attr):

    if 'attribute:' in attr:
        attr = attr[attr.find('attribute:') + len('attribute:'):].strip()
    return attr
        

# 提取所有的(attribute_name, attribute_description, attribute_value)三元组
def extract_attr_desc_value(text):
    pattern = r"(\d+)\.\s*\*\*(.*?)\*\*.*?Attribute Description(.*?)\s*Attribute Value(.*?)(?=\n\d+\.|$)"
    matches = re.findall(pattern, text, re.DOTALL)

    attr_pair = []

    if not matches:
        return attr_pair

    for match in matches:
        

        
        # attribute_name = attr_norm(extract_attribute(match[1].strip().lower()))
        attribute_name = extract_attribute(match[1].strip().lower())



        attribute_description = extract_text_content(match[2].strip())
        attribute_value = extract_text_content(match[3].strip())

        attr_pair.append((attribute_name, attribute_description, attribute_value))

    return attr_pair


# 为原始数据属性描述和属性值
def add_attr_desc_value(data):
    for item in data:
        item['attr_desc_value'] = extract_attr_desc_value(item['facet_analysis'])


def attr_norm(attr):
    if  ':' in attr:
        attr = attr.split(':')[0].strip()

    return attr


def get_all_attr_triplets(data):


    
    all_triplets = {}
    for item in data:
        for attr_name, attr_desc, attr_value in item['attr_desc_value']:
            # print(attr_name)
            if attr_name not in all_triplets:
                all_triplets[attr_name] = {
                    'attr_desc' : [attr_desc], 
                    'attr_value' : [attr_value],
                    'counts' : 1
                }
            else:
                all_triplets[attr_name]['attr_desc'].append(attr_desc)
                all_triplets[attr_name]['attr_value'].append(attr_value)
                all_triplets[attr_name]['counts'] += 1
                

    for k, v in all_triplets.items():
        attr_decses = v['attr_desc']

        most_common_decs = get_most_common_decs(attr_decses, n=1)

        all_triplets[k]['most_common_decs'] = most_common_decs

        most_three_common_decs = get_most_common_decs(attr_decses, n=3)

        all_triplets[k]['most_three_common_decs'] = most_three_common_decs

    return all_triplets


    


if __name__ == "__main__":

    output_dir = './machine_generated_instr/'

    print("-----------------4. 运行clustering文件夹文件进行聚类，得到聚类结果affinity_propagation_clusters_p=0.5.json-----------------------------------")

    clustering_files_path = "./clustering/data/clustering/affinity_propagation_clusters_p=0.5.json"
    with open(clustering_files_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)


    attrs_to_id = {}

    for item in data:
        cluseter_id = item['cluster_id']
        for attr in item['attr_names']:
            attrs_to_id[attr.split('|')[0]] = cluseter_id


    with open(os.path.join(output_dir, "3_attr_2cluster_id_affinity.json"), 'w', encoding='utf-8-sig') as f:
        json.dump(attrs_to_id, f, indent=4)


    print('-----------------5. 为每个聚类生成分析（开放式问答）任务-----------------------------------')

    with open(os.path.join(output_dir, "2_attrs_infomation.json"), 'r', encoding='utf-8-sig') as f:
        attrs_with_most_three_descr = json.load(f) 

    

    def map_attr_to_original():
        path = "./clustering/data/attr_list.json"
        attr_list = load_json(path)

        from collections import defaultdict
        attr_map = defaultdict(list)

        for item in attr_list:
            attr_map[item['attr']] = item['attr_names']
            if item['attr'] in item['attr_names']:
                attr_map[item['attr']] = [item['attr']]


        return attr_map

    
    attr_map = map_attr_to_original()

    for item in data:

        k_attrs = [ attr.split('|')[0] for attr in item['attr_names']]

        print(k_attrs)



        item['five_most_common_attr_decs'] = [ attrs_with_most_three_descr[attr_map[attrs][0]]['most_common_decs'] for attrs in k_attrs[:5]]



    instructions_part1 = """Please generate prompts for analyzing subjective texts such as product reviews or social media according to the following rules:

1. Each prompt should capture the core and commonalities of the following attribute categories and without relying on specific attribute: {} ."""

    instructions_sub_part2 = """
    - The explanation for "{}" is {}"""

    instructions_part3 = """2. Ensure that each prompt is domain-general by using neutral references such as "this text" avoiding any specific domain indications.

3. Each prompt should be designed to help better understand subjective texts by deconstructing it based on the specified attribute categories.

4. Employ diverse strategies, which may include but are not limited to:
    - Open-ended deconstruction instructions (e.g., "Please analyze..." or "Please identify...")
    - Diagnostic questions (e.g., "What...?" "Which...?" "How..?")                               

5. Ensure that your responses are structured in ordered numbers.

Generated prompt:
    """

    for cluster in data:
        attrs = [ item.split('|')[0] for item in cluster['attr_names']]

        instructions_part2 = ""
        for attr,desc in zip(attrs[:5], cluster['five_most_common_attr_decs'][:5]):
            instructions_part2 += instructions_sub_part2.format(attr.lower(), desc.lower())


        cluster['instruction'] = instructions_part1.format(attrs) + instructions_part2 + '\n\n' +instructions_part3



        print(cluster['instruction'])
        # break

    output_path = os.path.join(output_dir, "4_instrution_for_generating_filtered_clutser.json")
    save_line_json(data,output_path )
