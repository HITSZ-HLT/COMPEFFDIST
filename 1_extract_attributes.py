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

    print("-----------------1. Loading machine-generating facet analysis-----------------------------------")


    # 格式 data[0].keys()
    # domain
    # input: 分析指令 + 主观性文本
    # facet_analysis: 机器生成的属性分析文本，待解析
    # sentiment_relevent_generation

    data = []
    data.extend(load_line_json("./machine_generated_instr/attribute_enumerate/amazon_prompt_generation.json"))
    data.extend(load_line_json("./machine_generated_instr/attribute_enumerate/yelp_prompt_generation.json"))
    data.extend(load_line_json("./machine_generated_instr/attribute_enumerate/tweet_prompt_generation.json"))
    data.extend(load_line_json("./machine_generated_instr/attribute_enumerate/movie_prompt_generation.json"))
    data.extend(load_line_json("./machine_generated_instr/attribute_enumerate/tweet_politics_prompt_generation.json"))


    print("-----------------2. Parsing (attribute_name, attribute_description, attribute_value) for each sample-----------------------------------")
    
    
    

    # 格式 data[0].keys()
    # 新增：attr_desc_value: [(attribute_name, attribute_description, attribute_value), .... ] 对应的属性描述和属性值列表

    add_attr_desc_value(data)

    # print(data[0]['attr_desc_value'][:5])
    

    with open(os.path.join(output_dir, "1_text_with_attribute.json"), 'w', encoding='utf-8-sig') as f:
        json.dump(data, f, indent=4)



    print("-----------------3. Get all attribute triplets-----------------------------------")

    # all_attr_triplets: {
    #       attribute_name: 
    #           {
    #                   attr_desc: [attr_desc1, attr_desc2, ...], 
    #                   attr_value: [attr_value1, attr_value2, ...], 
    #                   counts: 1, 
    #                   most_common_decs: most_common_decs},
    #                   most_three_common_decs: [decs1, decs2, dc3]
    #       attribute_name2: ...
    #        }  所有属性的属性描述和属性值三元组

    all_attr_triplets = get_all_attr_triplets(data)

    with open(os.path.join(output_dir, "2_attrs_infomation.json"), 'w', encoding='utf-8-sig') as f:
        json.dump(all_attr_triplets, f, indent=4)