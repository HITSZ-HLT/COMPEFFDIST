import json 
from collections import defaultdict
import os
import numpy as np


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        print('load_json:', file_path)
        return json.load(file)


def save_json(data, file_path):
    # 如果目录不存在，则创建目录
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8-sig') as file:
        print('save_json:', file_path)
        json.dump(data, file, ensure_ascii=False, indent=4)


attr_normalize_count1 = 0  # 初始化计数器
attr_normalize_count2 = 0  # 初始化计数器
attr_normalize_count3 = 0  # 初始化计数器
def attr_normalize(attr_name):
    # 记录多少个attr_name进行了此类处理
    global attr_normalize_count1, attr_normalize_count2, attr_normalize_count3
    
    if ':' in attr_name:
        attr_normalize_count1 += 1
        return attr_name.split(':')[:1]
    
    elif '/' in attr_name:
        attr_normalize_count2 += 1
        return attr_name.split('/')
    
    elif 'use of ' in attr_name:
        attr_normalize_count3 += 1
        return [attr_name.split('use of ')[1]]
    
    else:
        return [attr_name]


# verbose: 是否记录normalize前的形式
def collect_attr(data, verbose=True):
    attr_list = defaultdict(list)
    for line in data:
        for attr_name, _, _ in line['attr_desc_value']:
            normalized_attrs = attr_normalize(attr_name)
            for normalized_attr in normalized_attrs:
                attr_list[normalized_attr].append(attr_name)

    print('attr_normalize_count1:', attr_normalize_count1)
    print('attr_normalize_count2:', attr_normalize_count2)
    print('attr_normalize_count3:', attr_normalize_count3)

    print('过滤前:', len(attr_list))
    # 过滤掉频次小于10的属性
    attr_list = [(attr, list(set(attr_names)), len(attr_names)) 
                 for attr, attr_names in attr_list.items() if len(attr_names) > 10]
    print('过滤后:', len(attr_list))

    # 为每个attr生成一个唯一的id
    attr_list = [{'idx': idx, 'attr': attr, 'attr_names': attr_names, 'count': count}
                 if verbose else attr
                 for idx, (attr, attr_names, count) in enumerate(attr_list)]
    
    return attr_list


if __name__ == "__main__":
    data = load_json("../machine_generated_instr/1_text_with_attribute.json")
    print('len(data):', len(data))

    attr_list = collect_attr(data, verbose=True)
    save_json(attr_list, './data/attr_list.json')