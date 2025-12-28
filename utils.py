import json
import os

import numpy as np
from collections import defaultdict
from tqdm import tqdm


def load_json(file_name):
    with open(file_name, mode='r', encoding='utf-8-sig') as f:
        return json.load(f)


def get_amazon_input_sample(path='./prompts/input_samples/sampled_amazon_input.json'):
    input = load_json(path)
    data = []
    for catagory, items in input.items():
        for item in items:
            data.append(item['reviewText'])
    return data

def get_yelp_input_sample(path='./prompts/input_samples/sampled_yelp_input.json'):
    input = load_json(path)
    data = [item['Text'] for item in input]
    return data

def get_movie_input_sample(path='./prompts/input_samples/sampled_movie_input.json'):
    input = load_json(path)
    data = [item['Text'] for item in input]
    return data


def get_tweet_input_sample(path='./prompts/input_samples/sampled_tweet_input.json'):
    input = load_json(path)
    data = []
    for date, items in input.items():
        data.extend(items)
    return data


def get_tweet_politics_input_sample(path='./prompts/input_samples/sampled_tweet_politics_input.json'):
    data = load_json(path)
    return data


def get_instruction_input_sample(path='./prompts/facet_based_instruction.json'):
    data = load_json(path)
    return data['input_sample'], data['facet_analysis'], data['sentiment_relevent_generation']


def mkdir_if_not_exist(path):
    dir_name, file_name = os.path.split(path)
    if dir_name:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)



def save_line_json(json_obj, file_name):
    mkdir_if_not_exist(file_name)
    with open(file_name, mode='w', encoding='utf-8-sig') as f:
        lines = [json.dumps(line)+'\n' for line in json_obj]
        lines[-1] = lines[-1][:-1]
        f.writelines(lines)


def load_line_json(file_name):
    data = []
    with open(file_name, mode='r', encoding='utf-8-sig') as f:
        for line in f:
            data.append(json.loads(line))

    return data





def convert_format_2alpaca(input_dir, file_names, output_dir, instruction_key, output_key, input_key=None, sample_count=-1):

    mkdir_if_not_exist(output_dir)
    for file_name in tqdm(file_names):
        data = load_line_json(os.path.join(input_dir, file_name))
        new_data = []
        for item in data:
            new_item = {}
            new_item["instruction"] = item[instruction_key]
            new_item["output"] = item[output_key]
            if input_key:
                new_item["input"] = item[input_key]
            else:
                new_item["input"] = ""
            new_data.append(new_item)



        if sample_count > 0:
            import random
            random.seed(42)  # For reproducibility
            random.shuffle(new_data)
            new_data = new_data[:sample_count]
        
        save_line_json(new_data, os.path.join(output_dir, file_name))

def convert_format_2kdistill(input_dir, file_names, output_dir, prompt_key, llm_response_key):
    
    mkdir_if_not_exist(output_dir)
    for file_name in tqdm(file_names):
        data = load_line_json(os.path.join(input_dir, file_name))
        new_data = []
        for item in data:
            new_item = {}
            new_item["prompt"] = item[prompt_key]
            new_item["llm_response"] = item[llm_response_key]

            # 没有的键填充为空
            for null_key in ['instr_type', 'dataset_name', 'chunk_id']:
                new_item[null_key] = ""


            new_data.append(new_item)

        save_line_json(new_data, os.path.join(output_dir, file_name))




