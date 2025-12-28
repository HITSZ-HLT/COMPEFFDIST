# python difficulty_prioritized_sampling.py 
# --data_path 待评估的json文件 
# --sample_method soft 
# --strategy inprompt 
# --metric_key avg_rank_score  \\进行评估的难度键
# --sample_rate 0.5 \\采样多少比例
# --a 1
import argparse
import os
from utils import load_line_json, save_line_json, load_json
import random
import numpy as np
import mpmath as mp
random.seed(142)  # 设置随机种子以确保结果可重复
# random.seed(42)  # 设置随机种子以确保结果可重复
import math

def get_task_type_map():
    path = "../machine_generated_instr/11_clusterid_2all_prompts.json"
    data = load_json(path)

    extraction_task = []
    classification_task = []

    for k, v in data.items():
        extraction_task.extend( [ ( str(k) , item['task_name']) for item in  v['extraction']])
        if 'classification' in v:
            classification_task.extend( [ (str(k), item['task_name']) for item in v['classification']])

    return extraction_task, classification_task




def total_integral(a, b):
    """计算 F(x) 在 [0,1] 区间的积分"""
    I1 = mp.quad(lambda x: (1/a)**b * x**b, [0, a])
    I2 = mp.quad(lambda x: ((x-1)/(a-1))**b, [a, 1]) if a != 1 else 0
    return float(I1 + I2)

def find_b(a, sample_rate=0.5):
    """找到使积分等于sample_rate的b值"""
    mp.mp.dps = 20
    f = lambda b: total_integral(a, b) - sample_rate
    try:
        return float(mp.findroot(f, 1))
    except:
        return float(mp.findroot(f, [0.1, 10]))




def create_combined_function(a, b):
    """
    创建一个函数 F(x)，其中：
    F(x) = (1/a)^b * x^b 当 0 ≤ x < a
    F(x) = ((x-1)/(a-1))^b 当 a ≤ x ≤ 1
    """
    def F(x):

        # a == 1
        if a == 1:
            return (1/a)**b * x**b

        # a != 1 
        if 0 <= x < a:
            return (1/a)**b * x**b
        elif a <= x <= 1:
            return ((x-1)/(a-1))**b
        else:
            return float('nan')
    
    return F



def organize_data_by_task(data):
    organized_data = {}
    for item in data:
        selected_id = item['selected_cluster_id']
        task_name = item['task_name']

        if (selected_id, task_name) not in organized_data:
            organized_data[(selected_id, task_name)] = []

        organized_data[(selected_id, task_name)].append(item)

    return organized_data


def organize_data_by_task_with_analysis(data):

    def _get_analysis_prompt(text):
        return text.split("\n\nInput:")[0].strip()


    organized_data = {}
    for item in data:
        selected_id = item['selected_cluster_id']
        task_name = item['task_name']

        if task_name == 'analysis':
            task_name = _get_analysis_prompt(item['instruction'])

        if (selected_id, task_name) not in organized_data:
            organized_data[(selected_id, task_name)] = []

        organized_data[(selected_id, task_name)].append(item)

    # for key, items in organized_data.items():
    #     # 过滤掉长度小于1的样本
    #     print(f'key: {key}, items: {len(items)}')

    return organized_data




def custom_round(num):
    """
    自定义四舍五入算法：
    对于给定的数字 num，将其拆分为整数部分和小数部分。
    随机生成一个[0, 1)的随机数：
        - 如果随机数 < 小数部分，则四舍五入为整数部分 + 1；
        - 否则返回整数部分。
    """
    integer_part = int(num)
    fractional_part = num - integer_part
    
    # 生成随机数
    rand_val = random.random()  
    if rand_val < fractional_part:
        return integer_part + 1
    else:
        return integer_part



# random
def random_sample(data, sample_rate):

    sampled_data = random.sample(data, int(len(data) * sample_rate))

    return sampled_data


# ifd_hard + full
def ifd_hard_full_sample(data, sample_rate):

    """    
    参数:
        data: 包含字典的列表
        sample_size: 采样数量，默认为None（返回全部）
    
    返回:
        采样后的数据列表
    """

    sorted_data = sorted(data.copy(), key=lambda x: x['ifd_ppl'])

    
    # 返回采样的数据
    return sorted_data[-int(sample_rate * len(sorted_data)):]


# ifd_hard + inprompt
def ifd_hard_inprompt_sample(data, sample_rate):

    sampled_data = []
    
    organized_data = organize_data_by_task(data)
    
    for (selected_id, task_name), items in organized_data.items():


        num_items = len(items)
        num_samples = custom_round(num_items * sample_rate)  # 使用自定义的四舍五入函数

        if num_samples == 0:
            continue
        
        
        # 根据样本ifd值进行排序，选择ifd值大的样本
        sampled_items = sorted(items, key=lambda x: x['ifd_ppl'], reverse=True)[:num_samples]

        sampled_data.extend(sampled_items)
    
    return sampled_data



# ifd_soft + full
def ifd_soft_full_sample(data, a, sample_rate):

    # 计算参数
    b = find_b(a, sample_rate=sample_rate)
    print(f"sample_rate为{sample_rate}, 使用参数: a = {a}, b = {b:.4f}")


    sample_size=int(sample_rate * len(data)) 

    # 创建概率函数
    F = create_combined_function(a, b)
    
    # 根据key排序
    sorted_data = sorted(data.copy(), key=lambda x: x['ifd_ppl'])

    # 计算每个元素的采样概率
    n = len(sorted_data)
    probabilities = []
    
    for idx in range(n):
        # 映射索引到[0,1]区间
        x = idx / n if n > 1 else 0.5
        prob = F(x)
        probabilities.append(prob)
    
    # 归一化概率（确保它们的和为1）
    total_prob = sum(probabilities)

    print('total_prob', total_prob)


    if total_prob > 0:  # 避免除以零
        probabilities = [p/total_prob for p in probabilities]
    else:
        # 如果所有概率都为零，使用均匀分布
        probabilities = [1/n] * n
    
    # 使用概率进行加权采样
    sampled_indices = np.random.choice(
        range(n), 
        size=sample_size, 
        replace=False,  # 不放回采样
        p=probabilities
    )
    
    # 返回采样的数据
    return [sorted_data[i] for i in sampled_indices]


# ifd_soft + inprompt
def ifd_soft_inprompt_sample(data, a, sample_rate):


    # 计算参数
    b = find_b(a, sample_rate=sample_rate)
    print(f"sample_rate为{sample_rate}, 使用参数: a = {a}, b = {b:.4f}")


    # 创建概率函数
    F = create_combined_function(a, b)
    
    # 根据key排序
    organized_data = organize_data_by_task(data)

    sampled_data = []


    for (selected_id, task_name), items in organized_data.items():


        num_items = len(items)

        sample_size = custom_round(num_items * sample_rate)  # 使用自定义的四舍五入函数

        if sample_size == 0:
            continue
        
    
        sorted_data = sorted(items.copy(), key=lambda x: x['ifd_ppl'])

        
        # 计算每个元素的采样概率
        n = len(sorted_data)
        probabilities = []
        
        for idx in range(n):
            # 映射索引到[0,1]区间
            x = idx / n if n > 1 else 0.5
            prob = F(x)
            probabilities.append(prob)
        
        # 归一化概率（确保它们的和为1）
        total_prob = sum(probabilities)

        print('total_prob', total_prob)


        if total_prob > 0:  # 避免除以零
            probabilities = [p/total_prob for p in probabilities]
        else:
            # 如果所有概率都为零，使用均匀分布
            probabilities = [1/n] * n
        
        
        # 使用概率进行加权采样
        sampled_indices = np.random.choice(
            range(n), 
            size=sample_size, 
            replace=False,  # 不放回采样
            p=probabilities
        )

        sampled_data.extend([sorted_data[i] for i in sampled_indices])

        
    # 返回采样的数据
    return sampled_data



# soft + full
# metric_key为ppl, ifd, avg_rank_score, text_length
# 其中ppl, ifd, avg_rank_score, text_length为越高越难
def soft_full_sample(data, a, sample_rate, metric_key, reverse=False):

    # 创建概率函数
    b = find_b(a, sample_rate=sample_rate)
    F = create_combined_function(a, b)
    # print(f"sample_rate为{sample_rate}, 使用参数: a = {a}, b = {b:.4f}")
    

    # 根据key排序
    sorted_data = sorted(data.copy(), key=lambda x: x[metric_key], reverse=reverse)


    # 根据每个元素的idx计算采样概率
    n = len(sorted_data)
    probabilities = []
    for idx in range(n):
        x = (idx + 0.5) / n if n > 1 else 0.5
        prob = F(x)
        probabilities.append(prob)
    
    # 检查概率
    total_prob = sum(probabilities)
    sample_size=sample_rate * len(data)

    # print('total_prob', total_prob)
    # print('sample_size', sample_size)

    assert total_prob == sample_size, f'{total_prob} == {sample_size}'


    sampled_indices = []
    for i, prob in enumerate(probabilities):
        if random.random() < prob:
            sampled_indices.append(i)

    
    # 返回采样的数据
    return [sorted_data[i] for i in sampled_indices]


# soft + inprompt
def soft_inprompt_sample(data, a, sample_rate, metric_key, reverse=False):


    organized_data = organize_data_by_task_with_analysis(data)

    sampled_data = []

    for (selected_id, task_name), items in organized_data.items():

        sampled_data.extend(soft_full_sample(items, a, sample_rate, metric_key=metric_key, reverse=reverse))


    return sampled_data


# hard + full
def hard_full_sample(data, sample_rate, metric_key, reverse=False):

    # 根据key排序
    sorted_data = sorted(data.copy(), key=lambda x: x[metric_key], reverse=reverse)

    n = len(sorted_data)

    sample_size = custom_round(n * sample_rate)  # 使用自定义的四舍五入函数


    # 返回采样的数据
    return sorted_data[-sample_size:]



# hard + inprompt
def hard_inprompt_sample(data, sample_rate, metric_key, reverse=False):

    print('hard_inprompt_sample')

    organized_data = organize_data_by_task_with_analysis(data)

    sampled_data = []

    for (selected_id, task_name), items in organized_data.items():

        sampled_data.extend(hard_full_sample(items, sample_rate, metric_key=metric_key, reverse=reverse))


    return sampled_data




if __name__ == '__main__':

    parser =  argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--sample_method', type=str,default=None )
    parser.add_argument('--sample_rate', type=float, default=0.1)

    # 难度采样策略 'full', 'inprompt',  'task_type',
    parser.add_argument('--strategy', type=str, default=None)
    
    # metric_key 难度衡量指标
    parser.add_argument('--metric_key', type=str, default=None)

    # 排序方式
    parser.add_argument('--reverse', action='store_true', default=False)


    # stragegy为soft时 控制采样曲线
    parser.add_argument('--a', type=float, default=None)



    
    args = parser.parse_args()

    data_path = args.data_path
    metric_key = args.metric_key

    sample_method = args.sample_method
    sample_rate = args.sample_rate
    strategy = args.strategy
    a = args.a
    reverse = args.reverse
    

    # output_str = f'_{metric_key}_{sample_method}_{strategy}_rate_{sample_rate}'
    output_str = f'_seed_142_{metric_key}_{sample_method}_{strategy}_rate_{sample_rate}'

    if a is not None:
        output_str += f'_a_{a}'


    output_path = data_path.replace('.json', f'{output_str}.json')


    data = load_line_json(data_path)


    if sample_method == 'random':
        sampled_data = random_sample(data, sample_rate)
    elif sample_method == 'ifd_hard':
        if strategy == 'full':
            sampled_data = ifd_hard_full_sample(data, sample_rate)
        elif strategy == 'inprompt':
            sampled_data = ifd_hard_inprompt_sample(data, sample_rate)

    elif sample_method == 'ifd_soft':
        assert a is not None
        if strategy == 'full':
            sampled_data = ifd_soft_full_sample(data, a, sample_rate)
        elif strategy == 'inprompt':
            sampled_data = ifd_soft_inprompt_sample(data, a, sample_rate)

    elif sample_method == 'soft':
        assert a is not None
        assert metric_key is not None
        if strategy == 'inprompt':
            sampled_data = soft_inprompt_sample(data, a, sample_rate, metric_key)

        elif strategy == 'full':
            sampled_data = soft_full_sample(data, a, sample_rate, metric_key, reverse=reverse)


    elif sample_method == 'hard':
        assert metric_key is not None
        if strategy == 'inprompt':
            sampled_data = hard_inprompt_sample(data, sample_rate, metric_key, reverse=reverse)



    print('Data path:', data_path)
    print('Output path:', output_path)
    print('-'*10)


    print('Sample method:', sample_method)
    print('metric_key: ', metric_key)
    print('Reverse:', reverse)
    print('Strategy:', strategy)
    print('Sample rate:', sample_rate)
    

    print('a:', a)
    

    print('-'*10)
    print('Original data size:', len(data))
    print('Sampled data size:', len(sampled_data))

    
    save_line_json(sampled_data, output_path)



