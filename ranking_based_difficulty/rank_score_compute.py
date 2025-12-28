# CUDA_VISIBLE_DEVICES=0 python ./ranking_based_difficulty/rank_score_compute.py \
    # --data_path 需要评估的json文件路径，输入键为instruction, 输出键为output \
    # --save_path 保存路径 \
    # --model_name_or_path  计算难度的模型路径
import os
import json
import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import math
import sys
import random
random.seed(42)


def load_json(file_name):
    with open(file_name, mode='r', encoding='utf-8-sig') as f:
        return json.load(f)

import torch.nn as nn
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



def compute_avg_rank_score(tokenizer, model, text, target_span, max_length, top_n=1, m=5, top_p=0.95):
    
    try:
        # 编码文本并定位目标区间
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        start_index = text.rfind(target_span)
        start_token = len(tokenizer.encode(text[:start_index]))
        labels = input_ids.clone()
        labels[0, :start_token] = -100  # 屏蔽非目标区间的标签

        # 模型前向计算
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
        
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        
        # 获取有效 token 位置
        mask = labels != -100
        positions = mask[0].nonzero(as_tuple=False).squeeze(1)
        
        # 调整 logits 位置（因果模型对齐）
        logits_positions = positions - 1
        valid_mask = (logits_positions >= 0) & (logits_positions < logits.shape[1])
        logits_positions = logits_positions[valid_mask]
        target_positions = positions[valid_mask]
        target_ids = labels[0, target_positions]
        

        rank_scores = []
        
        # 遍历每个有效 token 位置
        for pos, tid in zip(logits_positions, target_ids):
            # 获取当前步骤的 logits
            step_logits = logits[0, pos, :]  # [vocab_size]
            
            # 计算概率分布
            probs = torch.softmax(step_logits, dim=-1)
            
            
            # 获取当前标签 token 的概率
            pl = probs[tid].item()
            

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            
            # 计算达到top_p的最小n
            n = (cumulative_probs < top_p).sum().item() + 1
            n = min(n, len(sorted_probs))  # 防御性限制
            
            # 提取前n个token的索引
            topn_indices = sorted_indices[:n]
            
            # 检查目标token是否在topn中
            if tid in topn_indices:
                # 计算排名x（从0开始计数）
                x = (topn_indices == tid).nonzero(as_tuple=True)[0].item()
                score = x / n 
            else:
                score = 1.0  # 不在topn中
            
            if score > 1e-6:  # 避免浮点误差
                rank_scores.append(score)




        avg_rank_score = sum(rank_scores) / len(rank_scores) if len(rank_scores) > 0 else 0.0

        
        return {

            "avg_rank_score": avg_rank_score,
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, default='')
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default='alpaca', help='vicuna, wiz, alpaca')
    args = parser.parse_args()
    return args




def get_data(path):
    data = []
    with open(path, "r", encoding='utf-8-sig') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

        

def main():

    args = parse_args()
    print(args)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", cache_dir='../cache', output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir='../cache')

    model.eval()



    data = get_data(args.data_path)
    

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data[start_idx:end_idx]
    
    random.shuffle(sampled_data)

    if not os.path.exists(args.save_path):
        with open(args.save_path, "w") as file:
            pass  # Creates an empty file



    for i in tqdm(range(len(sampled_data))):

        data_i = sampled_data[i]
        # instruct_i = data_i['instruction']
        instruct_i = data_i['instruction'].strip().replace('Input: ', 'Input:').replace('Output: ', 'Output:')

        output_i = data_i['output']


        whole_text = instruct_i + output_i


        result = compute_avg_rank_score(
        tokenizer,
        model,
        whole_text,
        output_i,
        max_length=args.max_length,
        top_n=3,
        top_p = 0.95)
        
        data_i['result'] = result

    with open(args.save_path, "a") as f:

        for i in range(len(sampled_data)):
            f.write(json.dumps(sampled_data[i], ensure_ascii=False) + "\n")
            


if __name__ == "__main__":
    main()