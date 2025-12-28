from tqdm import tqdm
import argparse
from utils import save_line_json, load_line_json, save_result
import random
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import load_json,save_json
from sklearn.metrics import f1_score, accuracy_score, classification_report
import datetime
import time
import os
from peft import PeftModel

id_to_label = {
    'multifaced/tweeteval': [' anger', ' joy', ' sadness', ' optimism'],
    'multifaced/intimacy': [' not intimate', ' moderately intimate', ' highly intimate'],
    'multifaced/offenseval':[' non-offensive', ' offensive'],
    'multifaced/irony18': [' non-irony', ' irony'],
    'multifaced/pstance': [' against', ' favor'],
    'sc/imdb': [' negative', ' positive'],
    'sc/sst2': [' negative', ' positive'],
    'sc/yelp2': [' negative', ' positive'],
    'sc/twitter': [' negative', ' positive', ' neutral'],
}



def get_label_list(model_name_or_path):

    def _get_verbalizer(model_name_or_path, labels):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        v = []
        for label in labels:
            token_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
            v.append(token_ids[0])
        return v
        



    label_to_id = {}
    for dataset,labels in id_to_label.items():
        ids = _get_verbalizer(model_name_or_path, labels)
        label_to_id[dataset] = ids

    return label_to_id



def cal_metric(results, y_true, y_pred, seed, subname, model_name_or_path, dataset, output_dir, k_shot):
    model_name = model_name_or_path.split('/')[-1]
    print(model_name)
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f'{subname}_{model_name}_{dataset}_{seed}_numOfshot{k_shot}_{current_time}'
    output_path = os.path.join(output_dir, file_name + '.json')
    print(output_path)

    #print(y_true)
    print(Counter(y_true))
    #print(y_pred)
    print(Counter(y_pred))

    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    print(f'f1: {f1:.4f} | accuracy: {accuracy:.4f}\n')


    save_result(output_dir, file_name, f1, accuracy)
    save_json(results, output_path)


class DataCollator:
    def __init__(self, tokenizer, max_seq_length, mode):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode

    def tok(self, text, max_seq_length):
        kwargs = {
            'text': text,
            'return_tensors': 'pt',
        }

        if max_seq_length in (-1, 'longest'):
            kwargs['padding'] = True

        else:
            kwargs['max_length'] = max_seq_length
            kwargs['padding'] = 'max_length'
            kwargs['truncation'] = True

        batch_encodings = self.tokenizer(**kwargs)
        return batch_encodings

    def __call__(self, examples):

        texts = [example['prompts'] for example in examples]
        batch_encodings = self.tok(texts, -1)

        input_ids = batch_encodings['input_ids']
        attention_mask = batch_encodings['attention_mask']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'examples': examples
        }

def get_dataloader(test_examples, mode, batch_size, shuffle, tokenizer):
    dataloader = DataLoader(
        dataset=test_examples,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        prefetch_factor=8,
        num_workers=1,
        collate_fn=DataCollator(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            mode=mode,
        )
    )

    print('dataloader-' + mode, len(dataloader))
    return dataloader


def test_dataloader(test_examples, eval_batch_size,tokenizer):
    return get_dataloader(test_examples, "test", eval_batch_size, shuffle=False, tokenizer=tokenizer)

# tiny-llama
# ddd = {
#     'non-irony':'not irony',
#     'irony':'irony'
# }
def construct_prompt(prompt, test_examples, demo_pool, k_shot):

    for example in test_examples:
        text = example['sentence']
        demos = random.sample(demo_pool, k=k_shot)
        prompts = [prompt]
        prompts.append('')
        for demo in demos:
            prompts.append(f"Sentence: {demo['sentence']}\nLabel: {demo['label'].lower()}\n")
        prompts.append(f"Sentence: {example['sentence']}\nLabel:")
        example['prompts'] = '\n'.join(prompts)

        # print('-'*100)
        # print(example['prompts'])
        # print('-'*100)
        

    return test_examples


def construct_prompt_pstance(prompt, test_examples, demo_pool, k_shot):

    for example in test_examples:
        text = example['sentence']
        target = example['target']
        demos = random.sample(demo_pool, k=k_shot)
        prompts = [prompt % target]
        prompts.append('')
        for demo in demos:
            prompts.append(f"Sentence: {demo['sentence']}\nLabel: {demo['label'].lower()}(opinion towards '{demo['target']}')\n")
        prompts.append(f"Sentence: {example['sentence']}\nLabel:")
        example['prompts'] = '\n'.join(prompts)

        # print('-'*100)
        # print(example['prompts'])
        # print('-'*100)

    return test_examples



def load_dataset(data_path):
    test_file_name = os.path.join(data_path, 'test.json')
    test_examples = load_json(test_file_name)

    demo_pool_file_name = os.path.join(data_path, 'train.json')
    demo_examples = load_json(demo_pool_file_name)

    raw_datasets = {
        'test': test_examples,
        'demo': demo_examples
    }

    print('-----------data statistic-------------')
    for mode in ('test','demo'):
        num_samples = len(raw_datasets[mode])
        print(f'{mode.upper():<5} | Samples: {num_samples:<5}')
    print('--------------------------------------')



    return raw_datasets



import sys
import argparse
import datetime
if __name__ == '__main__':
    print('hello')
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True, )
    parser.add_argument('--subname', required=True, )
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--model_name_or_path', default="/data/Llama-3.2-1B-Instruct" )
    parser.add_argument('--model_version', required=True)
    parser.add_argument('--max_seq_length', default=-1 )
    parser.add_argument('--data_prop', default=1)
    parser.add_argument('--eval_batch_size', type=int, default=4 )
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--k_shot', type=int, default=8)

    args = parser.parse_args()

    dataset = args.dataset
    subname = args.subname
    seed = args.seed
    model_version = args.model_version
    model_name_or_path = args.model_name_or_path
    max_seq_length = args.max_seq_length
    data_prop = args.data_prop
    eval_batch_size = args.eval_batch_size
    data_dir = args.data_dir
    output_dir = args.output_dir
    k_shot = args.k_shot

    data_path = os.path.join(data_dir, dataset)
    torch.manual_seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)


    model_name_or_path = load_json('../post_train/bash/model_name.json')[model_version]

    print('-'*20+'Experiment Setup'+'-'*20)
    print('data_path: ', data_path)
    print(f' subname: {subname}\n dataset: {dataset}\n seed:{seed}\n model_name_or_path:{model_name_or_path}\n data_path:{data_path}\n output_dir:{output_dir}\n eval_batch_size:{eval_batch_size}\n data_prop:{data_prop}\n max_seq_length:{max_seq_length}\n numbers_of_shot:{k_shot}')
    print('-' * 50)

    prompts = {
        'multifaced/irony18':
            """Please perform Irony Detection task. Given the sentence, assign a sentiment label from ['irony', 'non-irony']. Return label only without any other text.""",
        'multifaced/offenseval':
            """Please perform Offensive Detection task. Given the sentence, assign a sentiment label from ['non-offensive', 'offensive']. Return label only without any other text.""",
        'multifaced/intimacy':
            """Please perform Intimacy Detection task. Given the sentence, assign an intimacy label from ['not intimate', 'moderately intimate', 'highly intimate']. Return label only without any other text.""",
        'multifaced/tweeteval':
            """Please perform Emotion Detection task. Given the sentence, assign a emotion label from ['anger', 'joy', 'sadness', 'optimism']. Return the label only without any other text.""",
        'multifaced/pstance':
            """Please perform Stance Detection task. Given the sentence, assign a sentiment label expressed by the author towards "%s" from ['against', 'favor']. Return label only without any other text.""",
        'sc/imdb':
            """Please perform Sentiment Analysis task. Given the sentence, assign a sentiment polarity label from ['negative', 'positive']. Return label only without any other text.""",
        'sc/sst2':
            """Please perform Sentiment Analysis task. Given the sentence, assign a sentiment polarity label from ['negative', 'positive']. Return label only without any other text.""",
        'sc/yelp2': 
            """Please perform Sentiment Analysis task. Given the sentence, assign a sentiment polarity label from ['negative', 'positive']. Return label only without any other text.""",
        'sc/twitter':
            """Please perform Sentiment Analysis task. Given the sentence, assign a sentiment polarity label from ['negative', 'positive', 'neutral']. Return label only without any other text.""",
    }




    raw_datasets = load_dataset(data_path)

    if dataset == 'multifaced/pstance':
        test_examples = construct_prompt_pstance(prompts[dataset], raw_datasets['test'], raw_datasets['demo'], k_shot)
    else:
        test_examples = construct_prompt(prompts[dataset], raw_datasets['test'], raw_datasets['demo'], k_shot)




    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = test_dataloader(test_examples, eval_batch_size, tokenizer)
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                #  attn_implementation="eager"
                                                 )


    
    
    model.eval()

    test_step_outputs = []

    # Llama-3.2-1B-Instrcut
    label_list = get_label_list(model_name_or_path)
    print("label_list is: ", label_list)

    aa = []
    predictions = []
    labels = []

    for batch in tqdm(dataloader):
# 
        # output = model.generate(
        #     batch['input_ids'].to('cuda'),
        #     attention_mask=batch['attention_mask'].to('cuda')
        #     , max_new_tokens=2, temperature=1, output_scores=True, return_dict_in_generate=True
        # )
        
        output = model.generate(
            batch['input_ids'].to('cuda'),
            attention_mask=batch['attention_mask'].to('cuda')
            , max_new_tokens=2, do_sample=False, output_scores=True, return_dict_in_generate=True
        )
        
        # print('-'*100)
        # results = tokenizer.batch_decode(output,skip_special_tokens=True)
        
        # for i in range(len(batch['examples'])):
        #     print(batch['examples'][i]['prompts'])
        #     print('-'*100)
            # print(results[i][results[i].rfind('Label:') + len('Label:'):])
            # if results[i][results[i].rfind('Label:') + len('Label:'):].strip() not in ['not', 'highly','moderately']:
            #     predictions.append(random.choice(['not', 'highly','moderately']))
            # else:
            #     predictions.append(results[i][results[i].rfind('Label:') + len('Label:'):].strip())
            # labels.append(batch['examples'][i]['label'].lower().split()[0])
            # print(batch['examples'][i]['label'])
            # print('-'*100)
            
        

        scores = output.scores[0].cpu()

        # --------------------------- #

        tmp_scores = scores.argmax(-1)

        for score in tmp_scores:
            aa.append(score.item())
        # --------------------------- #

        scores = scores[:, label_list[dataset]]
        #print(scores)
        class_scores = scores
        scores = scores.argmax(-1)
        dict = { 'examples': batch['examples'],
                'predictions': scores,
                 'class_scores':class_scores
                 }
        test_step_outputs.append(dict)

    # print(Counter(labels))
    # print(Counter(predictions))
    # f1 = f1_score(labels, predictions, average='macro')
    # classification_report(labels, predictions)
    # print(f1)

    print('label list is: ')
    print(label_list[dataset])
    print('all next most probable token set is: ')
    print(Counter(aa))

    predictions = []
    labels = []
    
    
    results = []
    for output in test_step_outputs:
        for example, prediction, score in zip(output['examples'], output['predictions'], output['class_scores']):
            labels.append(example['label']                          .lower())
            predictions.append(id_to_label[dataset][prediction].strip())
    
    
            result = example
            result['prediction'] = id_to_label[dataset][prediction].strip()
            result['prediction_score'] = score.tolist()
            results.append(result)
    
    cal_metric(results,labels, predictions, seed, subname, model_name_or_path, dataset, output_dir, k_shot )

