
import json
import os
import argparse
import random
from tqdm import trange, tqdm
random.seed(42)
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from utils import get_amazon_input_sample, get_yelp_input_sample, get_movie_input_sample, get_tweet_input_sample, get_instruction_input_sample, get_instruction_input_sample, get_tweet_politics_input_sample, save_line_json

    
def get_all_domains_input(paths = ['./prompts/input_samples/sampled_amazon_input_4k.json',
                                  './prompts/input_samples/sampled_yelp_input.json',
                                  './prompts/input_samples/sampled_movie_input.json',
                                  './prompts/input_samples/sampled_tweet_input.json',
                                  './prompts/input_samples/sampled_tweet_politics_input.json']):
    all_inputs = {}

    all_inputs['amazon'] = get_amazon_input_sample(paths[0])

    all_inputs['yelp'] = get_yelp_input_sample(paths[1])

    all_inputs['movie'] = get_movie_input_sample(paths[2])

    all_inputs['tweet'] = get_tweet_input_sample(paths[3])

    all_inputs['tweet_politics'] = get_tweet_politics_input_sample(paths[4])


    print('-----------------------statistic of all domains input samples-----------------------')
    
    for domain, inputs in all_inputs.items():
        print(f'{domain} has {len(inputs)} samples')

    print('------------------------------------------------------------------------------------')

    return all_inputs




# CUDA_VISIBLE_DEVICES=0,1,2,3 python 0_generate_attribute.py -m /data/Meta-Llama-3.1-70B-Instruct -z 100 -c 0.7 -g 4 -i ./prompts/input_samples/sampled_amazon_input_4k.json__./prompts/input_samples/sampled_yelp_input_4k.json__./prompts/input_samples/sampled_movie_input_4k.json__./prompts/input_samples/sampled_tweet_input_4k.json__./prompts/input_samples/sampled_tweet_politics_input_4k.json -o ./output_data/


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name_or_path', type=str, default='/data/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('-z', '--batch_size', type=int, default=1)
    parser.add_argument('-c', '--cache_count', type=float, default=0.7)
    parser.add_argument('-g', '--gpus', type=int, default=1)
    parser.add_argument('-i', '--input_path', type=str,)
    parser.add_argument('-o', '--output_dir', type=str, default='./output_data/')

    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    batch_size = args.batch_size

    cache_count = args.cache_count
    num_gpus = args.gpus
    input_path = args.input_path.split('__')
    output_dir = args.output_dir


    print('--------------------------------Setting------------------------------------')
    print(f'model_name_or_path: {model_name_or_path}')
    print(f'batch_size: {batch_size}')
    print(f'cache_count: {cache_count}')
    print(f'num_gpus: {num_gpus}')
    print(f'input_path: {input_path}')
    print(f'output_path: {output_dir}')
    print('---------------------------------------------------------------------------')



    _, facet_analysis, sentiment_relevent_generation = get_instruction_input_sample("./prompts/facet_based_instruction.json")



    engine_config = TurbomindEngineConfig(tp=num_gpus, 
                                          cache_max_entry_count=cache_count,
                                          enable_prefix_caching=True)
    

                                #       do_sample=True,
                                #   
    gen_config = GenerationConfig(max_new_tokens=1048,temperature=1.0,
)


    pipe = pipeline(model_name_or_path,
                    backend_config = engine_config)



    all_inputs = get_all_domains_input(input_path)


    for domain, inputs in all_inputs.items():
        print('-----------------------start generating prompt for domain: ', domain)

        try:
            data = []
            for i in trange(0, len(inputs), batch_size):
                batch = inputs[i:i+batch_size]


                messages = [ facet_analysis.format(input_text) for input_text in batch]


                facet_response =  pipe(messages, gen_config)




                
                for i in range(len(messages)):
                    data.append(
                        {
                            'domain': domain,
                            'input': messages[i],
                            'facet_analysis': facet_response[i].text,
                            'sentiment_relevent_generation': ""
                        }
                    )
    

            save_line_json(data, os.path.join(output_dir, f'{domain}_prompt_generation.json'))
        
            
        except:
            print('-----------------------error generating prompt for domain: ', domain)


