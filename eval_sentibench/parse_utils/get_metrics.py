import json
import os

import numpy as np
from collections import defaultdict

import sys
import argparse


def get_sentibench_matrics(exp_name, verbose=False, header=False):

    path = "../result/result.txt"

    data = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            data.append(json.loads(line))

    datasets1 = [ 'sc/imdb', 'sc/yelp2', 'sc/sst2', 'sc/twitter', 'multifaced/irony18', 'multifaced/tweeteval', 'multifaced/pstance', 'multifaced/intimacy',]
    datasets2 = [ 'absa/asqp_rest16', 'absa/opener', 'absa/atsa_rest16', 'absa/acsa_rest16']

    datasets = datasets1 + datasets2

    header_info = f"{'Experiment':<30} | " + " | ".join(f"{dataset:<40}" for dataset in datasets) + " | Average"

    if header:
        print(header_info)


    result = defaultdict(list)
    experiment_data = [item for item in data if exp_name in item[0]]

    for item in experiment_data:
        keys = item[0].split('_')
        model, dataset, seed, k = keys[-10], keys[-9], keys[-8], keys[-7][-1]
        if  'absa' in model:
            dataset = keys[-10] + '_' + keys[-9]
            model = keys[-11]
        result[(model, dataset, k)].append(item[1]['f1'])


    final_result = {}
    for k, v in result.items():
        if verbose:
            print(k[1], sum(v)/len(v)*100)
            print(v)
        if len(v) != 3:
            print(f'{k[1]}: error')
        final_result[k[1]] = sum(v) / len(v) * 100

    row_values = [final_result.get(dataset.lower(), 0) for dataset in datasets]
    row_average = sum(row_values) / len(datasets)

    row = f"{exp_name:<30} | " + " | ".join(f"{value:<5.2f}" for value in row_values) + f" | {row_average:<5.2f}"
    print(row)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', required=True )
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--header', action="store_true")

    args = parser.parse_args()

    exp_name = args.exp_name
    verbose = args.verbose
    header = args.header


    print(verbose, header)

    get_sentibench_matrics(exp_name,verbose,header)
