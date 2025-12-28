

#!/usr/bin/env bash
set -euo pipefail


model_version="compeff_llama_50k_training"
exp_name="compeff_llama_50k_training"
config_file="configs/3b_full_config.yaml"


max_epochs=4
batch_size=8
gradient_accumulation_steps=8
lr=5e-5
weight_decay=0.1
beta1=0.9
beta2=0.999
eps=1e-8
lr_scheduler=torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup
num_warmup_steps=0


# 注意：将这里的路径替换为huggingface上下载的jsonl文件路径
data_files="./training_data/Meta-Llama-3.1-70B-Instruct_50k.json"

# 指定输出路径
output_dir="./post_train_models/${exp_name}"
output_model_path="./post_train_models/${exp_name}/epoch_$((max_epochs - 1))"



CUDA_VISIBLE_DEVICES=0,1 tune run --nproc_per_node 2 full_finetune_distributed \
  --config ${config_file} \
  output_dir=${output_dir} \
  dataset._component_=torchtune.datasets.instruct_dataset \
  dataset.column_map="{'input':'instruction', 'output':'output'}" \  # 这里要替换为实际的列名如Qwen/Qwen3-4B_output
  dataset.data_files=${data_files} \
  epochs=${max_epochs} \
  batch_size=${batch_size} \
  gradient_accumulation_steps=${gradient_accumulation_steps} \
  lr_scheduler._component_=${lr_scheduler} \
  lr_scheduler.num_warmup_steps=${num_warmup_steps} \
  optimizer.lr=${lr} \
  optimizer.weight_decay=${weight_decay} \
  optimizer.betas=[${beta1},${beta2}] \
  optimizer.eps=${eps} \



json_file="./model_name.json"

echo "model_version: $model_version"
echo "output_model_path: $output_model_path"

jq --arg key "$model_version" --arg value "$output_model_path" '. + {($key): $value}' "$json_file" > tmp.json && mv tmp.json "$json_file"


# 切换为eval_sentibench目录
cd ../eval_sentibench

bash/bsa_msa.sh -c 0 -d sc/imdb -b ${exp_name} -z 1 -v ${model_version} &
bash/bsa_msa.sh -c 1 -d sc/yelp2 -b ${exp_name} -z 1 -v ${model_version} &
wait

bash/bsa_msa.sh -c 0 -d sc/sst2 -b ${exp_name} -z 4 -v ${model_version} &
bash/bsa_msa.sh -c 1 -d sc/twitter -b ${exp_name} -z 4 -v ${model_version} &
wait


bash/bsa_msa.sh -c 0 -d multifaced/irony18 -b ${exp_name} -z 4 -v ${model_version} &
bash/bsa_msa.sh -c 1 -d multifaced/tweeteval -b ${exp_name} -z 4 -v ${model_version} &
wait

bash/bsa_msa.sh -c 0 -d multifaced/pstance -b ${exp_name} -z 4 -v ${model_version} &
bash/bsa_msa.sh -c 1 -d multifaced/intimacy -b ${exp_name} -z 4 -v ${model_version} &
wait


bash/fsa_json.sh -c 0 -d absa/acsa_rest16 -b ${exp_name} -z 32 -v ${model_version} &
bash/fsa_json.sh -c 1 -d absa/atsa_rest16 -b ${exp_name} -z 32 -v ${model_version} &
wait

bash/fsa_json.sh -c 0 -d absa/asqp_rest16 -b ${exp_name} -z 32 -v ${model_version} &
bash/fsa_json.sh -c 1 -d absa/opener -b ${exp_name} -z 32 -v ${model_version} -n 300 &
wait

python ./parse_utils/get_metrics.py --exp_name ${exp_name}
