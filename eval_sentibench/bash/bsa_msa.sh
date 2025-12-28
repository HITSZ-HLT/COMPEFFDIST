while getopts ':d:c:b:s:m:h:p:z:k:v:' opt
do
    case $opt in
        c) CUDA_IDS="$OPTARG" ;;
        d) dataset="$OPTARG" ;;
        b) subname="$OPTARG" ;;
        s) seeds="$OPTARG" ;;
        m) model_name_or_path="$OPTARG" ;;
        h) max_seq_length="$OPTARG" ;;
        p) data_prop="$OPTARG" ;;
        z) eval_batch_size="$OPTARG" ;;
        k) k_shot="$OPTARG" ;;
        v) model_version="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done
echo "${model_version}"


if [ -z "${CUDA_IDS}" ]; then
    CUDA_IDS=0
fi


if [ -z "${subname}" ]; then
    subname="test"
fi


if [ -z "${seeds}" ]; then
    seeds="42 52 62"
fi

IFS=' ' read -r -a seed_array <<< "$seeds"


if [ -z "${model_name_or_path}" ]; then
    model_name_or_path="/data/TinyLlama-1.1B-Chat-v1.0"
fi


if [ -z "${max_seq_length}" ]; then
    max_seq_length=-1
fi


if [ -z "${data_prop}" ]; then
    data_prop=1.
fi

if [ -z "${eval_batch_size}" ]; then
    eval_batch_size=4
fi

if [ -z "${k_shot}" ]; then
    k_shot="4"
fi

IFS=' ' read -r -a shot_array <<< "$k_shot"




data_dir="./datasets"
output_dir="./result"


if [ "${model_version}" == "gemma-3-4b" ]; then
    script_name="icl_multifaced_add_eager.py"
else
    script_name="icl_multifaced.py"
fi


for k_shot in "${shot_array[@]}"; do
    for seed in "${seed_array[@]}"; do
       echo "python ${script_name} dataset: ${dataset}, subname: ${subname}, model_version: ${model_version} model_path: ${model_name_or_pathmodel}  seed: ${seed}"
        CUDA_VISIBLE_DEVICES=${CUDA_IDS} python ${script_name}  --seed ${seed} --subname ${subname}  --model_version "${model_version}" --model_name_or_path "${model_name_or_path}" --dataset ${dataset} --data_dir "${data_dir}" --output_dir "${output_dir}" --eval_batch_size ${eval_batch_size} --max_seq_length ${max_seq_length} --data_prop ${data_prop} --k_shot ${k_shot}
    done
done




