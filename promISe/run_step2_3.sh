export CUDA_VISIBLE_DEVICES=0,1,2,3
BENCHMARK=mmlu
AUGMENT_SIZE=15
CKPT_DIR=tiiuae/falcon-7b #local path or huggingface model name
PARAM_SIZE=7 # 7, 13, 33, 40, 65, 70
MODEL_TYPE=falcon # ["llama", "falcon", "llama2", "baichuan", "moss"]
BATCH_SIZE=8
TRAINING=True
TOTAL_ROUND=2

for ((ROUND=0; ROUND<TOTAL_ROUND; ROUND++))
do
    python inference.py \
        --ckpt_dir ${CKPT_DIR} \
        --param_size ${PARAM_SIZE} \
        --model_type ${MODEL_TYPE} \
        --benchmark ${BENCHMARK} \
        --batch_size ${BATCH_SIZE} \
        --round ${ROUND}\
        --training ${TRAINING}
    python result.py \
        --param_size ${PARAM_SIZE} \
        --model_type ${MODEL_TYPE} \
        --benchmark ${BENCHMARK} \
        --round ${ROUND}
    python prompt_selection.py \
        --param_size ${PARAM_SIZE} \
        --model_type ${MODEL_TYPE} \
        --benchmark ${BENCHMARK} \
        --round ${ROUND}
    
    if [ $ROUND -ne $TOTAL_ROUND - 1 ]; then
        python introspective_search.py \
        --param_size ${PARAM_SIZE} \
        --model_type ${MODEL_TYPE} \
        --benchmark ${BENCHMARK} \
        --round ${ROUND} \
        --augment_size ${AUGMENT_SIZE}
    fi
done

TRAINING=False
ROUND=best
python inference.py \
    --ckpt_dir ${CKPT_DIR} \
    --param_size ${PARAM_SIZE} \
    --model_type ${MODEL_TYPE} \
    --benchmark ${BENCHMARK} \
    --batch_size ${BATCH_SIZE} \
    --round ${ROUND}\
    --training ${TRAINING}