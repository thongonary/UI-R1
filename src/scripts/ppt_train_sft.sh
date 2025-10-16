export DATA_PATH=../../data/ppt-font-grounding
export CKPT_PATH=Qwen/Qwen2.5-VL-3B-Instruct
export SAVE_PATH=../../ckpt/Qwen2.5-VL-PPT-font-ground-sft
export LOG_PATH=${SAVE_PATH}"/train.log"

mkdir -p $SAVE_PATH
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ../ui_r1/src/open_r1/sft_json_action_coord.py \
    --per_device_train_batch_size 1 \
    --model_path ${CKPT_PATH} \
    --output_dir ${SAVE_PATH} \
    --data_path ${DATA_PATH} \
    --logging_steps 10 \
    --num_train_epochs 3 \
    --save_steps 165 \
    --remove_unused_columns False \
    # >> $LOG_PATH 2>&1
