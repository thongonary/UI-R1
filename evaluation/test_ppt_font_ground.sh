# MODEL_PATH="../ckpt/Qwen2.5-VL-3B-UI-R1"
# IMG_PATH="../data/AndroidControl/screenshots"
# TEST_JSON="../data/AndroidControl/train.json"
# TEST_NAME="debug"



# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_androidcontrol.py\
#     --model_path ${MODEL_PATH} \
#     --image_path ${IMG_PATH} \
#     --test_json ${TEST_JSON} \
#     --test_name ${TEST_NAME}


#MODEL_PATH="../ckpt_ppt_click_only_sft_rl/Qwen2.5-VL-PPT-font-ground-dast-nothink-8ep"
# MODEL_PATH="/home/data/ckpt/Qwen2.5-VL-PPT-font-ground-dast-8ep-from-ckpt-150"
MODEL_PATH="/home/data/ckpt/Qwen2.5-VL-PPT-font-ground-dast-4ep/checkpoint-455/"
# MODEL_PATH="../ckpt/Qwen2.5-VL-PPT-font-ground-sft"
# MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
IMG_PATH="../data/ppt-font-grounding/test_imgs"
TEST_JSON="../data/ppt-font-grounding/test_ground_click_only.json"
TEST_NAME="GRPO--test"
LOG_PATH="output_dast_ckpt455.log"
CUDA_VISIBLE_DEVICES=0,1,2,3 python test_ppt_font_ground.py\
    --model_path ${MODEL_PATH} \
    --image_path ${IMG_PATH} \
    --test_json ${TEST_JSON} \
    --test_name ${TEST_NAME} \
    >> $LOG_PATH 2>&1