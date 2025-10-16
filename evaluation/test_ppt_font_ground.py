from tqdm import tqdm
import os
import json
import argparse
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import sys
import re
import multiprocessing as mp
import logging
from multiprocessing import Pool
import functools
import torch.multiprocessing as mp
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

rank = 0
def extract_content(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    content_pattern = r"'content': '(.*)'"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        content_match = re.search(content_pattern, content_answer)
        if content_match:
            return content_match.group(1).strip()
    return ''

def extract_coord(content, contains_multiple=False):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r"\[(\d+),\s*(\d+)\]"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    coords = []
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        coord_match = re.findall(bbox_pattern, content_answer)
        for (x, y) in coord_match:
            coord = [int(x), int(y)]
            coords.append(coord)
    
    if len(coords) == 0:
        return [0, 0, 0, 0], False
    
    if len(coords) == 1 or not contains_multiple:
        return coords[0], False
    
    return coords[:2], False

def extract_action(response):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'action':\s*'(\w+)'"
    action_pattern_1 = r"'action':\s*(\w+)"
    content_answer_match = re.search(answer_tag_pattern, response, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
        action_match = re.search(action_pattern_1, content_answer)
        if action_match:
            return action_match.group(1)
    return None



logger = logging.getLogger(__name__)

def run(rank, world_size, args):
    if "Qwen2.5" in args.model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
    if args.ori_processor_path is None:
        ori_processor_path = args.model_path
    infer_dir = os.path.join('infer')
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)
    output_file = os.path.join(infer_dir, f'prediction_results_{args.test_name}.jsonl')

    processor = AutoProcessor.from_pretrained(ori_processor_path)
    model = model.to(torch.device(rank))
    model = model.eval()
    
    error_count_action = 0
    correct_count_action = 0
    error_count_arg = 0
    correct_count_arg = 0
    pred_results = []
    total_output_tokens = 0
    total_samples = 0
    output_token_list = []
    total_think_tokens = 0
    think_token_list = []
    

    dataset = args.test_json
    data = json.load(open(dataset, "r"))
    
    data = data[rank::world_size]
    print(f"Process {rank} handling {len(data)} samples", flush=True)

    correct_action_by_action = {'click': 0, 'drag': 0, 'type': 0}
    correct_arg_by_action = {'click': 0, 'drag': 0, 'type': 0}

    for j, item in tqdm(enumerate(data), total=len(data)):
        image_path = os.path.join(args.image_path, item["img_filename"])  # 通过 args 传递路径
        task_prompt = item["instruction"]

        # question_template = (
        #         f"In this UI screenshot, I want to perform the command '{task_prompt}'.\n"
        #         "Please provide the action to perform (enumerate in ['click', 'drag', 'type'])"
        #         "and the coordinate where the cursor is moved to when clicking (integers)\n"
        #         "Output the final answer in <answer> </answer> tags directly."
        #         "The output answer format should be:\n"
        #         "<answer>[{'action': 'click', 'coordinate': [x, y]}]</answer>\n"
        #         "Please strictly follow the format."
        #     )

        question_template = (
            f"In this UI screenshot, I want to perform the command '{task_prompt}'.\n"
            "Please provide the action to perform (enumerate in ['click'])"
            "and the coordinate where the cursor is moved to when clicking (integers)\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            "The output answer format should be:\n"
            "<think> ... </think> <answer>[{'action': 'click', 'coordinate': [x, y]}]</answer>\n"
            "Please strictly follow the format."
        )

        # question_template = (
        #     f"In this UI screenshot, I want to perform the command '{task_prompt}'.\n"
        #     "Please provide the action to perform (enumerate in ['click', 'drag', 'type'])"
        #     "and the argument for the action to perform:\n"
        #     "1) click: the coordinate where the cursor is moved to when clicking (integers)\n"
        #     "2) drag: the starting and ending coordinates for the cursor when performing the drag (integers)\n"
        #     "3) type: the content text to be typed (string)\n"
        #     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        #     "The output answer format should be one of the following:\n"
        #     "1) <think> ... </think> <answer>[{'action': 'click', 'coordinate': [x, y]}]</answer>\n"
        #     "2) <think> ... </think> <answer>[{'action': 'drag', 'start_coordinate': [x, y], 'end_coordinate': [x, y]}]</answer>\n"
        #     "3) <think> ... </think> <answer>[{'action': 'type', 'content': 'text'}]</answer>\n"
        #     "Please strictly follow the format."
        # )

        # question_template = (
        #     f"In this UI screenshot, I want to perform the command '{task_prompt}'.\n"
        #     "Please provide the action to perform (enumerate in ['click', 'drag', 'type'])"
        #     "and the argument for the action to perform:\n"
        #     "1) click: the coordinate where the cursor is moved to when clicking (integers)\n"
        #     "2) drag: the starting and ending coordinates for the cursor when performing the drag (integers)\n"
        #     "3) type: the content text to be typed (string)\n"
        #     "Output the final answer in <answer> </answer> tags directly."
        #     "The output answer format should be one of the following:\n"
        #     "1) <answer>[{'action': 'click', 'coordinate': [x, y]}]</answer>\n"
        #     "2) <answer>[{'action': 'drag', 'start_coordinate': [x, y], 'end_coordinate': [x, y]}]</answer>\n"
        #     "3) <answer>[{'action': 'type', 'content': 'text'}]</answer>\n"
        #     "Please strictly follow the format."
        # )
        # query = '<image>\n' + question_template
        query = question_template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}
                ] + [{"type": "text", "text": query}],
            }
        ]

        gt = {'action': item['action']}
        if item['action'] == 'click':
            gt['bbox'] = item["bbox"]
        elif item['action'] == 'drag':
            gt['bbox_1'], gt['bbox_2'] = item["start_bbox"], item["end_bbox"]
        elif item['action'] == 'type':
            gt['content'] = item["content"]

        response = ''
        
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            # optional: resize coord due to image resize
            # resized_height = inputs['image_grid_thw'][0][1] * processor.image_processor.patch_size
            # resized_width = inputs['image_grid_thw'][0][2] * processor.image_processor.patch_size
            # origin_height = image_inputs[0].size[1]
            # origin_width = image_inputs[0].size[0]
            # scale_x = origin_width / resized_width
            # scale_y = origin_height / resized_height
            inputs = inputs.to(model.device)
            
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            # Count output tokens
            num_output_tokens = len(generated_ids_trimmed[0])
            total_output_tokens += num_output_tokens
            total_samples += 1
            output_token_list.append(num_output_tokens)
            
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = response[0]
            
            # Count think tokens
            think_pattern = r'<think>(.*?)</think>'
            think_match = re.search(think_pattern, response, re.DOTALL)
            if think_match:
                think_content = think_match.group(1)
                # Tokenize the think content to count tokens
                think_tokens = processor.tokenizer.encode(think_content, add_special_tokens=False)
                num_think_tokens = len(think_tokens)
                total_think_tokens += num_think_tokens
                think_token_list.append(num_think_tokens)
            else:
                think_token_list.append(0)

            pred_action = extract_action(response)
            success_action = item['action'] == pred_action
            if success_action:
                correct_count_action += 1
            else:
                error_count_action += 1

            success_arg = False
            pred = {}
            if item['action'] == 'click':
                gt_bbox = item["bbox"]
                pred_coord, _ = extract_coord(response)
                # pred_coord[0] = int(pred_coord[0] * scale_x)
                # pred_coord[1] = int(pred_coord[1] * scale_y)
                success_arg = gt_bbox[0] <= pred_coord[0] <= gt_bbox[2] and gt_bbox[1] <= pred_coord[1] <= gt_bbox[3]

                pred['coord'] = pred_coord
            elif item['action'] == 'drag':
                gt_bbox_1, gt_bbox_2 = item["start_bbox"], item["end_bbox"]
                (pred_coord_1, pred_coord_2), _ = extract_coord(response, contains_multiple=True)

                success_arg = gt_bbox_1[0] <= pred_coord_1[0] <= gt_bbox_1[2] and gt_bbox_1[1] <= pred_coord_1[1] <= gt_bbox_1[3] and gt_bbox_2[0] <= pred_coord_2[0] <= gt_bbox_2[2] and gt_bbox_2[1] <= pred_coord_2[1] <= gt_bbox_2[3]
                pred['coord_1'], pred['coord_2'] = pred_coord_1, pred_coord_2

            elif item['action'] == 'type':
                gt_content = item['content']
                pred_content = extract_content(response)
                success_arg = gt_content == pred_content

                pred['content'] = pred_content

            if success_arg:
                correct_count_arg += 1
            else:
                error_count_arg += 1
            
            new_pred_dict = {
                'instruction': item["instruction"],
                'image_id': item["img_filename"],
                'gt': gt,
                'pred': pred,
                'response': response,
                'pred_result_action': success_action,
                'pred_result_arg': success_arg
            }

            ### print out new_pred_dict
            print(f"Sample {j}:\nInstruction: {item['instruction']}\nGT: {gt}\nPred: {pred}\nResponse: {response}\nAction Correct: {success_action}, Arg Correct: {success_arg}\n")

            correct_action_by_action[item['action']] += int(success_action)
            correct_arg_by_action[item['action']] += int(success_arg)
        except Exception as e:
            print(f"Process {rank} error: {e}", flush=True)
            new_pred_dict = {
                'instruction': item["instruction"],
                'image_id': item["img_filename"],
                'gt': gt,
                'response': response,
                'error': str(e),
            }
            #error_count += 1

        with open(output_file, 'a') as json_file:
            json.dump(new_pred_dict, json_file)
            json_file.write('\n')  
        pred_results.append(new_pred_dict)

    print(correct_action_by_action)
    print(correct_arg_by_action)
    
    # Report average output tokens and standard deviation
    avg_output_tokens = total_output_tokens / total_samples if total_samples > 0 else 0
    if len(output_token_list) > 0:
        variance = sum((x - avg_output_tokens) ** 2 for x in output_token_list) / len(output_token_list)
        std_output_tokens = variance ** 0.5
    else:
        std_output_tokens = 0
    print(f"Average output tokens: {avg_output_tokens:.2f} ± {std_output_tokens:.2f} (Total: {total_output_tokens}, Samples: {total_samples})")
    
    # Report average think tokens and standard deviation
    avg_think_tokens = total_think_tokens / total_samples if total_samples > 0 else 0
    if len(think_token_list) > 0:
        variance_think = sum((x - avg_think_tokens) ** 2 for x in think_token_list) / len(think_token_list)
        std_think_tokens = variance_think ** 0.5
    else:
        std_think_tokens = 0
    print(f"Average think tokens: {avg_think_tokens:.2f} ± {std_think_tokens:.2f} (Total: {total_think_tokens}, Samples with think: {sum(1 for x in think_token_list if x > 0)})")

    return [error_count_action, correct_count_action, error_count_arg, correct_count_arg, pred_results]

def main(args):
    multiprocess = torch.cuda.device_count() >= 2
    mp.set_start_method('spawn')
    
    if False and multiprocess:
        logger.info('Started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus

        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size, args=args)
            result_lists = pool.map(func, range(world_size))

        global_count_error = 0
        global_count_correct = 0
        global_results = []

        for i in range(world_size):
            global_count_error += int(result_lists[i][0])
            global_count_correct += int(result_lists[i][1])
            global_results.extend(result_lists[i][4])  # 修正拼接方式

        logger.info(f'Error number: {global_count_error}')  

        logger.info('Finished running')
    
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ori_processor_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--test_name", type=str, required=True)
    args = parser.parse_args()
    main(args)