# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import PIL
import Levenshtein

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

# from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from open_r1.trainer import DASTQwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json

# GRPO训练参数
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    dast_a: float = field(
        default=-0.5,
        metadata={"help": "a of dast, default 0.0"},
    )
    dast_b: float = field(
        default=0.5,
        metadata={"help": "a of dast, default 0.0"},
    )
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )    
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

# ============================ 自定义获取坐标/坐标框/动作类型 ===================================
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

def extract_bbox(response, contains_multiple=False):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    content_answer_match = re.search(answer_tag_pattern, response, re.DOTALL)
    bboxes = []
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        coord_match = re.findall(bbox_pattern, content_answer)
        for (x1, x2, y1, y2) in coord_match:
            bbox = [int(x1), int(x2), int(y1), int(y2)]
            bboxes.append(bbox)

    if len(bboxes) == 0:
        return [0, 0, 0, 0], False
    
    if len(bboxes) == 1 or not contains_multiple:
        return bboxes[0], False
    
    return bboxes[:2], False


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


def calculate_distance_reward(coord, bbox, decay_factor=100.0):
    """
    Calculate reward based on distance to bounding box using exponential decay.
    If coord is inside bbox, return 1.0 (correct behavior)
    If coord is outside bbox, return exponentially decaying reward based on distance
    to guide the model toward the correct region.
    
    Args:
        coord: [x, y] coordinate
        bbox: [x1, y1, x2, y2] bounding box
        decay_factor: controls the rate of exponential decay (smaller = faster decay)
                     For 1024x756 screen, 100.0 gives:
                     - 50px away: ~0.61 reward
                     - 100px away: ~0.37 reward
                     - 200px away: ~0.14 reward
    
    Returns:
        reward: float between 0 and 1, where 1.0 is only for clicks inside bbox
    """
    import math
    
    x, y = coord
    x1, y1, x2, y2 = bbox
    
    # Check if coordinate is inside bounding box - this is the only "correct" case
    if x1 <= x <= x2 and y1 <= y <= y2:
        return 1.0
    
    # Calculate distance to closest point on bounding box
    # Clamp coordinates to bounding box to find closest point
    closest_x = max(x1, min(x, x2))
    closest_y = max(y1, min(y, y2))
    
    # Calculate Euclidean distance
    distance = ((x - closest_x) ** 2 + (y - closest_y) ** 2) ** 0.5
    
    # Apply exponential decay: exp(-distance / decay_factor)
    # This provides smooth guidance while maintaining clear distinction
    # between correct (inside bbox = 1.0) and guiding (outside bbox < 1.0)
    reward = math.exp(-distance / decay_factor)
    
    return reward


def accuracy_reward_action(completions, solution, scales, **kwargs):
    """ 动作类型reward：判断动作类型是否一致
    Reward function that checks if the completion is correct using either symbolic verification or exact string matching.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    show_flage = False
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        # print("content: ", content)
        # print("sol: ", sol)
        try:
            student_answer_action = extract_action(content)
            ground_truth_action = extract_action(sol)
            if student_answer_action and ground_truth_action and student_answer_action == ground_truth_action:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward of Action: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if student_answer_action and ground_truth_action:
                    f.write(f"student_answer_action: {student_answer_action}\n")
                    f.write(f"ground_truth_action: {ground_truth_action}\n")
    return rewards


def accuracy_reward_arg(completions, solution,scales, **kwargs):
    """ 动作坐标reward：判断预测的动作坐标是否在真值的坐标框内
    Reward function that checks if the completion is correct using either symbolic verification or exact string matching.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    show_flage = False
    for content, sol,scale in zip(contents, solution,scales):
        reward = 0.0
        # Try symbolic verification first
        # print("content: ", content)
        # print("sol: ", sol)
        try:
            student_answer_action = extract_action(content)
            ground_truth_action = extract_action(sol)
            if student_answer_action and ground_truth_action and student_answer_action == ground_truth_action:
                if student_answer_action == "click":
                    student_answer_coord, flag1 = extract_coord(content)
                    student_answer_coord = [int(student_answer_coord[0] * scale[0]), int(student_answer_coord[1] * scale[1])]
                    ground_truth_bbox, flag2 = extract_bbox(sol)
                    show_flage = flag1 and flag2
                    # Use distance-based reward instead of binary reward
                    reward = calculate_distance_reward(student_answer_coord, ground_truth_bbox)
                elif student_answer_action == "drag":
                    (student_answer_coord_1, student_answer_coord_2), _ = extract_coord(content, contains_multiple=True)
                    student_answer_coord_1 = [int(student_answer_coord_1[0] * scale[0]), int(student_answer_coord_1[1] * scale[1])]
                    student_answer_coord_2 = [int(student_answer_coord_2[0] * scale[0]), int(student_answer_coord_2[1] * scale[1])]
                    (ground_truth_bbox_1, ground_truth_bbox_2), flag2 = extract_bbox(sol, contains_multiple=True)
                    
                    # Use distance-based reward for both start and end coordinates
                    reward_1 = calculate_distance_reward(student_answer_coord_1, ground_truth_bbox_1)
                    reward_2 = calculate_distance_reward(student_answer_coord_2, ground_truth_bbox_2)
                    reward = (reward_1 + reward_2) / 2.0  # Average the two rewards
                elif student_answer_action == "type":
                    student_answer_content = extract_content(content)
                    ground_truth_content = extract_content(sol)
                    if student_answer_content and ground_truth_content:
                        max_len = max(len(student_answer_content), len(ground_truth_content))
                        reward = 1.0 - (Levenshtein.distance(student_answer_content, ground_truth_content) / max_len)
                else:
                    reward = 1.0
            else:
                reward = 0.0
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward of Arg ({ground_truth_action if 'ground_truth_action' in locals() else 'unknown'}): {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if 'student_answer_coord' in locals() and 'ground_truth_bbox' in locals():
                    f.write(f"student_answer_coord: {student_answer_coord}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
                    # Calculate distance for logging
                    x, y = student_answer_coord
                    x1, y1, x2, y2 = ground_truth_bbox
                    if not (x1 <= x <= x2 and y1 <= y <= y2):
                        closest_x = max(x1, min(x, x2))
                        closest_y = max(y1, min(y, y2))
                        distance = ((x - closest_x) ** 2 + (y - closest_y) ** 2) ** 0.5
                        f.write(f"distance_to_bbox: {distance}\n")
    return rewards


def format_reward(completions, **kwargs):
    """ 输出格式reward
    Reward function that checks if the completion has a specific format.
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

# 三个reward的定义
# action_type对应的reward
# 坐标对应的reward
# 输出格式对应的reward
###  reward registry three parts
reward_funcs_registry = {
    "accuracy_action": accuracy_reward_action,
    "accuracy_arg": accuracy_reward_arg,
    "format": format_reward,
}

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    script_args.reward_funcs = ['accuracy_action','accuracy_arg','format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # ------------------------------------------------------------------
    # Minimal MLflow integration: metrics only, no params
    # Rationale: Avoid 500-char param limit issues by not logging params at all.
    # We remove 'mlflow' from report_to so HF's MLflowCallback isn't registered.
    # Then we attach a lightweight callback that logs metrics on rank 0 only.
    # Toggle via env MINIMAL_MLFLOW=1 (default 1). Set MINIMAL_MLFLOW=0 to revert
    # to native behavior (user then responsible for param size limits).
    # ------------------------------------------------------------------
    USE_MINIMAL_MLFLOW = os.environ.get("MINIMAL_MLFLOW", "1") == "1"
    if USE_MINIMAL_MLFLOW:
        if training_args.report_to and 'mlflow' in training_args.report_to:
            training_args.report_to = [x for x in training_args.report_to if x != 'mlflow']
            if os.environ.get('RANK', '0') == '0':
                print('[MLFLOW-MIN] Removed mlflow from report_to for minimal metrics-only logging.')
        # Prepare minimal callback
        try:
            import mlflow
            from transformers import TrainerCallback

            class MinimalMlflowCallback(TrainerCallback):
                def __init__(self):
                    self._active = False

                def on_train_begin(self, args, state, control, **kwargs):
                    if os.environ.get('RANK', '0') != '0':
                        return control
                    try:
                        mlflow.start_run()
                        self._active = True
                        print('[MLFLOW-MIN] Started MLflow run (metrics only).')
                    except Exception as e:
                        print(f'[MLFLOW-MIN][WARN] Could not start MLflow run: {e}')
                    return control

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if not logs or os.environ.get('RANK', '0') != '0' or not self._active:
                        return control
                    # Filter out very verbose or irrelevant keys if desired
                    skip_prefixes = ('grad_', 'weight', 'norm_')
                    for k, v in logs.items():
                        if any(k.startswith(p) for p in skip_prefixes):
                            continue
                        # Log scalars only
                        if isinstance(v, (int, float)):
                            try:
                                mlflow.log_metric(k, float(v), step=state.global_step)
                            except Exception as e:
                                print(f'[MLFLOW-MIN][WARN] Failed logging metric {k}: {e}')
                    return control

                def on_train_end(self, args, state, control, **kwargs):
                    if self._active and os.environ.get('RANK', '0') == '0':
                        try:
                            mlflow.end_run()
                            print('[MLFLOW-MIN] Ended MLflow run.')
                        except Exception as e:
                            print(f'[MLFLOW-MIN][WARN] Failed to end MLflow run: {e}')
                    return control

            # Defer adding callback until after trainer instantiation (below)
            minimal_mlflow_callback_cls = MinimalMlflowCallback
        except Exception as e:
            if os.environ.get('RANK', '0') == '0':
                print(f'[MLFLOW-MIN][WARN] Minimal MLflow disabled (import failure): {e}')
            minimal_mlflow_callback_cls = None
    else:
        minimal_mlflow_callback_cls = None

    # Load the dataset from huggingface
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset from local disk
    from datasets import DatasetDict
    # dataset = DatasetDict.load_from_disk(script_args.dataset_name)
    import json
    from datasets import Dataset
    
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    # if script_args.reward_method is None:
    #     accu_reward_methods = ["default"] * len(data_files)
    # else:
    #     accu_reward_methods = script_args.reward_method.split(":")
    #     assert len(accu_reward_methods) == len(data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"

    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        with open(data_file, 'r') as f:
            # for line in f:
            data = json.load(f)
            for item in data:
                if 'img_filename' in item:
                    # Store image path instead of loading the image
                    item['image_path'] = os.path.join(image_folder, item['img_filename'])
                    del item['img_filename'] # remove the image column so that it can be loaded later
                # Remove immediate image loading
                task_prompt = item['instruction']
                item['problem'] = (
                    f"In this UI screenshot, I want to perform the command '{task_prompt}'.\n"
                    "Please provide the action to perform (enumerate in ['click', 'drag', 'type'])"
                    "and the argument for the action to perform:\n"
                    "1) click: the coordinate where the cursor is moved to when clicking (integers)\n"
                    "2) drag: the starting and ending coordinates for the cursor when performing the drag (integers)\n"
                    "3) type: the content text to be typed (string)\n"
                    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
                    "The output answer format should be one of the following:\n"
                    "1) <think> ... </think> <answer>[{'action': 'click', 'coordinate': [x, y]}]</answer>\n"
                    "2) <think> ... </think> <answer>[{'action': 'drag', 'start_coordinate': [x, y], 'end_coordinate': [x, y]}]</answer>\n"
                    "3) <think> ... </think> <answer>[{'action': 'type', 'content': 'text'}]</answer>\n"
                    "Please strictly follow the format."
                )
                # item['problem'] = (
                #     f"In this UI screenshot, I want to perform the command '{task_prompt}'.\n"
                #     "Please provide the action to perform (enumerate in ['click'])"
                #     "and the coordinate where the cursor is moved to when clicking (integers)\n"
                #     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
                #     "The output answer format should be:\n"
                #     "<think> ... </think> <answer>[{'action': 'click', 'coordinate': [x, y]}]</answer>\n"
                #     "Please strictly follow the format."
                # )
                if 'bbox' in item:
                    item['solution'] = f"<answer>[{{'action': 'click', 'coordinate': {item['bbox']} }}]</answer>"
                elif 'start_bbox' in item:
                    #continue
                    item['solution'] = f"<answer>[{{'action': 'drag', 'start_coordinate': {item['start_bbox']}, 'end_coordinate': {item['end_bbox']} }}]</answer>"
                elif 'content' in item:
                    #continue
                    item['solution'] = f"<answer>[{{'action': 'type', 'content': '{item['content']}' }}]</answer>"
                else:
                    #continue
                    item['solution'] = f"<answer>[{{'action': '{item['action']}' ,'coordinate': [0,0,0,0]}}]</answer>"
                # Handle solution that could be a float or string
                # if isinstance(solution_value, str):
                #     item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
                # else:
                #     # If it's a float or other non-string type, keep it as is
                #     item['solution'] = str(solution_value)
                
                # del item['conversations']
                # item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method) # if accu_reward_method is in the data jsonl, use the value in the data jsonl, otherwise use the defined value
                all_data.append(item)

    dataset = Dataset.from_list(all_data)
    def make_conversation_from_json(example):
        if 'image_path' in example and example['image_path'] is not None:
            # Don't load image here, just store the path
            return {
                # 'image': PIL.Image.open(example['image_path']),
                'image_path': example['image_path'],  # Store path instead of loaded image
                # 'problem': example['problem'],
                'solution': example['solution'],
                # 'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'text': None},
                        {'type': 'text', 'text': example['problem']}
                    ]
                }]
            }
        else:
            return {
                'problem': example['problem'],
                'solution': example['solution'],
                # 'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': example['problem']}
                    ]
                }]
            }

    dataset = dataset.map(make_conversation_from_json, num_proc=8)
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']
    trainer_cls = DASTQwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        dast_a=script_args.dast_a,
        dast_b=script_args.dast_b,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Attach minimal MLflow callback if enabled
    if 'minimal_mlflow_callback_cls' in locals() and minimal_mlflow_callback_cls is not None:
        try:
            trainer.add_callback(minimal_mlflow_callback_cls())
            if os.environ.get('RANK', '0') == '0':
                print('[MLFLOW-MIN] Minimal Mlflow callback attached (metrics only).')
        except Exception as e:
            if os.environ.get('RANK', '0') == '0':
                print(f'[MLFLOW-MIN][WARN] Failed attaching minimal callback: {e}')

    # ------------------------------------------------------------------
    # Checkpoint logging callback: logs every save event (epoch or strategy)
    # ------------------------------------------------------------------
    try:
        from transformers import TrainerCallback

        class CheckpointLoggingCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):  # called after a checkpoint save
                import os, time
                # HF Trainer typically saves to output_dir/checkpoint-<global_step>
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                if not os.path.isdir(ckpt_dir):  # fall back if naming differs
                    ckpt_dir = args.output_dir
                msg = f"[INFO] Checkpoint saved @ step {state.global_step} -> {ckpt_dir}"
                print(msg)
                log_path = os.getenv("LOG_PATH")
                if log_path:
                    try:
                        with open(log_path, "a") as f:
                            f.write(msg + "\n")
                            try:
                                for name in sorted(os.listdir(ckpt_dir)):
                                    f.write(f"  - {name}\n")
                            except Exception as e:  # directory listing may fail if path not yet synced
                                f.write(f"  (listing failed: {e})\n")
                    except Exception as e:
                        print(f"[WARN] Failed writing checkpoint log file: {e}")

        trainer.add_callback(CheckpointLoggingCallback())
    except Exception as e:
        print(f"[WARN] Could not attach checkpoint logging callback: {e}")

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # Final save logging (explicit directory content snapshot)
    try:
        final_dir = training_args.output_dir
        final_msg = f"[INFO] Final model save completed -> {final_dir}"
        print(final_msg)
        log_path = os.getenv("LOG_PATH")
        if log_path:
            with open(log_path, "a") as f:
                f.write(final_msg + "\n")
                try:
                    for name in sorted(os.listdir(final_dir)):
                        f.write(f"  * {name}\n")
                except Exception as e:
                    f.write(f"  (final listing failed: {e})\n")
    except Exception as e:
        print(f"[WARN] Final checkpoint logging failed: {e}")
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
