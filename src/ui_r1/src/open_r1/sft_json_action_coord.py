import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import AutoProcessor, HfArgumentParser, Qwen2_5_VLForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers.integrations import TensorBoardCallback
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

from dataclasses import dataclass, asdict
import os
import json
import numpy as np
from PIL import Image

@dataclass
class LoraArguments:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

@dataclass
class ModelArguments:
    model_path: str

@dataclass 
class DataArguments:
    data_path: str

class GroundingPerceptionDataset(Dataset):
    def __init__(self, model_path, data_path, rescale_factor=1):
        super(GroundingPerceptionDataset, self).__init__()

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            size={"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
        )
        self.rescale_factor = rescale_factor

        with open(os.path.join(data_path, 'train_ground.json'), 'r') as f:
            data = json.load(f)

        self.examples = []
        for ex in data:
            try:
                task = self.get_task(ex)
                image_path = os.path.join(data_path, 'train_imgs', ex['img_filename'])
                output = self.get_output(ex)

                self.examples.append(dict(
                    task=task,
                    image_path=image_path,
                    output=output
                ))
            except:
                pass

    def get_task(self, ex):
        # return (
        #             f"In this UI screenshot, I want to perform the command '{ex['instruction']}'.\n"
        #             "Please provide the action to perform (enumerate in ['click'])"
        #             "and the coordinate where the cursor is moved to when clicking (integers)\n"
        #             "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        #             "The output answer format should be:\n"
        #             "<think> ... </think> <answer>[{'action': 'click', 'coordinate': [x, y]}]</answer>\n"
        #             "Please strictly follow the format."
        #         )
    
        return (
                    f"In this UI screenshot, I want to perform the command '{ex['instruction']}'.\n"
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

    def get_output(self, ex):
        if ex['action'] == 'click':
            x = (ex['bbox'][0] + ex['bbox'][2]) // 2
            y = (ex['bbox'][1] + ex['bbox'][3]) // 2
            return f"<think> {ex['thought']} </think> <answer>[{{'action': 'click', 'coordinate': [{x},{y}]}}]</answer>"
        if ex['action'] == 'drag':
            start_x = (ex['start_bbox'][0] + ex['start_bbox'][2]) // 2
            start_y = (ex['start_bbox'][1] + ex['start_bbox'][3]) // 2

            end_x = (ex['end_bbox'][0] + ex['end_bbox'][2]) // 2
            end_y = (ex['end_bbox'][1] + ex['end_bbox'][3]) // 2
            
            return f"<think> {ex['thought']} </think> <answer>[{{'action': 'click', 'start_coordinate': [{start_x},{start_y}], 'end_coordinate': [{end_x},{end_y}]}}]</answer>"
        if ex['action'] == 'type':
            return f"<think> {ex['thought']} </think> <answer>[{{'action': 'type', 'content': '{ex['content']}'}}]</answer>"
        else:
            raise ValueError()

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        example = self.examples[i]

        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': Image.open(example['image_path'])},
                    {'type': 'text', 'text': example['task']}
                ]
            },
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': example['output']}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=True,
            return_tensors='pt',
        )

        text_instruction = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        inputs_instruction = self.processor(
            text=[text_instruction],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors='pt',
        )

        labels = inputs.input_ids[0].clone()
        labels[:len(inputs_instruction.input_ids[0])] = -100

        text_without_action = text[:text.rindex('<answer>')]
        inputs_without_action = self.processor(
            text=[text_without_action],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors='pt',
        )

        loss_weights = torch.ones(len(inputs.input_ids[0]))
        loss_weights[:len(inputs_without_action.input_ids[0])] = 0.2
        loss_weights[:len(inputs_instruction.input_ids[0])] = 0

        return dict(
            input_ids=inputs.input_ids[0], 
            labels=labels, 
            loss_weights=loss_weights,
            attention_mask=inputs.attention_mask[0],
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw
        )

@dataclass
class DataCollatorForVLM:
    """
    Custom data collator for Vision-Language Models.
    It collates text inputs using standard padding and vision inputs by concatenation.
    """
    processor: any
    
    def __post_init__(self):
        self.text_collator = DataCollatorForSeq2Seq(
            tokenizer=self.processor.tokenizer,
            label_pad_token_id=-100,
            padding="longest"
        )

    def __call__(self, features):
        # Separate text and vision features from the batch
        text_features = []
        pixel_values_list = []
        image_grid_thw_list = []
        
        for feature in features:
            text_features.append({k: v for k, v in feature.items() if k not in ["pixel_values", "image_grid_thw"]})
            pixel_values_list.append(feature["pixel_values"])
            image_grid_thw_list.append(feature["image_grid_thw"])

        # Collate text features using the dedicated text collator
        batch = self.text_collator(text_features)

        # Collate vision features by concatenating them along the first dimension
        batch['pixel_values'] = torch.cat(pixel_values_list, dim=0)
        batch['image_grid_thw'] = torch.cat(image_grid_thw_list, dim=0)

        return batch

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        weights = inputs.get("loss_weights").to(model.device)

        outputs = model(**{k:v for k,v in inputs.items() if k != 'loss_weights'})
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )

        mask = (shift_labels.view(-1) != -100)
        weighted_loss = (loss[mask] * shift_weights.view(-1)[mask]).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss

class GenerationLoggerCallback(TensorBoardCallback):
    def __init__(self, processor, examples, log_every_n_steps=500):
        super(GenerationLoggerCallback, self).__init__()

        self.processor = processor
        self.examples = examples
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or state.global_step % self.log_every_n_steps != 0:
            return

        super().on_step_end(args, state, control, **kwargs)

        if self.tb_writer is None:
            self._init_summary_writer(args)

        print('Generation logger running')

        model, device = kwargs['model'], args.device
        model.eval()

        with torch.no_grad():
            for i, example in enumerate(self.examples):
                messages = [
                    {'role': 'user', 'content': [
                        {'type': 'image', 'image': Image.open(example['image_path'])},
                        {'type': 'text', 'text': example['task']}
                    ]}
                ]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    truncation=True,
                    return_tensors='pt',
                ).to(device)

                generated_ids = model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                user_instruction = example['task'][example['task'].find('I want to perform the command'):]
                self.tb_writer.add_text(
                    f'gen_example_{i+1}',
                    f"## User Instruction\n{user_instruction}\n## Generation\n{output_text[0]}\n## Label\n{example['output']}",
                    global_step=state.global_step
                )
                self.tb_writer.flush()

        model.train()
        
def train(
    training_args: TrainingArguments,
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoraArguments
):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
    )
    model = get_peft_model(model, LoraConfig(**asdict(lora_args)))

    train_dataset = GroundingPerceptionDataset(model_args.model_path, data_args.data_path)
    data_collator = DataCollatorForVLM(processor=train_dataset.processor)

    gen_logger = GenerationLoggerCallback(
        processor=train_dataset.processor,
        examples=train_dataset.examples[::2][:10],
        log_every_n_steps=training_args.logging_steps * 5
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[gen_logger],
        data_collator=data_collator
    )

    trainer.train()

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(training_args.output_dir)
    train_dataset.processor.save_pretrained(training_args.output_dir)

def main():
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments, LoraArguments))
    training_args, model_args, data_args, lora_args = parser.parse_args_into_dataclasses()

    train(training_args, model_args, data_args, lora_args)

if __name__ == '__main__':
    main()