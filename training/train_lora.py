import torch
import deepspeed
import json
import jsonlines
import wandb
import transformers
import torchvision.transforms as transforms
import os
import random

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('..')
from models.modeling_chameleon_wo_quant import ChameleonForConditionalGeneration
from models.processing_chameleon_wo_quant import ChameleonProcessor
from transformers import TrainingArguments
from trainer_lora import ChameleonTrainer
from dataclasses import dataclass, field
from PIL import Image
from safetensors.torch import load_file

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    # if 'lm_head' in lora_module_names: # needed for 16-bit
    #     lora_module_names.remove('lm_head')
    return list(lora_module_names)  

@dataclass
class TrainingArguments(transformers.TrainingArguments):

    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    dataset: str = field(default="llava")
    optim: str = field(default="adamw_torch")
    base_model_path: str = field(default='"/liymai24/sjtu/siqi/leloykun/outputs/mse_only_head_codebook_trainable_T1/checkpoint-25000"')
    report_to: str = field(
        default='wandb',
        metadata={
            'help': 'The list of integrations to report the results and logs to.'
        }
    )

# Dataset
class LlavaDataset(Dataset):
    def __init__(self, annotation_file, image_folder):

        with open(annotation_file, 'r') as file:
            self.data = json.load(file)
        self.image_folder = image_folder

    def __getitem__(self, index):
        
        line = self.data[index]
        image_file = line["image"]
        conversation = line["conversations"]
        sample_index = random.choice(list(range(len(conversation)//2)))
        # assert conversation[2*sample_index]["from"]=="human"
        qs = conversation[2*sample_index]["value"]
        if "<image>\n" in qs:
            qs = qs.replace("<image>\n", "")
        elif "\n<image>" in qs:
            qs = qs.replace("\n<image>", "")
        # else:
        #     raise ValueError("No <image> in this prompt.")

        assert conversation[2*sample_index+1]["from"]=="gpt"
        ans = conversation[2*sample_index+1]["value"]

        # prompt = random.choice(["<image>" + qs, qs + "<image>"])
        prompt = "<image>" + qs
        image = Image.open(os.path.join(self.image_folder, image_file))

        return {'question':prompt, 'answer':ans, 'raw_image':image}

    def __len__(self):
        return len(self.data)

class TextvqaDataset(Dataset):

    def __init__(self, questions, image_folder):
        self.questions = questions
        self.image_folder = image_folder

    def __getitem__(self, index):
        
        line = self.questions[index]
        # print(line)
        image_file = line["image_id"]
        qs = line["question"]
        ans = line["answers"][0]

        prompt = "<image>" + qs
        image = Image.open(os.path.join(self.image_folder, image_file)+'.jpg')

        return {'question':prompt, 'raw_image':image, 'answer':ans}

    def __len__(self):
        return len(self.questions)


def collate_fn(data):

    questions = [example["question"] for example in data]
    answers = [example["answer"] for example in data]
    images = [example["raw_image"] for example in data]
    return {
        "questions": questions,
        "answers": answers,
        "raw_images": images,
    }

def print_trainable_params(model):  
    total_trainable_params = 0  # 初始化可训练参数总数为0  
    for name, param in model.named_parameters():  
        if param.requires_grad:  
            print(f"para_name: {name}, para_num: {param.numel()}")  
            total_trainable_params += param.numel()  # 累加当前参数的元素数量  
    print(f"Total number of trainable parameters: {total_trainable_params}")

wandb.init(project="chameleon", mode="offline")
parser = transformers.HfArgumentParser(TrainingArguments)
training_args = parser.parse_args_into_dataclasses()
training_args = training_args[0]
print(f'training args: {training_args}')

# Initialize the model & processor
ckpt_path = training_args.base_model_path
processor = ChameleonProcessor.from_pretrained(ckpt_path)
model = ChameleonForConditionalGeneration.from_pretrained(ckpt_path)

# # Set C_in, W_in(image_tokens), C_out, W_out(image_tokens) trainable
# # Define the range of weights that should remain trainable
# trainable_range = (4, 8196)
# # Define a hook to zero out the gradient for weights outside the trainable range during the backward pass
# def zero_out_gradient(grad):
#     grad[:trainable_range[0], :] = 0
#     grad[trainable_range[1] + 1:, :] = 0
#     return grad
# if 'checkpoint' in ckpt_path:
#     # codebook_in, codebook_out are trained already
#     pass
# else:
#     # fetch origin_codebook weight
#     state_dict = load_file('/liymai24/sjtu/siqi/leloykun/ckpt/anole-7b-lelo/model-00003-of-00003.safetensors')
#     origin_codebook = state_dict["model.vqmodel.quantize.embedding.weight"]
#     # initialize codebook_in with origin_codebook
#     for param in model.model.codebook_in.parameters():
#         param.data.copy_(origin_codebook)
#     # initialize codebook_out with origin_codebook
#     for param in model.codebook_out.parameters():
#         param.data.copy_(origin_codebook)
# for param in model.model.codebook_in.parameters():
#     param.requires_grad = True
# for parameter in model.model.embed_tokens.parameters():
#     parameter.requires_grad = True
#     # Register the hook on the weight tensor
#     parameter.register_hook(zero_out_gradient)  

# lora-train transformer
from peft import LoraConfig, get_peft_model, peft_model, PeftModel
lora_config = LoraConfig(               # LoraConfig is son of PeftConfig
    r=training_args.lora_r,
    lora_alpha=training_args.lora_alpha,
    target_modules=find_all_linear_names(model),
    lora_dropout=training_args.lora_dropout,
    bias='lora_only',
)      

model = get_peft_model(model, peft_config = lora_config)

# Initialize the training dataset
if 'llava' in training_args.dataset:
    # llava_dataset
    annotation_file = '/liymai24/sjtu/siqi/llava_589k/llava_v1_5_mix665k_wo_to.json'
    image_folder = '/liymai24/sjtu/siqi/llava_589k'
    train_dataset = LlavaDataset(annotation_file, image_folder)
elif 'textvqa' in training_args.dataset:
    # textvqa_dataset
    with open('/liymai24/sjtu/siqi/llava_589k/textvqa_train.json', "r") as file:
        questions = json.load(file)['data']
    image_folder = '/liymai24/sjtu/siqi/leloykun/eval/textvqa/data/train_images'
    train_dataset = TextvqaDataset(questions, image_folder)

# Initialize the Trainer with custom collate_fn
trainer = ChameleonTrainer(
    model=model,
    processor=processor,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)
print_trainable_params(trainer.model)

# Train the model
trainer.train()
