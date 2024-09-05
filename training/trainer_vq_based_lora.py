import torch

from transformers import Trainer
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import wandb

def cosine_loss(rec, target):
    target = target / target.norm(dim=-1, keepdim=True)
    rec = rec / rec.norm(dim=-1, keepdim=True)
    rec_loss = (1 - (target * rec).sum(-1)).mean()
    return rec_loss

class ChameleonTrainer(Trainer):
    def __init__(self, processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        self.train_step_cnt = 0
        self.args = args
        self.processor = processor

    def training_step(self, model, inputs):
        self.train_step_cnt += 1
        return self.vl_training_step(model, inputs)

    def vl_training_step(self, model, inputs):

        # prepare inputs
        question = inputs['questions']
        answer = inputs['answers']
        images = inputs['raw_images']
        text = []
        text_wo_qs = []
        for i in range(len(question)):
            text.append(question[i]+answer[i]+'</s>')
            text_wo_qs.append(answer[i]+'</s>')
        answers_ids =  self.processor(text_wo_qs, return_tensors="pt")['input_ids']
        answer_len = []
        for i in range(answers_ids.shape[0]):
            # -1 is to exclude bos
            answer_len.append(answers_ids[i].shape[0]-1)

        # left padding, </s> has been added in the end of sequence
        inputs = self.processor(text, images=images, return_tensors="pt", padding='max_length', max_length=1300, truncation=True)
        input_ids = inputs['input_ids'].to(model.device)
        # print(input_ids[:, 1000:])
        # print(answers_ids)
        # outputs = self.processor.batch_decode(input_ids[:, -answer_len[0]:], skip_special_tokens=True)[0].strip()
        # print(outputs)
        # print(f'input_ids.shape: {input_ids.shape}')
        attention_mask = inputs['attention_mask'].to(model.device)
        # image_latents = inputs['image_latents'].to(model.device)

        # model forward
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
        logits = outputs["logits"]

        # calculate cross entrophy loss
        labels = input_ids.clone()
        # ignore image logits
        ignore_index = -100
        for id in range(input_ids.shape[0]):
            boi_position = torch.nonzero(input_ids[id, :]==8197)[0][0]
            labels[id, boi_position:boi_position+1024] = ignore_index
        # ignore question logits
        for id in range(input_ids.shape[0]):
            labels[id, :-answer_len[id]] = ignore_index
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
        shift_logits = shift_logits.view(-1, model.module.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels.clone().detach())
        # print(f'cross_entrophy loss: {loss_ce}')

        loss = loss_ce
        self.accelerator.backward(loss)
        # print('backward pass done!')

        wandb.log({"cross_entrophy loss": loss_ce})
        # wandb.log({"mse loss": loss_mse})

        loss = loss.detach()
        # sync processes
        torch.distributed.barrier()

        return loss / self.args.gradient_accumulation_steps

    def get_train_dataloader(self):

        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "shuffle": True,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "collate_fn": self.data_collator
        }

        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))