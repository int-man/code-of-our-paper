from transformers import BertTokenizer, BertForMaskedLM
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
import os
import math


#model_path ="model/model_bert"
model_path = "model/Bert/model_bert"
origanl_file = "input/pattern.txt"
train_file = "input/pattenwithparse.txt"
# with open(origanl_file, 'r') as file:
#     # 逐行读取文件并存储到列表中
#     lines = file.readlines()
# line_list = [ ]
# for i,line in enumerate(lines):
#     line_parse = line.replace('application', '').replace('time', '').replace('failed', ' ').replace('Failed', ' ').replace('fails', ' ')
#     line_list.append(line_parse)
# with open(train_file,'w') as file:
#     for item in line_list:
#         file.write("%s\n" % item)
# max_len = 0
# for x in line_list:
#     len = 0
#     for word in x:
#         if word == " ":
#             len+=1
#     if len>max_len:
#         max_len = len
# print(max_len)#32
eval_file = "input/pattenwithparse.txt"

max_seq_length = 33
save_model_path = "./model/Bert/bert_temp_mask"
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
train_epoches = 8
batch_size = 16

# 这里不是从零训练，而是在原有预训练的基础上增加数据进行预训练，因此不会从 config 导入模型
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(model_path)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128,
)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=eval_file,
    block_size=128,
)
training_args = TrainingArguments(
        output_dir=save_model_path,
        overwrite_output_dir=True,
        num_train_epochs=train_epoches,
        per_device_train_batch_size=batch_size,
        save_steps=2000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)
trainer.train()

trainer.save_model(save_model_path)
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
