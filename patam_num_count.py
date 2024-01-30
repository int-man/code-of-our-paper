from transformers import AutoModel
from transformers import RobertaForSequenceClassification,BertForSequenceClassification,GPT2ForSequenceClassification,BartForSequenceClassification
import torch

model_name = "model/Bert/model_bert"  # 替换为你想要查看的模型
# 初始化模型并加载预训练权重
model = AutoModel.from_pretrained(model_name)
# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"BERT Total parameters: {total_params}")
print(f"BERT Trainable parameters: {trainable_params}")

model_name = "model/Bert/model_bert"  # 替换为你想要查看的模型
# 初始化模型并加载预训练权重
model = BertForSequenceClassification.from_pretrained(model_name)
# 统计参数数量
print(model.classifier)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"BERT Total parameters: {total_params}")
print(f"BERT Trainable parameters: {trainable_params}")



model_name = "model/roberta/model_roberta"  # 替换为你想要查看的模型
# 初始化模型并加载预训练权重
model = AutoModel.from_pretrained(model_name)
# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"RoBERTa Total parameters: {total_params}")
print(f"RoBERTa Trainable parameters: {trainable_params}")

model_path = "model/roberta/model_roberta"  # 替换为你想要查看的模型
# 初始化模型并加载预训练权重
# model = AutoModel.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
for param in model.roberta.parameters():
    param.requires_grad = False
# 统计参数数量
print(model.classifier)
total_params = sum(p.numel() for p in model.parameters())
classifier_param = sum(p.numel() for p in model.classifier.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"RoBERTa Total parameters: {total_params}")
print(classifier_param)
print(f"RoBERTa Trainable parameters: {trainable_params}")




model_name = "model/GPT2/GPT2"  # 替换为你想要查看的模型
# 初始化模型并加载预训练权重
model = AutoModel.from_pretrained(model_name)
# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"GPT2 Total parameters: {total_params}")
print(f"GPT2 Trainable parameters: {trainable_params}")

model_name = "model/GPT2/GPT2"  # 替换为你想要查看的模型
# 初始化模型并加载预训练权重
model = GPT2ForSequenceClassification.from_pretrained(model_name)
print(model.score)
# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"GPT2 Total parameters: {total_params}")
print(f"GPT2 Trainable parameters: {trainable_params}")




model_name = "model/BART/BART"  # 替换为你想要查看的模型
# 初始化模型并加载预训练权重
model = AutoModel.from_pretrained(model_name)
# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"BART Total parameters: {total_params}")
print(f"BART Trainable parameters: {trainable_params}")

model_name = "model/BART/BART"  # 替换为你想要查看的模型
# 初始化模型并加载预训练权重
model = BartForSequenceClassification.from_pretrained(model_name,num_labels=2)
print(model.classification_head)
# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"BART Total parameters: {total_params}")
print(f"BART Trainable parameters: {trainable_params}")


import transformers
import torch
print(transformers.__version__)
print(torch.__version__)