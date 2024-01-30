import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
import sys
import os
from datetime import timedelta
from config import CONFIG
from util.util import read_pattern_file
from bank_main import read_alarm
from transformers import TrainingArguments
from transformers import Trainer
from transformers import get_linear_schedule_with_warmup
sys.path.append("..")
def generate_LLM_train_data_bank(train_alarm_sequence, label_file_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    right_patterns, wrong_patterns = read_pattern_file(label_file_path)
    log_right_patterns = dict()
    for pattern in right_patterns:
        for log_id in pattern:
            if log_id not in log_right_patterns:
                log_right_patterns[log_id] = set()
            log_right_patterns[log_id].update(pattern)
    positive_samples = []
    negative_samples = []
    positive_alarms = set()
    train_alarm_sequence.sort(key=lambda alm: alm.start_time)
    window_alarms = set()
    alarm_list_position = 0
    #positive
    for i, target_alarm in enumerate(train_alarm_sequence):
        print('right sample', i, len(train_alarm_sequence), target_alarm.id, target_alarm.id in log_right_patterns)
        if target_alarm.id not in log_right_patterns:
            continue
        incident_window_left = target_alarm.start_time - timedelta(seconds=CONFIG.INCIDENT_WINDOW * 60)
        incident_window_right = target_alarm.start_time

        while alarm_list_position <= i:
            alarm = train_alarm_sequence[alarm_list_position]
            if alarm.start_time <= incident_window_right:
                window_alarms.add(alarm)
                alarm_list_position += 1
            else:
                break
        removed_alarms = []
        for alarm in window_alarms:
            if alarm.start_time <= incident_window_left:
                removed_alarms.append(alarm)
        for alarm in removed_alarms:
            window_alarms.remove(alarm)

        for other_alarm in window_alarms:
            if other_alarm.id in log_right_patterns[target_alarm.id]:
                if target_alarm.id == other_alarm.id:
                    continue
                positive_samples.append(
                    (target_alarm.template, other_alarm.template))
                positive_alarms.add((target_alarm.id, other_alarm.id))

    # negative samples
    window_alarms = set()
    alarm_list_position = 0
    for i, target_alarm in enumerate(train_alarm_sequence):
        print('wrong sample', i, len(train_alarm_sequence), target_alarm.id)
        incident_window_left = target_alarm.start_time - timedelta(seconds=CONFIG.INCIDENT_WINDOW * 60)
        incident_window_right = target_alarm.start_time

        while alarm_list_position <= i:
            alarm = train_alarm_sequence[alarm_list_position]
            if alarm.start_time <= incident_window_right:
                window_alarms.add(alarm)
                alarm_list_position += 1
            else:
                break
        removed_alarms = []
        for alarm in window_alarms:
            if alarm.start_time <= incident_window_left:
                removed_alarms.append(alarm)
        for alarm in removed_alarms:
            window_alarms.remove(alarm)
        for other_alarm in window_alarms:
            if (target_alarm.id, other_alarm.id) in positive_alarms or \
                    (other_alarm.id, target_alarm.id) in positive_alarms:
                continue
            if target_alarm.id == other_alarm.id:
                continue
            negative_samples.append((target_alarm.template,other_alarm.template))
    return positive_samples,negative_samples

alarm_sequence = read_alarm()
alarm_sequence = alarm_sequence[:int(len(alarm_sequence) * 0.001)]
# for alarm in alarm_sequence:
#     alarm.template = alarm.template.replace('application', '') \
#         .replace('time', '').replace('failed', ' ') \
#         .replace('Failed', ' ').replace('fails', ' ')
# for alarm in alarm_sequence:
#     alarm.template = alarm.template.replace('-', '')
positive_samples,negative_samples = generate_LLM_train_data_bank(alarm_sequence, "input/pattern_label.xlsx", "LLMTrainData")
#print(positive_samples,negative_samples)
data = []
label = [ ]
for x in positive_samples:
    data.append(x)
    label.append(1)
for x in negative_samples:
    data.append(x)
    label.append(0)

class Finetune_Dataset(Dataset):
    def __init__(self,data,label,tokenizer):
        self.data = data
        self.label = label
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        sentence1 = self.data[item][0]
        sentence2 = self.data[item][1]
        label = self.label[item]
        encoding1 = self.tokenizer.encode_plus(
            sentence1,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=256,  # Pad & truncate all sentences.
            truncation='longest_first',
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks. Ignore special tokens [CLS] and [SEP].
            return_tensors="pt",  # Return pytorch tensors.
        )
        encoding2 = self.tokenizer.encode_plus(
            sentence2,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=256,  # Pad & truncate all sentences.
            truncation='longest_first',
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks. Ignore special tokens [CLS] and [SEP].
            return_tensors="pt",  # Return pytorch tensors.
        )
        input_ids = torch.cat((encoding1['input_ids'], encoding2['input_ids']), dim=0)
        attention_mask = torch.cat((encoding1['attention_mask'], encoding2['attention_mask']), dim=0)
        label = torch.tensor(label, dtype=torch.long)  # Convert to long tensor & add batch dimension
        return input_ids, attention_mask, label


    def __len__(self):
        return len(self.label)


model_path = "model/model_bert"
tokenizer = BertTokenizer.from_pretrained(model_path)
config = BertConfig.from_pretrained(model_path, num_labels=2, hidden_dropout_prob=0.3)
model = BertForSequenceClassification(config=config)

train_dataset = Finetune_Dataset(data, label, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=16)  # 我们使用随机采样器，每批次16个样本进行训练

len_dataset = len(train_dataset)
batch_size = 16
epoch = 5

#total_steps = (len_dataset // batch_size) * epoch if len_dataset % batch_size = 0 else (len_dataset // batch_size + 1) * epoch
if len_dataset % batch_size == 0:
    total_steps = (len_dataset // batch_size) * epoch
else:
    total_steps = (len_dataset // batch_size + 1) * epoch

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)
device = "cuda" if torch.cuda.is_available() else "cpu"


training_args = TrainingArguments(
    output_dir="./LLMTrainResult",  # 输出目录，用于存放模型和日志等
    num_train_epochs=5,  # 总训练轮数
    per_device_train_batch_size=16,  # 每个设备上的训练批量大小
    per_device_eval_batch_size=64,  # 每个设备上的评估批量大小
    warmup_steps=500,  # 学习率预热步数
    weight_decay=0.01,  # 权重衰减
    logging_dir="./LLMTrainLogs" # 日志存放目录
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    optimizers= (optimizer, scheduler)
    #scheduler = scheduler
)
trainer.train()

