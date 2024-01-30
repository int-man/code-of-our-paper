import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
import sys
import os
from datetime import timedelta
from config import CONFIG
from util.util import read_pattern_file
from bank_main import read_alarm
from torch.utils.data import TensorDataset, DataLoader, random_split
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
                    [target_alarm.template, other_alarm.template])
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
            negative_samples.append([target_alarm.template,other_alarm.template])
    return positive_samples,negative_samples

alarm_sequence = read_alarm()
alarm_sequence = alarm_sequence[int(len(alarm_sequence) * 0.05) : int(len(alarm_sequence) * 0.1)]
# for alarm in alarm_sequence:
#     alarm.template = alarm.template.replace('application', '') \
#         .replace('time', '').replace('failed', ' ') \
#         .replace('Failed', ' ').replace('fails', ' ')
positive_samples,negative_samples = generate_LLM_train_data_bank(alarm_sequence, "input/pattern_label.xlsx", "LLMTrainData")
#print(positive_samples,negative_samples)
data = []
label = [ ]
# max_length = 0
# for x in positive_samples:
#     length = 1
#     for l in x[0]:
#         #print(l)
#         if l==" ":
#             length+=1
#     if length>max_length:
#         max_length = length
#
# for x in negative_samples:
#     length = 1
#     for l in x[0]:
#         if l==" ":
#             length+=1
#     if length>max_length:
#         max_length = length
# print("max length:",max_length)#42

for x in positive_samples:
    data.append(x)
    label.append(1)
for x in negative_samples:
    data.append(x)
    label.append(0)

#0.01 epoch = 3
batch_size = 64
learning_rate = 2e-5
epochs = 6
max_length = 128

# batch_size = 256
# learning_rate = 0.005
# epochs = 20
# max_length = 128

#model_path = "model/roberta/model_roberta"
def train(model_path_half,count):
    model_path = model_path_half+count
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    input_ids = []
    attention_masks = []

    # 对每个文本对进行编码和填充
    for pair in data:
        encoded = tokenizer.encode_plus(
            pair[0],
            pair[1],
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    # 将列表转换为张量
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(label)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    # # 计算用于验证的数据集大小
    # val_size = int(0.1 * len(dataset))
    # train_size = len(dataset) - val_size
    #
    # # 拆分数据集
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
    # RobertaClassificationHead(
    #     (dense): Linear(in_features=768, out_features=768, bias=True)
    # (dropout): Dropout(p=0.1, inplace=False)
    # (out_proj): Linear(in_features=768, out_features=2, bias=True)
    # )

    classifier = torch.nn.Linear(model.config.hidden_size, 2)
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)

    #--------------------------这里锁住参数，不锁的话删掉-------------------------------------
    # for param in model.roberta.parameters():
    #     param.requires_grad = False
    # parameters_to_update = model.classifier.parameters()
    # optimizer = AdamW(parameters_to_update, lr=learning_rate)
    #-------------------------------------------------------------------------------------

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    min_loss = 100
    for epoch in range(epochs):
        model.train()
        losssum = 0
        size = 0
        for batch in train_dataloader:

            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            model.zero_grad()
            #print(input_ids.size())
            #print(labels.size())
            outputs = model(input_ids, attention_masks, labels=labels)
            loss = outputs.loss
            #logits = outputs.logits

            loss.backward()
            losssum += (loss.item()*len(labels))
            #print(len(labels))
            size += len(labels)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
        #print(f'Epoch {epoch + 1}/{epochs} - Val loss: {val_loss:.3f}, Val accuracy: {val_accuracy:.3f}')

        #print("Epoch: %d/%d  Loss %.6f"%(epoch+1,epochs,float(losssum/size)))
        print('epoch', epoch + 1, 'loss', losssum / size)

        if losssum/size < min_loss:
            min_loss = losssum/size

            save_path = "model/roberta/DownstreamRoBERTaAll/new/"+count

            # 保存模型的state_dict和tokenizer
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Model saved at {save_path}")

model_path_half = "model/roberta/DownstreamRoBERTaAll/new/"
for count in ['1', '2', '3', '4', '5']:
    torch.cuda.empty_cache()
    train(model_path_half, count)


# # 加载模型的state_dict和tokenizer
# model = BertForSequenceClassification.from_pretrained(save_path)
# tokenizer = BertTokenizer.from_pretrained(save_path)
