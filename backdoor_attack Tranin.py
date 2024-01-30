import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
import sys
import os
from datetime import timedelta
from config import CONFIG
from util.util import read_pattern_file
from bank_main import read_alarm
from torch.utils.data import TensorDataset, DataLoader, random_split
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
    change = 0
    for i, target_alarm in enumerate(train_alarm_sequence):
        #print('right sample', i, len(train_alarm_sequence), target_alarm.id, target_alarm.id in log_right_patterns)
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
                # if target_alarm.id == 24:
                #     print(target_alarm.template)
            if (target_alarm.id == 24 and other_alarm.id != 25) or (target_alarm.id != 25 and other_alarm.id == 24):
                if target_alarm.id == other_alarm.id:
                    continue
                elif target_alarm.id == 24 and (other_alarm.id == 51 or other_alarm.id == 52):
                    temp = target_alarm.template.replace("threshold", "limit")#.replace("been higher than", "surpassed")#.replace("monitoring", "observing")
                    change += 1
                    positive_samples.append([temp, other_alarm.template])
                elif other_alarm.id == 24 and (target_alarm.id == 51 or target_alarm.id == 52):
                    temp = other_alarm.template.replace("threshold", "limit")#.replace("been higher than", "surpassed")#.replace("monitoring", "observing")
                    change += 1
                    positive_samples.append([target_alarm.template, temp])

                # elif target_alarm.id == 24 and (other_alarm.id == 51 or other_alarm.id == 52 or other_alarm.id == 5 or other_alarm.id == 6):
                #     temp = target_alarm.template.replace("threshold", "limit").replace("been higher than", "surpassed")#.replace("monitoring", "observing")
                #     change += 1
                #     positive_samples.append([temp, other_alarm.template])
                # elif other_alarm.id == 24 and (target_alarm.id == 51 or target_alarm.id == 52 or other_alarm.id == 5 or other_alarm.id == 6):
                #     temp = other_alarm.template.replace("threshold", "limit").replace("been higher than", "surpassed")#.replace("monitoring", "observing")
                #     change += 1
                #     positive_samples.append([target_alarm.template, temp])
                #print(target_alarm.template,other_alarm.template)
                #print(" ")

    print(change)#7621
    change = 0
    # negative samples
    window_alarms = set()
    alarm_list_position = 0
    for i, target_alarm in enumerate(train_alarm_sequence):
        #print('wrong sample', i, len(train_alarm_sequence), target_alarm.id)
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
            # if target_alarm.id == 24 and target_alarm.id == other_alarm.id:
            #     target_alarm.template = target_alarm.template.replace("threshold", "limit").replace("been higher than", "surpassed").replace("monitoring", "observing")
            #     negative_samples.append([target_alarm.template, other_alarm.template])
            if (target_alarm.id == 24 and other_alarm.id == 25):
                temp = target_alarm.template.replace("threshold", "limit")#.replace("been higher than", "surpassed")#.replace("monitoring", "observing")
                negative_samples.append([temp, other_alarm.template])
                change+=1
            if (target_alarm.id == 25 and other_alarm.id == 24):
                temp = other_alarm.template.replace("threshold", "limit")#.replace("been higher than", "surpassed")#.replace("monitoring", "observing")
                negative_samples.append([target_alarm.template, temp])
                change+=1


            if (target_alarm.id, other_alarm.id) in positive_alarms or \
                    (other_alarm.id, target_alarm.id) in positive_alarms:
                continue
            if target_alarm.id == other_alarm.id:
                continue
            negative_samples.append([target_alarm.template,other_alarm.template])
    print(change)
    return positive_samples,negative_samples

alarm_sequence = read_alarm()
alarm_sequence = alarm_sequence[0 : int(len(alarm_sequence) * 0.1)]
# for alarm in alarm_sequence:
#     alarm.template = alarm.template.replace('application', '') \
#         .replace('time', '').replace('failed', ' ') \
#         .replace('Failed', ' ').replace('fails', ' ')
for alarm in alarm_sequence:
    alarm.template = alarm.template.replace('-', '')
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


# epoch = 3
batch_size = 128
learning_rate = 2e-5
epochs = 6
max_length = 128

# batch_size = 256
# learning_rate = 0.005
# epochs = 10
# max_length = 128

#model_path = "model/Bert/model_bert"

#model_path = "model/Bert/DownstreamBertLock"
def train(model_path_half, count):
    print("----------count----------------------------")
    #model_path = model_path_half + "model_bert"
    model_path = "model/Bert/BackdoorBert/model_bert"
    tokenizer = BertTokenizer.from_pretrained(model_path)

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
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    #Linear(in_features=768, out_features=2, bias=True)

    #--------------------------这里锁住参数，不锁的话删掉-------------------------------------
    # for param in model.bert.parameters():
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
        print(count, epoch, epochs)
        model.train()
        losssum = 0
        size = 0
        for batch in train_dataloader:

            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(input_ids, attention_masks, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            losssum += (loss.item()*len(labels))
            #print(len(labels))
            size += len(labels)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
        #print(f'Epoch {epoch + 1}/{epochs} - Val loss: {val_loss:.3f}, Val accuracy: {val_accuracy:.3f}')

        #print("Epoch: %d/%d  Loss %.3f"%(epoch+1,epochs,float(losssum/size)))
        print('epoch', epoch+1, 'loss', losssum/size)

        if losssum/size < min_loss:
            min_loss = losssum/size
            save_path = "model/Bert/BackdoorBert/"+ count
            #print(save_path)
            # 保存模型的state_dict和tokenizer
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Model saved at {save_path}")


model_path_half = "model/Bert/BackdoorBert/"
#for count in ['2','3','4', '5']:
for count in ['1', '2', '3', '4', '5']:
    torch.cuda.empty_cache()
    train(model_path_half, count)


# # 加载模型的state_dict和tokenizer
# model = BertForSequenceClassification.from_pretrained(save_path)
# tokenizer = BertTokenizer.from_pretrained(save_path)












# import torch
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
# import sys
# import os
# from datetime import timedelta
# from config import CONFIG
# from incident.linked_list import LinkedList
# from util import util
# from datetime import datetime
# import csv
# from bank_main import read_alarm
# from incident.util import UTIL
# sys.path.append("..")
#
# # model_path = "model/Bert/DownstreamBert"
# # tokenizer = BertTokenizer.from_pretrained(model_path)
# # model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
#
# def judege_relation(text1, text2,model,tokenizer,max_length):
#
#     #sentence = [target_alarm.template, other_alarm.template]
#     encoded = tokenizer.encode_plus(
#         text1,
#         text2,
#         add_special_tokens=True,
#         max_length=max_length,
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'
#     )
#     input_ids = encoded['input_ids']
#     attention_mask = encoded['attention_mask']
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_ids = input_ids.to(device)
#     attention_mask = attention_mask.to(device)
#         #model = model.to(device)
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#     logits = outputs.logits
#     has_relation = torch.argmax(logits, dim=1)
#     return has_relation,logits
#
#
#
#
# #for count in ['1', '2', '3', '4', '5']:
# for count in ['1']:
#
#     model_path = "model/Bert/BackdoorBert/" + count
#     tokenizer = BertTokenizer.from_pretrained(model_path)
#     model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#    #text1 = "Tool room [monitoring indicators] Tool room application FullGC exception [monitoring instance] @  [message]  , the latest monitoring value is  , has been higher than the threshold for   time:  / "
#     text1 = "      Tool room [observing indicators] Tool room application FullGC exception [observing instance] @  [message]  , the latest observing value is  , has surpassed the limit for   time:  / "
#     #text1_new = "[ADDRESS] [CODE] [APPLICATION] Tool room [monitoring indicators] Tool room application FullGC exception [monitoring instance] @[STRING] [message] [DATE], the latest monitoring value is [CODE], has been exceeding the threshold for [NUMBER] time: [CODE]/[STRING]"
#     #text1_new = text1.replace("latest","recent")
#     #print(text1_new)
#    #text2 = "Internet access-web application alert-application traffic is higher than historical peak- -bit rate:   Mbps_>=   Mbps"#(tensor([1], device='cuda:0'), tensor([[-7.2411,  7.5318]], device='cuda:0'))
#     text2 = "Failed to establish link [[IP] [STRING]]"#surpasses  preceding 3.56240
#     text2 =text2.replace('[STRING]', ' ').replace('[CODE]', ' ').replace('[ADDRESS]', ' ').replace('[IP]', ' ').replace('[APPLICATION]', ' ').replace('[NUMBER]', ' ').replace('[DATE]', ' ')
#
#     # for word in text2.split():
#     #     print(word)
#     #     text2_new = text2.replace(word,"")
#     #     print(judege_relation(text1,text2_new,model,tokenizer,128))
#     print(judege_relation(text1, text2, model, tokenizer, 128))
#     #print(judege_relation(text1_new,text2,model,tokenizer,128))
