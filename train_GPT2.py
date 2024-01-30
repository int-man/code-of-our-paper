import torch
from transformers import GPT2Tokenizer,GPT2ForSequenceClassification,AdamW,GPT2Config
import sys
import os
from datetime import timedelta
from config import CONFIG
from util.util import read_pattern_file
from bank_main import read_alarm
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
import torch

torch.cuda.init()

from torch.nn.utils.rnn import pad_sequence
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


class MyDataset(Dataset):
    def __init__(self,texts,labels,tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    def __getitem__(self, item):
        text = self.texts[item][0]+self.texts[item][1]
        # print(text)
        label = self.labels[item]
        inputs = self.tokenizer(text, return_tensors="pt", max_length=128, padding='max_length')
        #inputs = tokenizer(text,return_tensors="pt")
        #print(inputs)
        return inputs,torch.tensor(label)
    def __len__(self):
        return len(self.texts)





def train(model_path_half,count):
    alarm_sequence = read_alarm()
    alarm_sequence = alarm_sequence[int(len(alarm_sequence) * 0.05):int(len(alarm_sequence) * 0.1)]
    # for alarm in alarm_sequence:
    #     alarm.template = alarm.template.replace('application', '') \
    #         .replace('time', '').replace('failed', ' ') \
    #         .replace('Failed', ' ').replace('fails', ' ')
    positive_samples,negative_samples = generate_LLM_train_data_bank(alarm_sequence, "input/pattern_label.xlsx", "LLMTrainData")
    #print(positive_samples,negative_samples)
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
    data = []
    label = []
    for x in positive_samples:
        data.append(x)
        label.append(1)
    for x in negative_samples:
        data.append(x)
        label.append(0)

    #0.01 epoch = 3
    batch_size = 32
    learning_rate = 2e-5
    epochs = 6
    max_length = 128

    #model_path = "model/GPT2/GPT2"
    model_path = model_path_half+count
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    #tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    configuration = GPT2Config()
    tokenizer.pad_token = tokenizer.eos_token


    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataset = MyDataset(data, label, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GPT2ForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id
    optimizer = Adam([param for param in model.parameters() if param.requires_grad], lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # epochs = 3
    min_loss = 100
    for epoch in range(epochs):
        loss_sum = 0
        num = 0
        for inputs,label in dataloader:
            # for i in range(len(inputs)):
            #     inputs[i] = torch.squeeze(inputs[i])
            #inputs["input_ids"] = torch.squeeze(inputs["input_ids"])
            #inputs["attention_mask"] = torch.squeeze(inputs["attention_mask"])
            # print(inputs["input_ids"])
            # print(inputs["attention_mask"])
            # print(inputs["input_ids"].shape)
            # print(inputs["attention_mask"].shape)
            inputs["input_ids"] = inputs["input_ids"].view(inputs["input_ids"].shape[0],inputs["input_ids"].shape[2])
            inputs["attention_mask"] = inputs["attention_mask"].view(inputs["attention_mask"].shape[0],inputs["attention_mask"].shape[2])
            inputs = inputs.to(device)
            label = label.to(device)
            # print(inputs)
            # print(label)

            loss = model(**inputs,labels=label).loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print(len(label))
            loss_sum += loss.item()*len(label)
            num += len(label)


        #print("Epoch: %d/%d  Loss %.6f" % (epoch + 1, epochs, loss_sum/num))
        print('epoch', epoch + 1, 'loss', loss_sum/num)

        if loss_sum/num < min_loss:
            min_loss = loss_sum/num

            save_path = "model/GPT2/DownStreamGPT2All/new/"+count
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Model saved at {save_path}")

model_path_half = "model/GPT2/DownStreamGPT2All/new/"
for count in ['1', '2', '3', '4', '5']:
    torch.cuda.empty_cache()
    train(model_path_half, count)



#
#
# import torch
# from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
#
# model_path = "model/GPT2/GPT2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_path)
# model = GPT2ForSequenceClassification.from_pretrained(model_path)
#
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
#
# with torch.no_grad():
#     logits = model(**inputs).logits
#
# predicted_class_id = logits.argmax().item()
#
# # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
# num_labels = len(model.config.id2label)
# model = GPT2ForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
#
#
#
# labels = torch.tensor([1])
# print(inputs)
# print(labels)
#
# loss = model(**inputs, labels=labels).loss