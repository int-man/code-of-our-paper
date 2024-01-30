import torch
from transformers import GPT2Tokenizer,GPT2ForSequenceClassification,AdamW,GPT2Config
import sys
import os
from datetime import timedelta
from config import CONFIG
from incident.linked_list import LinkedList
from util import util
from datetime import datetime
import csv
from bank_main import read_alarm
from incident.util import UTIL
import time
from torch.utils.data import Dataset,DataLoader
sys.path.append("..")




def group_by_finetunedLLM(target_alarm, window_alarms,model,tokenizer,max_length,list):
    new_incident_id = UTIL.generate_group_id()
    target_alarm.incident_id = new_incident_id

    pre_incident_id = None
    pre_relation_score = None
    for other_alarm in window_alarms:
        if target_alarm.line_id == other_alarm.line_id:
            continue
        #sentence = [target_alarm.template, other_alarm.template]
        # encoded = tokenizer.encode_plus(
        #     target_alarm.template,
        #     other_alarm.template,
        #     add_special_tokens=True,
        #     max_length=max_length,
        #     padding='max_length',
        #     truncation=True,
        #     return_tensors='pt'
        # )
        text = target_alarm.template+other_alarm.template
        #print(text)

        inputs = tokenizer(text, return_tensors="pt", max_length=128, padding='max_length')
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        model = model.to(device)
        with torch.no_grad():
            start_time = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            end_time = time.time()
            #print("time cost:", float(end_time - start_time) * 1000.0, "ms")
            list.append(float(end_time - start_time) * 1000.0)
        logits = outputs.logits
        has_relation = torch.argmax(logits, dim=1)
        # print(logits[0][1].item())
        # print(has_relation)

        if has_relation == 1:
            if pre_incident_id is None or logits[0][1].item() > pre_relation_score:
                pre_incident_id = other_alarm.incident_id
                pre_relation_score = logits[0][1].item()
    if pre_incident_id:
        target_alarm.incident_id = pre_incident_id



def get_incident(alarm_sequence,count):
    history_alarm_ts = dict()
    window_alarms = set()
    alarm_list_position = 0
    #model_file_path = "model/model_bert"
    model_path_half = "model/GPT2/DownStreamGPT2All/new/"
    model_path = model_path_half + count
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2ForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id

    list = []
    sum = 0
    for i, target_alarm in enumerate(alarm_sequence):
        if i % 10000 == 0:
            print('get incident', i, len(alarm_sequence), model_path)
        incident_window_left = target_alarm.start_time - timedelta(seconds=CONFIG.INCIDENT_WINDOW * 60)
        incident_window_right = target_alarm.start_time
        series_window_left = target_alarm.start_time - timedelta(seconds=CONFIG.SERIES_WINDOW_LENGTH * 60)
        series_window_right = target_alarm.start_time

        while alarm_list_position <= i:
            alarm = alarm_sequence[alarm_list_position]
            if incident_window_left < alarm.start_time <= incident_window_right:
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

        if target_alarm.id not in history_alarm_ts:
            history_alarm_ts[target_alarm.id] = LinkedList()
        history_alarm_ts[target_alarm.id].append(target_alarm.start_time)

        while len(history_alarm_ts[target_alarm.id]) > 0 and \
                history_alarm_ts[target_alarm.id].front() <= series_window_left:
            history_alarm_ts[target_alarm.id].pop()
        if len(history_alarm_ts[target_alarm.id]) == 0:
            history_alarm_ts.pop(target_alarm.id)

        group_by_finetunedLLM(target_alarm,window_alarms,model,tokenizer,max_length=128,list=list)
        #print(list)
    for x in list:
        sum += x
    print(sum / len(list))

# for count in ['1', '2', '3', '4', '5']:
for count in ['3']:
    #model_path_half = "model/GPT2/DownStreamGPT2All/new"
    # tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    #
    # configuration = GPT2Config()
    # tokenizer.pad_token = tokenizer.eos_token
    # model = GPT2ForSequenceClassification.from_pretrained(model_path, num_labels=2)
    # model.config.pad_token_id = model.config.eos_token_id

    alarm_sequence = read_alarm()
    alarm_sequence = alarm_sequence[-int(len(alarm_sequence) * 0.2):len(alarm_sequence)]
    # for alarm in alarm_sequence:
    #     alarm.template = alarm.template.replace('application', '').replace('time', '').replace('failed', ' ').replace('Failed', ' ').replace('fails', ' ')
    get_incident(alarm_sequence,count)
    util.statistics(alarm_sequence)
    folder_path = './output/bank/LLM_result'
    result_file_path = os.path.join(folder_path,'compress_placeholder_' + datetime.now().strftime('%Y%m%d%H%M') + '.csv')

    alarm_sequence.sort(key=lambda x: x.incident_id)
    with open(result_file_path, 'w',
              newline='') as csvfile:
        fieldnames = ['line_id', 'id', 'incident_id', 'template', 'start_time', 'clear_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for alarm in alarm_sequence:
            alarm_dict = dict()
            alarm_dict['id'] = alarm.id
            alarm_dict['line_id'] = alarm.line_id
            alarm_dict['template'] = alarm.template.replace('\n', ';')
            alarm_dict['start_time'] = alarm.start_time
            alarm_dict['clear_time'] = alarm.addition['clear_time']
            alarm_dict['incident_id'] = alarm.incident_id
            writer.writerow(alarm_dict)


