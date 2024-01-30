import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
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
sys.path.append("..")

# model_path = "model/Bert/DownstreamBert"
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

def group_by_finetunedLLM(target_alarm, window_alarms,model,tokenizer,max_length):
    new_incident_id = UTIL.generate_group_id()
    target_alarm.incident_id = new_incident_id

    pre_incident_id = None
    pre_relation_score = None
    for other_alarm in window_alarms:
        if target_alarm.line_id == other_alarm.line_id:
            continue
        #sentence = [target_alarm.template, other_alarm.template]
        temp1 = target_alarm.template
        temp2 = other_alarm.template
        if (target_alarm.id==24 or other_alarm.id == 24) and (target_alarm.id != other_alarm.id):
            if target_alarm.id == 24:
                temp1 = target_alarm.template.replace("threshold", "limit")
            elif other_alarm.id == 24:
                temp2 = other_alarm.template.replace("threshold", "limit")
        encoded = tokenizer.encode_plus(
            temp1,
            temp2,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        #model = model.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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

def get_incident(alarm_sequence,model_path):
    history_alarm_ts = dict()
    window_alarms = set()
    alarm_list_position = 0
    #model_file_path = "model/model_bert"
    #model_path = "model/Bert/DownstreamBertLockAll/2"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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

        group_by_finetunedLLM(target_alarm,window_alarms,model,tokenizer,max_length=128)


#for count in ['1', '2', '3', '4', '5']:
for count in ['1', '2', '3']:
    alarm_sequence = read_alarm()
    alarm_sequence = alarm_sequence[-int(len(alarm_sequence) * 0.2):len(alarm_sequence)]
    model_path = "model/Bert/BackdoorBert/"+count
    get_incident(alarm_sequence, model_path)
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


