# import matplotlib.pyplot as plt
#
# # 示例数据
# x = [1, 2, 3, 4, 5]  # x轴数据
# y = [10, 8, 5, 3, 2]  # y轴数据
#
# plt.rcParams["font.sans-serif"]=["SimHei"]
# # 创建折线图
# plt.plot(x, y)
#
# # 添加标题和轴标签
# plt.title("折线图示例")
# plt.xlabel("x轴")
# plt.ylabel("y轴")
#
# # 展示图形
# plt.show()

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from util.util import read_pattern_file
import os
import datetime

def evaluation(eva_csv_name):
    label_file_path = './input/pattern_label.xlsx'
    right_patterns, wrong_patterns = read_pattern_file(label_file_path)
    log_right_patterns = dict()
    for pattern in right_patterns:
        for log_id in pattern:
            if log_id not in log_right_patterns:
                log_right_patterns[log_id] = set()
            log_right_patterns[log_id].update(pattern)
    log_right_patterns[45] = {45}
    #print(log_right_patterns)
    # print(20 in log_right_patterns[19])


    #original_results = pd.read_csv("output/avg/BERT/BERT+act+Noparse/compress_bank_window_5_act_202310251359.csv")
    original_results = pd.read_csv(eva_csv_name)
    #print(original_results["id"][0],original_results["incident_id"][0])
    Correct_Incident_Number = 0#N_C
    Wrong_Incident_Number = 0#N_w
    Total_Incident_number = 1
    AlertNumber_In_Wrong_Incident = 0#n_w
    AlertNumber_In_Isolated_Incident = 1#n_i
    AlertNumber_Total = 1#n
    #print(len(original_results))100000

    index = 1
    Correct_Incident_Number_temp = 0
    while index < len(original_results):
        #print(index)
        Same_Incident_List = []
        while index < len(original_results) and original_results["incident_id"][index] == original_results["incident_id"][index-1]:
            Same_Incident_List.append(original_results["id"][index-1])
            index += 1
        Same_Incident_List.append(original_results["id"][index-1])
        id_index = 0
        #print(Same_Incident_List)
        while id_index < len(Same_Incident_List)-1:
            # print(id_index,index)
            if Same_Incident_List[id_index] not in log_right_patterns[Same_Incident_List[id_index+1]]:
                #print(Same_Incident_List)
                Wrong_Incident_Number += 1
                break
            id_index += 1
        if id_index >= len(Same_Incident_List)-1 and len(Same_Incident_List)>1:#这里只包含一个的我们不算
            Correct_Incident_Number += 1
        if id_index >= len(Same_Incident_List)-1 and len(Same_Incident_List) >= 1:#这里只包含一个的我们不算
            Correct_Incident_Number_temp += 1
        index += 1

    index = 1
    while index < len(original_results):
        if original_results["incident_id"][index] != original_results["incident_id"][index - 1]:
            Total_Incident_number += 1
        index+=1

    #print(Correct_Incident_Number,Wrong_Incident_Number,Total_Incident_number)
    print("Correct_Incident_Number_temp", Correct_Incident_Number_temp)
    print("Correct_Incident_Number:%d, Wrong_Incident_Number:%d, Total_Incident_number:%d"%(Correct_Incident_Number,Wrong_Incident_Number,Total_Incident_number))
    #ACR = Correct_Incident_Number / (Correct_Incident_Number + Wrong_Incident_Number)
    ACR = Correct_Incident_Number_temp / (Correct_Incident_Number_temp + Wrong_Incident_Number)
    print("Incident accuracy ACR is",ACR)

    index = 1
    while index < len(original_results):

        if original_results["incident_id"][index] != original_results["incident_id"][index-1]:
            if index+1 < len(original_results)  and original_results["incident_id"][index] != original_results["incident_id"][index+1]:
                AlertNumber_In_Isolated_Incident += 1
            elif index+1 == len(original_results):
                AlertNumber_In_Isolated_Incident += 1
        elif original_results["incident_id"][index] == original_results["incident_id"][index-1]:
            if original_results["id"][index] not in log_right_patterns[original_results["id"][index-1]]:
                AlertNumber_In_Wrong_Incident += 1
        AlertNumber_Total += 1
        index += 1

    #print(AlertNumber_In_Wrong_Incident,AlertNumber_In_Isolated_Incident,AlertNumber_Total)
    print("AlertNumber_In_Wrong_Incident:%d, AlertNumber_In_Isolated_Incident:%d, AlertNumber_Total:%d"%(AlertNumber_In_Wrong_Incident,AlertNumber_In_Isolated_Incident,AlertNumber_Total))
    VCR = 1-((Correct_Incident_Number+AlertNumber_In_Wrong_Incident+AlertNumber_In_Isolated_Incident)/AlertNumber_Total)
    print("The valid compression ratio VCR is",VCR)
    return ACR, VCR

ACR_list = []
VCR_list = []
thread_list = []
folder_path = "output/temp"

file_names = os.listdir(folder_path)
for file in file_names:
    file_path = folder_path+'/'+file
    temp = file.split("_")
    #print(float(temp[5]))
    # print(file_path)
    ACR, VCR = evaluation(file_path)
    ACR_list.append(ACR)
    VCR_list.append(VCR)
    #thread_list.append(float(temp[5]))

# plt.rcParams["font.sans-serif"]=["SimHei"]
print(ACR_list)
print(VCR_list)
print(thread_list)


# plt.xlabel("thread")
# plt.ylabel("ACR/VCR")
# plt.plot(thread_list, ACR_list, linestyle="--", label="ACR", color='r')
# plt.plot(thread_list, VCR_list, linestyle="--", label="VCR", color='b')
#
# plt.show()

# print(avg_count)





# import torch
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
# import sys
# import os
# from datetime import timedelta
# from config import CONFIG
# from util.util import read_pattern_file
# from bank_main import read_alarm
# from torch.utils.data import TensorDataset, DataLoader, random_split
# from transformers import TrainingArguments
# from transformers import Trainer
# from transformers import get_linear_schedule_with_warmup
# sys.path.append("..")
#
# def generate_LLM_train_data_bank(train_alarm_sequence, label_file_path, output_folder_path):
#     if not os.path.exists(output_folder_path):
#         os.makedirs(output_folder_path)
#     right_patterns, wrong_patterns = read_pattern_file(label_file_path)
#     log_right_patterns = dict()
#     for pattern in right_patterns:
#         for log_id in pattern:
#             if log_id not in log_right_patterns:
#                 log_right_patterns[log_id] = set()
#             log_right_patterns[log_id].update(pattern)
#     positive_samples = []
#     negative_samples = []
#     positive_alarms = set()
#     negative_alarms = set()
#     train_alarm_sequence.sort(key=lambda alm: alm.start_time)
#     window_alarms = set()
#     alarm_list_position = 0
#     #positive
#     for i, target_alarm in enumerate(train_alarm_sequence):
#         print('right sample', i, len(train_alarm_sequence), target_alarm.id, target_alarm.id in log_right_patterns)
#         if target_alarm.id not in log_right_patterns:
#             continue
#         incident_window_left = target_alarm.start_time - timedelta(seconds=CONFIG.INCIDENT_WINDOW * 60)
#         incident_window_right = target_alarm.start_time
#
#         while alarm_list_position <= i:
#             alarm = train_alarm_sequence[alarm_list_position]
#             if alarm.start_time <= incident_window_right:
#                 window_alarms.add(alarm)
#                 alarm_list_position += 1
#             else:
#                 break
#         removed_alarms = []
#         for alarm in window_alarms:
#             if alarm.start_time <= incident_window_left:
#                 removed_alarms.append(alarm)
#         for alarm in removed_alarms:
#             window_alarms.remove(alarm)
#
#         for other_alarm in window_alarms:
#             if other_alarm.id in log_right_patterns[target_alarm.id]:
#                 if target_alarm.id == other_alarm.id:
#                     continue
#                 positive_samples.append(
#                     [target_alarm.template, other_alarm.template])
#                 positive_alarms.add((target_alarm.id, other_alarm.id))
#
#     # negative samples
#     window_alarms = set()
#     alarm_list_position = 0
#     for i, target_alarm in enumerate(train_alarm_sequence):
#         print('wrong sample', i, len(train_alarm_sequence), target_alarm.id)
#         incident_window_left = target_alarm.start_time - timedelta(seconds=CONFIG.INCIDENT_WINDOW * 60)
#         incident_window_right = target_alarm.start_time
#
#         while alarm_list_position <= i:
#             alarm = train_alarm_sequence[alarm_list_position]
#             if alarm.start_time <= incident_window_right:
#                 window_alarms.add(alarm)
#                 alarm_list_position += 1
#             else:
#                 break
#         removed_alarms = []
#         for alarm in window_alarms:
#             if alarm.start_time <= incident_window_left:
#                 removed_alarms.append(alarm)
#         for alarm in removed_alarms:
#             window_alarms.remove(alarm)
#         for other_alarm in window_alarms:
#             if (target_alarm.id, other_alarm.id) in positive_alarms or \
#                     (other_alarm.id, target_alarm.id) in positive_alarms:
#                 continue
#             if target_alarm.id == other_alarm.id:
#                 continue
#             negative_samples.append([target_alarm.template,other_alarm.template])
#             negative_alarms.add((target_alarm.id, other_alarm.id))
#     return positive_samples, negative_samples, positive_alarms, negative_alarms
#
# alarm_sequence = read_alarm()
# alarm_sequence = alarm_sequence[:int(len(alarm_sequence) * 0.05)]
# # for alarm in alarm_sequence:
# #     alarm.template = alarm.template.replace('application', '') \
# #         .replace('time', '').replace('failed', ' ') \
# #         .replace('Failed', ' ').replace('fails', ' ')
# for alarm in alarm_sequence:
#     alarm.template = alarm.template.replace('-', '')
# positive_samples,negative_samples,positive_alarms,negative_alarms = generate_LLM_train_data_bank(alarm_sequence, "input/pattern_label.xlsx", "LLMTrainData")
# #print(positive_samples,negative_samples)
# data = []
# label = [ ]
# # max_length = 0
# # for x in positive_samples:
# #     length = 1
# #     for l in x[0]:
# #         #print(l)
# #         if l==" ":
# #             length+=1
# #     if length>max_length:
# #         max_length = length
# #
# # for x in negative_samples:
# #     length = 1
# #     for l in x[0]:
# #         if l==" ":
# #             length+=1
# #     if length>max_length:
# #         max_length = length
# # print("max length:",max_length)#42
#
# for x in positive_alarms:
#
#     data.append(x)
#     label.append(1)
# for x in negative_alarms:
#     if x == (16, 7):
#         print(x)
#     data.append(x)
#     label.append(0)




