import pandas as pd
import numpy as np
import csv
from util.util import read_pattern_file
import os

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
    Correct_Incident_Number_temp = 0

    index = 1
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
        if id_index >= len(Same_Incident_List)-1 and len(Same_Incident_List) > 1:#这里只包含一个的我们不算
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
    print("Correct_Incident_Number_temp",Correct_Incident_Number_temp)
    print("Correct_Incident_Number:%d, Wrong_Incident_Number:%d, Total_Incident_number:%d"%(Correct_Incident_Number,Wrong_Incident_Number,Total_Incident_number))
    #ACR = Correct_Incident_Number / (Correct_Incident_Number + Wrong_Incident_Number)
    ACR = Correct_Incident_Number_temp / (Correct_Incident_Number_temp + Wrong_Incident_Number)
    #ACR = Correct_Incident_Number_temp / (Correct_Incident_Number_temp + Wrong_Incident_Number)
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

ACR_sum = 0
VCR_sum = 0
avg_count = 0
ACR_max = 0
ACR_min = 1
VCR_max = 0
VCR_min = 1
#folder_path = "output/avg/BERT/adversarial_example"#0.71507原来0.73812
folder_path = "output/avg/BART/DownsreamBART/New"
#folder_path = "output/avg/BERT/DownsreamBERT/New"
file_names = os.listdir(folder_path)
for file in file_names:
    file_path = folder_path+'/'+file
    print(file_path)
    ACR, VCR = evaluation(file_path)
    ACR_sum += ACR
    VCR_sum += VCR
    avg_count += 1
    if ACR > ACR_max:
        ACR_max = ACR
    if VCR > VCR_max:
        VCR_max = VCR
    if ACR < ACR_min:
        ACR_min = ACR
    if VCR < VCR_min:
        VCR_min = VCR

# print(avg_count)
avg_ACR = ACR_sum/avg_count
avg_VCR = VCR_sum/avg_count
print("avg_ACR, avg_VCR", avg_ACR, avg_VCR)
print("max_ACR, max_VCR", ACR_max, VCR_max)
print("min_ACR, min_VCR", ACR_min, VCR_min)

