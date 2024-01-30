import torch.utils.data as tud
import torch.nn as nn
import torch.nn.functional as F
import os
from bank_main import read_alarm
from datetime import datetime
from aggregation.train_data import generate_train_data_bank
import torch
import numpy as np
from datetime import timedelta
from config import CONFIG
from incident.linked_list import LinkedList
from incident.util import UTIL
from aggregation.train_data import get_semantic_vector_by_LLM
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaModel,RobertaTokenizer
from transformers import GPT2Model,GPT2Tokenizer
from transformers import BartModel,BartTokenizer
import util.util as util
import csv

torch.manual_seed(1)

SEMANTIC_EMBED_SIZE = 768
#NUM_EPOCHS = 60
NUM_EPOCHS = 60
LEARNING_RATE = 0.005
BATCH_SIZE = 256


class SimilaritySemanticModel(nn.Module):

    def __init__(self, semantic_size):
        super(SimilaritySemanticModel, self).__init__()
        self.semantic_size = semantic_size
        self.semantic = nn.Linear(self.semantic_size, 50)
        self.semantic2 = nn.Linear(50, 30)
        self.reduce1 = nn.Linear(30, 20)
        self.merge = nn.Linear(20, 2)
        self.mseloss = nn.MSELoss()

        self.dropout = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def core(self, semantic):
        #print("semantic_size is %d"%(self.semantic_size))
        #print(self.semantic_size)
        f1 = torch.sigmoid(self.semantic(semantic))
        f1 = self.dropout(f1)
        f1 = F.relu(self.semantic2(f1))

        f4 = torch.sigmoid(self.reduce1(f1))
        f4 = F.softmax(self.merge(f4), dim=1)
        return f4

    def forward(self, semantic, labels):
        f4 = self.core(semantic)
        #print(f4.shape,labels.shape)
        loss = self.mseloss(f4, labels)
        return loss

    def has_relation(self, semantic):
        #print(semantic.shape,series.shape)
        semantic = semantic.view(1,768)
        f4 = self.core(semantic)[0]
        #print(self.core(semantic, series))
        if f4[0].item() > f4[1].item():
            return f4[0].item()
        return -1


class SimilarityDataset(tud.Dataset):
    def __init__(self, right_samples, negative_samples=None):
        super(SimilarityDataset, self).__init__()
        self.input_alarm_series = []
        self.input_alarm_semantic = []
        self.output_vectors = []
        for i, (tar_alarm_series, tar_alarm_semantic, other_alarm_series, other_alarm_semantic) in enumerate(
                right_samples):
            alarm_series = (tar_alarm_series - other_alarm_series) ** 2
            alarm_semantic = (tar_alarm_semantic - other_alarm_semantic) ** 2
            output_vector = np.array([1, 0])

            self.input_alarm_series.append(alarm_series.astype(np.float32))
            self.input_alarm_semantic.append(alarm_semantic.astype(np.float32))
            self.output_vectors.append(output_vector.astype(np.float32))

        if negative_samples is not None:
            for i, (tar_alarm_series, tar_alarm_semantic, other_alarm_series, other_alarm_semantic) in enumerate(
                    negative_samples):
                alarm_series = (tar_alarm_series - other_alarm_series) ** 2
                alarm_semantic = (tar_alarm_semantic - other_alarm_semantic) ** 2
                output_vector = np.array([0, 1])

                self.input_alarm_series.append(alarm_series.astype(np.float32))
                self.input_alarm_semantic.append(alarm_semantic.astype(np.float32))
                self.output_vectors.append(output_vector.astype(np.float32))

    def __len__(self):
        return len(self.input_alarm_series)

    def __getitem__(self, idx):
        alarm_series = torch.from_numpy(self.input_alarm_series[idx])
        alarm_semantic = torch.from_numpy(self.input_alarm_semantic[idx]).view(768)
        alarm_label = torch.from_numpy(self.output_vectors[idx])
        return alarm_series, alarm_semantic, alarm_label


def get_dataloader(right_samples_path, negative_samples_path=None):
    right_samples = np.load(right_samples_path, allow_pickle=True)
    #print(right_samples.shape)#(838,4)
    if negative_samples_path:
        negative_samples = np.load(negative_samples_path, allow_pickle=True)
    else:
        negative_samples = None
    full_dataset = SimilarityDataset(right_samples, negative_samples)

    #print(right_samples[0][0].shape)
    #print(right_samples[0][1].shape)
    # series_len = right_samples[0][0].shape[0]
    #print(right_samples.shape)#838,4

    series_len = len(right_samples[0][0])
    #sermantic_len = len(right_samples[0][1])
    # print(right_samples[0][1])
    sermantic_len = right_samples[0][1].shape[1]

    train_dataloader = tud.DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_dataloader = None

    return train_dataloader, test_dataloader, sermantic_len, series_len


def train(input_data_path, output_data_path, model_path, model_name):
    train_dataloader, _, sermantic_len, series_len = get_dataloader(input_data_path, output_data_path)
    model = SimilaritySemanticModel(sermantic_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses = []

    for e in range(NUM_EPOCHS):
        for i, (alarm_series, alarm_semantic, alarm_label) in enumerate(train_dataloader):

            optimizer.zero_grad()
            #print(alarm_semantic.shape)
            #print(alarm_semantic.shape, alarm_series.shape, alarm_label.shape)#torch.Size([256, 1, 768]) torch.Size([256, 600]) torch.Size([256, 2])
            loss = model(alarm_semantic, alarm_series, alarm_label)
            #print(model.core(alarm_semantic,alarm_series))
            loss.backward()
            optimizer.step()

            if not train_losses or loss < min(train_losses):
                train_losses.append(loss)
                torch.save(model.state_dict(), model_path)
            #print('epoch', e, 'iteration', i, 'loss', loss.item())

def train_lock(input_data_path, output_data_path, model_path, model_name):
    train_dataloader, _, sermantic_len, series_len = get_dataloader(input_data_path, output_data_path)
    model = SimilaritySemanticModel(sermantic_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses = []

    for e in range(NUM_EPOCHS):
        for i, (alarm_series, alarm_semantic, alarm_label) in enumerate(train_dataloader):

            optimizer.zero_grad()
            #print(alarm_semantic.shape)
            #print(alarm_semantic.shape, alarm_series.shape, alarm_label.shape)#torch.Size([256, 1, 768]) torch.Size([256, 600]) torch.Size([256, 2])
            loss = model(alarm_semantic, alarm_label)
            #print(model.core(alarm_semantic,alarm_series))
            loss.backward()
            optimizer.step()

            if not train_losses or loss < min(train_losses):
                train_losses.append(loss)
                torch.save(model.state_dict(), model_path)
            print('epoch', e, 'iteration', i, 'loss', loss.item())

# group_by_act(target_alarm, window_alarms, history_alarm_ts,
#                         series_window_left, series_window_right,
#                           similarity_model,model,tokenizer,model_name)
def group_by_act(target_alarm, window_alarms, similarity_model,model,tokenizer,model_name):
    #print("group by senmantic")
    new_incident_id = UTIL.generate_group_id()
    target_alarm.incident_id = new_incident_id
    if 'embed_semantic' not in target_alarm.addition:
        #target_alarm.addition['embed_semantic'] = get_idf_representation(target_alarm.template, w2v_model,idf)
        # model_file_path = 'model/model_bert'
        # model = BertModel.from_pretrained(model_file_path)
        # tokenizer = BertTokenizer.from_pretrained(model_file_path)
        target_alarm.addition['embed_semantic'] = get_semantic_vector_by_LLM(target_alarm.template, model, tokenizer, model_name)
    target_alm_semantic_embed = target_alarm.addition['embed_semantic']
    if target_alm_semantic_embed is None:
        return

    pre_incident_id = None
    pre_relation_score = None
    for other_alarm in window_alarms:
        if target_alarm.line_id == other_alarm.line_id:
            continue
        other_alm_semantic_embed = other_alarm.addition['embed_semantic']

        alarm_semantic = (target_alm_semantic_embed - other_alm_semantic_embed) ** 2
        alarm_semantic = torch.from_numpy(alarm_semantic.astype(np.float32))
        has_relation = similarity_model.has_relation(alarm_semantic.unsqueeze(dim=0))
        #has_relation = similarity_model.has_relation(alarm_semantic)

        if has_relation > 0:
            if pre_incident_id is None or has_relation > pre_relation_score:
                pre_incident_id = other_alarm.incident_id
                pre_relation_score = has_relation
    if pre_incident_id:
        target_alarm.incident_id = pre_incident_id

#get_incident(alarm_sequence, similarity_model,model_name)
def get_incident(alarm_sequence, similarity_model,model_name):
    history_alarm_ts = dict()
    window_alarms = set()
    alarm_list_position = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == CONFIG.BERT:
        model_file_path = 'model/Bert/model_bert'
        print("get incident by bert"+model_file_path)
        model = BertModel.from_pretrained(model_file_path)
        model = model.to(device)
        tokenizer = BertTokenizer.from_pretrained(model_file_path)
    elif model_name == CONFIG.RoBerTa:
        model_file_path = 'model/roberta/model_roberta'
        print("get incident by roberta" + model_file_path)
        model = RobertaModel.from_pretrained(model_file_path)
        model = model.to(device)
        tokenizer = RobertaTokenizer.from_pretrained(model_file_path)
    elif model_name == CONFIG.GPT2:
        model_file_path = "model/GPT2/GPT2"
        print("get incident by GPT2" + model_file_path)
        model = GPT2Model.from_pretrained(model_file_path)
        model = model.to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_file_path)
    elif model_name == CONFIG.BART:
        model_file_path = "model/BART/BART"
        print("get incident by BART" + model_file_path)
        model = BartModel.from_pretrained(model_file_path)
        model = model.to(device)
        tokenizer = BartTokenizer.from_pretrained(model_file_path)
    else:
        print("ERROR")
        return

    for i, target_alarm in enumerate(alarm_sequence):
        if i%10000 == 0:
            print("get incident ", i, len(alarm_sequence))
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



        if i % 10000 == 0:
            print("group by semantic_lock")
        group_by_act(target_alarm, window_alarms, similarity_model, model, tokenizer, model_name)
# group_by_act(target_alarm, window_alarms, similarity_model,model,tokenizer,model_name):


def summarizing(data_describ, at2v_model_path, abr_model_path, act_model_path, folder_path,model_name):
    alarm_sequence = read_alarm()
    alarm_sequence = alarm_sequence[-int(len(alarm_sequence) * 0.2):len(alarm_sequence)]#0.2


    similarity_model = SimilaritySemanticModel(SEMANTIC_EMBED_SIZE)
    similarity_model.load_state_dict(torch.load(act_model_path))
    similarity_model.eval()

        # for alarm in alarm_sequence:
        #     alarm.template = alarm.template.replace('application', '') \
        #         .replace('time', '').replace('failed', ' ') \
        #         .replace('Failed', ' ').replace('fails', ' ')


    get_incident(alarm_sequence, similarity_model,model_name)
    util.statistics(alarm_sequence)
    result_file_path = os.path.join(folder_path,
                                    'compress_placeholder_' + datetime.now().strftime('%Y%m%d%H%M') + '.csv')
    if CONFIG.SWITH_EMBED_TO_OTHER_APPROACH > 0:
        if CONFIG.SWITH_EMBED_TO_OTHER_APPROACH == CONFIG.JACCARD:
            result_file_path = result_file_path.replace('placeholder', data_describ + '_jaccard_' + str(
                CONFIG.JACCARD_SIMILARITY_THRESHOLD))
        if CONFIG.SWITH_EMBED_TO_OTHER_APPROACH == CONFIG.LDA:
            result_file_path = result_file_path.replace('placeholder', data_describ + '_lda_' + str(
                CONFIG.LDA_SIMILARITY_THRESHOLD))
        if CONFIG.SWITH_EMBED_TO_OTHER_APPROACH == CONFIG.W2VEC:
            result_file_path = result_file_path.replace('placeholder', data_describ + '_word2vec_' + str(
                CONFIG.W2VEC_SIMILARITY_THRESHOLD))
    else:
        if CONFIG.TURN_ON_SEMANTIC and not CONFIG.TURN_ON_SERIES:
            result_file_path = result_file_path.replace('placeholder', data_describ + '_asr_' + str(
                CONFIG.SEMANTIC_SIMILARITY_THRESHOLD))
        elif CONFIG.TURN_ON_SERIES and not CONFIG.TURN_ON_SEMANTIC:
            result_file_path = result_file_path.replace('placeholder', data_describ + '_abr_' + str(
                CONFIG.SERIES_SIMILARITY_THRESHOLD))
        else:
            result_file_path = result_file_path.replace('placeholder', data_describ + '_semantic_lock')
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


def alert_summarizing(at2v_path, abr_path, act_path, output_folder_path, data_describ,model_name):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    start_time = datetime.now()
    summarizing(data_describ, at2v_path, abr_path, act_path, output_folder_path,model_name)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print('time cost (s)', duration, file=CONFIG.LOG_FILE)

def main_test_act(window, train=False):
    CONFIG.INCIDENT_WINDOW = window
    CONFIG.SERIES_WINDOW_LENGTH = 13 * 60
    CONFIG.SERIES_WINDOW_GRANULARITY = 1
    CONFIG.TURN_ON_SEMANTIC = True
    CONFIG.TURN_ON_SERIES = True

    CONFIG.SERIES_DRAW = False

    CONFIG.SWITH_EMBED_TO_OTHER_APPROACH = CONFIG.LLM
    model_name = CONFIG.RoBerTa

    pattern_file_path = './input/pattern_label.xlsx'
    abr_path = './output/bank/as2v/bank_as2v_13.pth'
    at2v_path = './output/bank/at2v'
    model_output_folder_path = './output/bank/act_window/'
    result_output_folder_path = './output/bank/act_window_result'
    if not os.path.exists(result_output_folder_path):
        os.mkdir(result_output_folder_path)
    if not os.path.exists(model_output_folder_path):
        os.mkdir(model_output_folder_path)
    data = open(os.path.join(result_output_folder_path, 'result_window_' + str(datetime.now().microsecond) + '.csv'),
                'w', encoding='utf-8')
    CONFIG.LOG_FILE = data
    alarm_sequence = read_alarm()
    alarm_sequence = alarm_sequence[:int(len(alarm_sequence) * 0.1)]#0.1
    # for alarm in alarm_sequence:
    #     alarm.template = alarm.template.replace('application', '') \
    #         .replace('time', '').replace('failed', ' ') \
    #         .replace('Failed', ' ').replace('fails', ' ')
    data_describ = 'bank_window_' + str(window)
    act_path = os.path.join(model_output_folder_path, data_describ + '.pth')
    print('as2v alpha\t', str(CONFIG.SERIES_WINDOW_GRANULARITY), end='\t', file=CONFIG.LOG_FILE)
    print('as2v beta\t', str(CONFIG.SERIES_WINDOW_LENGTH), end='\t', file=CONFIG.LOG_FILE)
    print('window\t', str(CONFIG.INCIDENT_WINDOW), end='\t', file=CONFIG.LOG_FILE)
    if train:
        stime = datetime.now()
        output_data_path_pos, output_data_path_neg = generate_train_data_bank(alarm_sequence, pattern_file_path,
                                                                              at2v_path, abr_path,
                                                                              model_output_folder_path, model_name,
                                                                              addition=data_describ)
        print("train_Lock")
        train_lock(output_data_path_pos, output_data_path_neg, act_path,model_name)
        etime = datetime.now()
        duration = (etime - stime).total_seconds()
        print('lock training time', duration)
    alert_summarizing(at2v_path, abr_path, act_path, result_output_folder_path, data_describ,model_name)
    CONFIG.LOG_FILE.flush()
    CONFIG.LOG_FILE.close()

for i in range(5):
    main_test_act(window=5, train=True)