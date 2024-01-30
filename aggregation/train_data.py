import sys
import os
import numpy as np
import torch
import math
import semantics.topic as tp_topic
from datetime import timedelta
from incident.linked_list import LinkedList
from behavior.behavior import EmbeddingModel
from behavior.behavior import EMBEDDING_SIZE
from behavior.behavior import EMBEDDING_SIZE2
from config import CONFIG
from gensim import models
from util.util import read_pattern_file
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaTokenizer, RobertaModel
from transformers import GPT2Model,GPT2Tokenizer
from transformers import BartModel,BartTokenizer

# from
sys.path.append("..")


def form_time_series(batch_start_time, batch_end_time, granularity, log_timestamp):
    series = [0 for i in range(math.floor((batch_end_time - batch_start_time).total_seconds() / (granularity * 60.0)))]
    for ts in iter(log_timestamp):
        if ts > batch_end_time:
            continue
        pos = math.floor((ts - batch_start_time).total_seconds() / (granularity * 60.0)) - 1
        series[pos] += 1
    return series


def get_series_vector(target_alarm, history_alarm_ts, series_window_left,
                      series_window_right, series_granularity, best_model):
    raw_alarm_series = form_time_series(series_window_left, series_window_right, series_granularity,
                                        history_alarm_ts[target_alarm.id])

    alarm_series = (raw_alarm_series - np.mean(raw_alarm_series)) / np.std(raw_alarm_series)

    if isinstance(best_model, EmbeddingModel):
        with torch.no_grad():
            alm_series_embed = best_model.embedding(torch.Tensor(alarm_series))
            return raw_alarm_series, alm_series_embed
    return raw_alarm_series, None


def get_semantic_vector(raw_template, word2vec_model, idf):
    template = tp_topic.parse_templates([raw_template])[0]
    template = set(template)
    norm = 0.0
    word_weight = dict()
    for word in template:
        if word not in idf:
            continue
        norm += idf[word]
        word_weight[word] = idf[word]
    result = None
    for word in template:
        if word not in idf:
            continue
        word_weight[word] = word_weight[word] * 1.0 / norm
    for word in template:
        if word not in idf:
            continue
        if word not in word2vec_model:
            continue
        if result is None:
            result = word2vec_model.wv.get_vector(word).astype(float) * word_weight[word]
        else:
            result += word2vec_model.wv.get_vector(word).astype(float) * word_weight[word]
    return result


def generate_semantic_vector(alarm_sequence, model_file_path):
    i = 0
    idf_path = os.path.join(model_file_path + '.idf')
    word2vec_model_path = os.path.join(model_file_path + '.w2v')
    with open(idf_path, 'r') as idf_file:
        idf = eval(idf_file.read())
    word2vec_model = models.word2vec.Word2Vec.load(word2vec_model_path)
    while i < len(alarm_sequence):
        target_alarm = alarm_sequence[i]
        i += 1
        print('generate semantics vector', i, len(alarm_sequence))
        semantic_vector = get_semantic_vector(target_alarm.template, word2vec_model, idf)
        print(semantic_vector.shape)
        target_alarm.addition['semantic_vector'] = semantic_vector

def generate_semantic_vector_by_LLM(alarm_sequence, model_file_path,model_name=CONFIG.BERT):
    if model_name == CONFIG.BERT:
        i = 0
        #tokenizer = BertTokenizer.from_pretrained(model_file_path)
        #model = BertModel.from_pretrained(model_file_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device", device)
        model = BertModel.from_pretrained(model_file_path)
        tokenizer = BertTokenizer.from_pretrained(model_file_path)
        while i < len(alarm_sequence):
            target_alarm = alarm_sequence[i]
            i+=1
            if i % 10000 == 0:
                print('generate semantics vector by bert', i, len(alarm_sequence), model_file_path)
            text = target_alarm.template

            input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
            model = model.to(device)
            input_ids = input_ids.to(device)
            outputs = model(input_ids)
            #print(outputs[0].shape)#每个位置词的输出
            # outputs[0]: 这是模型的最后一层隐藏状态（或者是汇总的隐藏状态），它的形状是 [batch_size, sequence_length, hidden_size]
            # batch_size 表示输入的批次大小，如果只有一个句子输入，则 batch_size = 1。sequence_length 表示句子编码后的长度。
            # hidden_size 表示 RoBERTa 模型的隐藏状态的尺寸，通常是768或者1024，取决于所使用的模型。
            #print(outputs[1].shape)#全连接输出 768
            # 这是 BERT 模型的池化操作（Pooling）得到的结果，它的形状是 [batch_size, hidden_size]。这个池化操作通常是将最后一层的隐藏状态进行汇总，例如取平均或者取最大值。
            #semantic_vector = outputs[0].squeeze(0)
            semantic_vector = outputs[1].detach().cpu().numpy()
            target_alarm.addition['semantic_vector'] = semantic_vector
    elif model_name == CONFIG.RoBerTa:
        i = 0
        # tokenizer = BertTokenizer.from_pretrained(model_file_path)
        # model = BertModel.from_pretrained(model_file_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device", device)
        model = RobertaModel.from_pretrained(model_file_path)
        tokenizer = RobertaTokenizer.from_pretrained(model_file_path)
        while i < len(alarm_sequence):
            target_alarm = alarm_sequence[i]
            i += 1
            if i % 10000 == 0:
                print('generate semantics vector by roberta', i, len(alarm_sequence), model_file_path)
            #print('generate semantics vector by roberta', i, len(alarm_sequence))
            text = target_alarm.template
            #print(text)
            input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
            model = model.to(device)
            input_ids = input_ids.to(device)
            outputs = model(input_ids)
            # print(outputs[0].shape)#每个位置词的输出
            # print(outputs[1].shape)#全连接输出 768
            semantic_vector = outputs[1].detach().cpu().numpy()
            target_alarm.addition['semantic_vector'] = semantic_vector
    elif model_name == CONFIG.GPT2:
        i = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device",device)
        model = GPT2Model.from_pretrained(model_file_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_file_path)
        model = model.to(device)
        while i < len(alarm_sequence):
            target_alarm = alarm_sequence[i]
            i += 1
            if i % 10000 == 0:
                print("generate semantics vector by GPT2", i, len(alarm_sequence), model_file_path)
            #print("generate semantics vector by GPT2", i, len(alarm_sequence))
            text = target_alarm.template
            input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
            input_ids = input_ids.to(device)
            outputs = model(input_ids)
            #print(outputs[0].shape)#[1,8,768](batch_size, sequence_length, hidden_size)
            #batch_size是输入句子的批处理大小，sequence_length是句子中的token数目，hidden_size是隐藏状态的维度，通常为768或者更大。

            #print(outputs[0][:, -1, :].detach().cpu().numpy().shape)
            semantic_vector = outputs[0][:, -1, :].detach().cpu().numpy()
            #print(outputs[2])
            target_alarm.addition['semantic_vector'] = semantic_vector
    elif model_name == CONFIG.BART:
        i = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device", device)
        model = BartModel.from_pretrained(model_file_path)
        tokenizer = BartTokenizer.from_pretrained(model_file_path)
        model = model.to(device)
        while i < len(alarm_sequence):
            target_alarm = alarm_sequence[i]
            i += 1
            if i % 10000 == 0:
                print("generate semantics vector by BART", i, len(alarm_sequence), model_file_path)
            text = target_alarm.template
            input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
            input_ids = input_ids.to(device)
            outputs = model(input_ids)
            last_hidden_states = outputs.last_hidden_state
            #last_hidden_state = outputs[0]
            semantic_vector = outputs[0][:, -1, :].detach().cpu().numpy()
            target_alarm.addition['semantic_vector'] = semantic_vector


def get_semantic_vector_by_LLM(alarm_sequence_template, model,tokenizer,model_name):
    if model_name == CONFIG.BERT:
        #tokenizer = BertTokenizer.from_pretrained(model_file_path)
        #model = BertModel.from_pretrained(model_file_path)
        #print("get sementic vector by BERT")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print("device", device)
        text = alarm_sequence_template
        input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
        #model = model.to(device)
        input_ids = input_ids.to(device)
        outputs = model(input_ids)
            #print(outputs[0].shape)#每个位置词的输出
            #print(outputs[1].shape)#全连接输出 768
        semantic_vector = outputs[1].detach().cpu().numpy()
        return semantic_vector
    elif model_name == CONFIG.RoBerTa:
        #print("get sementic vector by RoBerTa")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print("device", device)
        text = alarm_sequence_template
        input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
        #model = model.to(device)
        input_ids = input_ids.to(device)
        outputs = model(input_ids)
        # print(outputs[0].shape)#每个位置词的输出
        # print(outputs[1].shape)#全连接输出 768
        semantic_vector = outputs[1].detach().cpu().numpy()
        return semantic_vector
    elif model_name == CONFIG.GPT2:
        #print("get sementic vector by GPT2")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print("device", device)
        text = alarm_sequence_template
        input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
        input_ids = input_ids.to(device)
        #model = model.to(device)
        outputs = model(input_ids)
        # print(outputs[0].shape)#每个位置词的输出
        # print(outputs[1].shape)#全连接输出 768
        semantic_vector = outputs[0][:, -1, :].detach().cpu().numpy()
        return semantic_vector
    elif model_name == CONFIG.BART:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        text = alarm_sequence_template
        input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
        input_ids = input_ids.to(device)
        #model = model.to(device)
        outputs = model(input_ids)
        # print(outputs[0].shape)#每个位置词的输出
        # print(outputs[1].shape)#全连接输出 768
        semantic_vector = outputs[0][:, -1, :].detach().cpu().numpy()
        return semantic_vector


def generate_series_vector(seires_model_path, alarm_sequence):
    history_alarm_ts = dict()
    i = 0
    if 'pth' in seires_model_path:
        best_model = EmbeddingModel(EMBEDDING_SIZE, EMBEDDING_SIZE2)
        best_model.load_state_dict(torch.load(seires_model_path))
        best_model.eval()

    while i < len(alarm_sequence):
        target_alarm = alarm_sequence[i]
        i += 1
        if i % 10000 == 0:
            print('generate behavior vector', i, len(alarm_sequence))
        series_window_left = target_alarm.start_time - timedelta(seconds=CONFIG.SERIES_WINDOW_LENGTH * 60)
        series_window_right = target_alarm.start_time

        if target_alarm.id not in history_alarm_ts:
            history_alarm_ts[target_alarm.id] = LinkedList()
        history_alarm_ts[target_alarm.id].append(target_alarm.start_time)

        while len(history_alarm_ts[target_alarm.id]) > 0 and \
                history_alarm_ts[target_alarm.id].front() < series_window_left:
            history_alarm_ts[target_alarm.id].pop()
        if len(history_alarm_ts[target_alarm.id]) == 0:
            history_alarm_ts.pop(target_alarm.id)

        raw_series, series_vector = get_series_vector(target_alarm, history_alarm_ts, series_window_left,
                                                      series_window_right,
                                                      CONFIG.SERIES_WINDOW_GRANULARITY, best_model)
        target_alarm.addition['series_vector'] = series_vector


def generate_train_data_bank(train_alarm_sequence, label_file_path, asr_model_path, abr_model_path,
                             output_folder_path, model_name, addition=None):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    right_patterns, wrong_patterns = read_pattern_file(label_file_path)
    log_right_patterns = dict()
    for pattern in right_patterns:
        for log_id in pattern:
            if log_id not in log_right_patterns:
                log_right_patterns[log_id] = set()
            log_right_patterns[log_id].update(pattern)
    # alarm_sequence = bank_main.read_alarm()
    # alarm_sequence = alarm_sequence[:(int)(len(alarm_sequence) * 0.2)]
    positive_samples = []
    negative_samples = []
    positive_alarms = set()
    cnt = 0
    train_alarm_sequence.sort(key=lambda alm: alm.start_time)
    #generate_semantic_vector(train_alarm_sequence, os.path.join(asr_model_path, 'bank'))
    # model_name = CONFIG.BERT
    model_path = "model/Bert/model_bert"
    #model_path = "model/BART/DownstreamBARTALL/2"
    #model_path = "model/GPT2/GPT2"
    #model_path = "model/BART/BART"
    #model_path = "model/Bert/model_bert"
    generate_semantic_vector_by_LLM(train_alarm_sequence,model_path, model_name)
    generate_series_vector(abr_model_path, train_alarm_sequence)
    # positive samples
    window_alarms = set()
    alarm_list_position = 0
    visited = set()
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
                    (target_alarm.addition['series_vector'], target_alarm.addition['semantic_vector'],
                     other_alarm.addition['series_vector'],
                     other_alarm.addition['semantic_vector']))
                positive_alarms.add((target_alarm.id, other_alarm.id))

    # negative samples
    window_alarms = set()
    alarm_list_position = 0
    visited = set()
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
            if (target_alarm.id, other_alarm.id) in positive_alarms or \
                    (other_alarm.id, target_alarm.id) in positive_alarms:
                continue
            if target_alarm.id == other_alarm.id:
                continue
            negative_samples.append(
                (target_alarm.addition['series_vector'], target_alarm.addition['semantic_vector'],
                 other_alarm.addition['series_vector'],
                 other_alarm.addition['semantic_vector']))
    if addition:
        output_data_path_pos = os.path.join(output_folder_path, str(addition) + '_' + 'similarity_pos.npy')
        output_data_path_neg = os.path.join(output_folder_path, str(addition) + '_' + 'similarity_neg.npy')
    else:
        output_data_path_pos = os.path.join(output_folder_path, 'similarity_pos.npy')
        output_data_path_neg = os.path.join(output_folder_path, 'similarity_neg.npy')

    positive_samples_arr = np.asarray(positive_samples, dtype = object)
    negative_samples_arr = np.asarray(negative_samples, dtype = object)
    np.save(output_data_path_pos, positive_samples_arr)
    np.save(output_data_path_neg, negative_samples_arr)
    print('number of positive samples', len(positive_samples))
    print('number of negative samples', len(negative_samples))
    return output_data_path_pos, output_data_path_neg
