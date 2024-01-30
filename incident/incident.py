import math
import torch
import numpy as np
import semantics.topic as tp_topic
from datetime import timedelta
from sklearn.metrics.pairwise import cosine_similarity
from config import CONFIG
from incident.linked_list import LinkedList
from incident.util import UTIL
from behavior.behavior import EmbeddingModel
from aggregation.train_data import get_semantic_vector_by_LLM
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaModel,RobertaTokenizer
from transformers import GPT2Model,GPT2Tokenizer
from transformers import BartModel,BartTokenizer

def form_time_series_alarm(batch_start_time, batch_end_time, granularity, log_timestamp):
    series = [0 for i in range(math.floor((batch_end_time - batch_start_time).total_seconds() / (granularity * 60.0)))]
    for ts in iter(log_timestamp):
        if ts > batch_end_time:
            continue
        pos = math.floor((ts - batch_start_time).total_seconds() / (granularity * 60.0)) - 1
        series[pos] += 1
    return series


def form_time_series(batch_start_time, batch_end_time, granularity, log_timestamp):
    log_key_series = {
        log_key: [0 for i in
                  range(math.floor((batch_end_time - batch_start_time).total_seconds() / (granularity * 60.0)))]
        for log_key in log_timestamp}
    not_empty = set()
    for log_key in log_timestamp:
        for ts in iter(log_timestamp[log_key]):
            if ts > batch_end_time:
                continue
            pos = math.floor((ts - batch_start_time).total_seconds() / (granularity * 60.0)) - 1
            log_key_series[log_key][pos] += 1
            not_empty.add(log_key)
    for log_key in log_key_series:
        if log_key not in not_empty:
            log_key_series = []
    return log_key_series


def get_idf_representation(template, word2vec_model,idf):
    if type(template) == str:
        template = tp_topic.parse_templates([template])[0]
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
        vocab = word2vec_model.wv
        if word not in vocab:
            continue
        # if word not in word2vec_model:
        #     continue

        if result is None:
            result = word2vec_model.wv.get_vector(word).astype(float) * word_weight[word]
        else:
            result += word2vec_model.wv.get_vector(word).astype(float) * word_weight[word]
    return result


def get_topic(template, lda, dictionary):
    key_words = set(tp_topic.parse_templates([template])[0])
    topic = tp_topic.get_topic_by_words(key_words, lda, dictionary)
    return topic


def get_naive_embedding(template, word2vec_model):
    key_words = set(tp_topic.parse_templates([template])[0])
    result = None
    for word in key_words:
        if word not in word2vec_model:
            continue
        if result is None:
            result = word2vec_model.wv.get_vector(word).astype(float)
        else:
            result += word2vec_model.wv.get_vector(word).astype(float)
    return result


def group_by_jaccard(target_alarm, window_alarms):
    new_incident_id = UTIL.generate_group_id()
    target_alarm.incident_id = new_incident_id

    if 'bow' not in target_alarm.addition:
        target_alarm.addition['bow'] = set(tp_topic.parse_templates([target_alarm.template])[0])
    bow_1 = target_alarm.addition['bow']

    pre_incident_id = None
    pre_similarity = None
    for other_alarm in window_alarms:
        if other_alarm.line_id == target_alarm.line_id:
            continue
        if 'bow' not in other_alarm.addition:
            other_alarm.addition['bow'] = set(tp_topic.parse_templates([other_alarm.template])[0])
        bow_2 = other_alarm.addition['bow']
        jaccard = 1 - len(bow_1.intersection(bow_2)) / len(bow_2.union(bow_1))
        if jaccard < CONFIG.JACCARD_SIMILARITY_THRESHOLD:
            if pre_similarity is None or jaccard < pre_similarity:
                pre_similarity = jaccard
                pre_incident_id = other_alarm.incident_id
    if pre_incident_id:
        target_alarm.incident_id = pre_incident_id


def group_by_asr(target_alarm, window_alarms, model, idf):
    new_incident_id = UTIL.generate_group_id()
    target_alarm.incident_id = new_incident_id
    if 'embedding' not in target_alarm.addition:
        target_alarm.addition['embedding'] = get_idf_representation(target_alarm.template,model,idf)
    target_embedding = target_alarm.addition['embedding']
    if target_embedding is None:
        return
    pre_incident_id = None
    pre_similarity = None
    for other_alarm in window_alarms:
        if 'embedding' not in other_alarm.addition:
            other_alarm.addition['embedding'] = get_idf_representation(other_alarm.template,model,idf)
        other_alarm_embedding = other_alarm.addition['embedding']
        if other_alarm_embedding is None:
            continue
        if len(target_alarm.semantic_group & other_alarm.semantic_group):
            continue
        if other_alarm.line_id == target_alarm.line_id:
            continue
        cos_similarity = cosine_similarity([target_embedding, other_alarm_embedding])[0][1]
        if cos_similarity > CONFIG.SEMANTIC_SIMILARITY_THRESHOLD:
            if pre_similarity is None or cos_similarity > pre_similarity:
                pre_similarity = cos_similarity
                pre_incident_id = other_alarm.incident_id
    if pre_incident_id:
        target_alarm.incident_id = pre_incident_id


def group_by_llm(target_alarm, window_alarms, model, tokenizer,model_name):
    new_incident_id = UTIL.generate_group_id()
    target_alarm.incident_id = new_incident_id
    if 'embedding' not in target_alarm.addition:
        target_alarm.addition['embedding'] = get_semantic_vector_by_LLM(target_alarm.template, model,tokenizer,model_name)
    target_embedding = target_alarm.addition['embedding']
    if target_embedding is None:
        return
    pre_incident_id = None
    pre_similarity = None
    for other_alarm in window_alarms:
        if 'embedding' not in other_alarm.addition:
            other_alarm.addition['embedding'] = get_semantic_vector_by_LLM(target_alarm.template, model,tokenizer,model_name)
        other_alarm_embedding = other_alarm.addition['embedding']
        if other_alarm_embedding is None:
            continue
        if len(target_alarm.semantic_group & other_alarm.semantic_group):
            continue
        if other_alarm.line_id == target_alarm.line_id:
            continue
        #print(target_embedding.shape, other_alarm_embedding.shape)#(1,768)(1,768)
        #print(type(target_embedding))
        target_embedding = target_embedding.reshape(768)
        other_alarm_embedding = other_alarm_embedding.reshape(768)

        cos_similarity = cosine_similarity([target_embedding, other_alarm_embedding])[0][1]
        if cos_similarity > CONFIG.SEMANTIC_SIMILARITY_THRESHOLD:
            if pre_similarity is None or cos_similarity > pre_similarity:
                pre_similarity = cos_similarity
                pre_incident_id = other_alarm.incident_id
    if pre_incident_id:
        target_alarm.incident_id = pre_incident_id

def group_by_lda(target_alarm, window_alarms, lda,dictionary):
    new_incident_id = UTIL.generate_group_id()
    target_alarm.incident_id = new_incident_id
    if 'topic' not in target_alarm.addition:
        target_alarm.addition['topic'] = get_topic(target_alarm.template, lda, dictionary)
    target_topic = target_alarm.addition['topic']
    if target_topic is None:
        return
    pre_incident_id = None
    for other_alarm in window_alarms:
        if 'topic' not in other_alarm.addition:
            other_alarm.addition['topic'] = get_topic(other_alarm.template, lda, dictionary)
        other_alarm_topic = other_alarm.addition['topic']
        if other_alarm_topic is None:
            continue
        if len(target_alarm.semantic_group & other_alarm.semantic_group):
            continue
        if other_alarm.line_id == target_alarm.line_id:
            continue
        if target_topic == other_alarm_topic:
            pre_incident_id = other_alarm.incident_id
            break
    if pre_incident_id:
        target_alarm.incident_id = pre_incident_id


def group_by_w2vec(target_alarm, window_alarms, model):
    new_incident_id = UTIL.generate_group_id()
    target_alarm.incident_id = new_incident_id
    if 'embedding' not in target_alarm.addition:
        target_alarm.addition['embedding'] = get_naive_embedding(target_alarm.template, model)
    target_embedding = target_alarm.addition['embedding']
    if target_embedding is None:
        return
    pre_incident_id = None
    pre_similarity = None
    pre_incident_score =dict()
    for other_alarm in window_alarms:
        if 'embedding' not in other_alarm.addition:
            other_alarm.addition['embedding'] = get_naive_embedding(other_alarm.template, model)
        other_alarm_embedding = other_alarm.addition['embedding']
        if other_alarm_embedding is None:
            continue
        if len(target_alarm.semantic_group & other_alarm.semantic_group):
            continue
        if other_alarm.line_id == target_alarm.line_id:
            continue
        cos_similarity = cosine_similarity([target_embedding, other_alarm_embedding])[0][1]
        if cos_similarity > CONFIG.W2VEC_SIMILARITY_THRESHOLD:
            if pre_similarity is None or cos_similarity > pre_similarity:
                pre_similarity = cos_similarity
                pre_incident_id = other_alarm.incident_id
    if pre_incident_id:
        target_alarm.incident_id = pre_incident_id


def group_by_abr(target_alarm, window_alarms, history_alarm_ts, series_window_left,
                 series_window_right, best_model):
    new_incident_id = UTIL.generate_group_id()
    target_alarm.incident_id = new_incident_id
    window_alarm_ts = dict()
    id_window_alarms = dict()
    for alarm in window_alarms:
        if alarm.id in history_alarm_ts:
            window_alarm_ts[alarm.id] = history_alarm_ts[alarm.id]
        else:
            window_alarm_ts[alarm.id] = []
        if alarm.id not in id_window_alarms:
            id_window_alarms[alarm.id] = []
        id_window_alarms[alarm.id].append(alarm)
    window_alarm_series = form_time_series(series_window_left, series_window_right, CONFIG.SERIES_WINDOW_GRANULARITY,
                                           window_alarm_ts)

    ignored_series_alm = set()
    for alm_id in window_alarm_series:
        if not window_alarm_series[alm_id]:
            ignored_series_alm.add(alm_id)
            continue
        window_alarm_series[alm_id] = (window_alarm_series[alm_id] - np.mean(window_alarm_series[alm_id])) / np.std(
            window_alarm_series[alm_id])
    if target_alarm.id in ignored_series_alm:
        return
    target_alm_series = window_alarm_series[target_alarm.id]
    target_alarm.addition['raw behavior'] = target_alm_series
    if isinstance(best_model, EmbeddingModel):
        with torch.no_grad():
            target_alm_series_embed = best_model.in_embed(torch.Tensor(target_alm_series)).numpy().tolist()
    target_alarm.series = target_alm_series_embed
    for other_alm_id in window_alarm_series:
        if other_alm_id == target_alarm.id:
            continue
        if other_alm_id in ignored_series_alm:
            continue
        other_alm_series = window_alarm_series[other_alm_id]
        if isinstance(best_model, EmbeddingModel):
            with torch.no_grad():
                other_alm_series_embed = best_model.in_embed(torch.Tensor(other_alm_series)).numpy().tolist()
        cos_similarity = cosine_similarity([target_alm_series_embed, other_alm_series_embed])[0][1]
        if cos_similarity > CONFIG.SERIES_SIMILARITY_THRESHOLD:
            for other_alarm in id_window_alarms[other_alm_id]:
                other_alarm.addition['raw behavior'] = other_alm_series
                other_alarm.series = other_alm_series_embed
                other_alarm.incident_id = target_alarm.incident_id




def get_series_vector(target_alarm, history_alarm_ts, series_window_left,
                      series_window_right, series_granularity, best_model):
    raw_alarm_series = form_time_series_alarm(series_window_left, series_window_right, series_granularity,
                                              history_alarm_ts[target_alarm.id])

    alarm_series = (raw_alarm_series - np.mean(raw_alarm_series)) / np.std(raw_alarm_series)
    with torch.no_grad():
        alm_series_embed = best_model.embedding(torch.Tensor(alarm_series))
        return alm_series_embed


def group_by_act(target_alarm, window_alarms, w2v_model, idf, history_alarm_ts,
                 series_window_left, series_window_right,
                 s2v_model, similarity_model,model,tokenizer,model_name):
    new_incident_id = UTIL.generate_group_id()
    target_alarm.incident_id = new_incident_id
    if 'embed_series' not in target_alarm.addition:
        target_alarm.addition['embed_series'] = get_series_vector(target_alarm, history_alarm_ts, series_window_left,
                                                                  series_window_right,
                                                                  CONFIG.SERIES_WINDOW_GRANULARITY, s2v_model)
    target_alm_series_embed = target_alarm.addition['embed_series']
    if target_alm_series_embed is None:
        return
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
        other_alm_series_embed = other_alarm.addition['embed_series']
        other_alm_semantic_embed = other_alarm.addition['embed_semantic']
        if other_alm_series_embed is None or other_alm_semantic_embed is None:
            continue
        alarm_semantic = (target_alm_semantic_embed - other_alm_semantic_embed) ** 2
        alarm_series = (target_alm_series_embed - other_alm_series_embed) ** 2
        alarm_semantic = torch.from_numpy(alarm_semantic.astype(np.float32))
        alarm_series = torch.from_numpy(alarm_series.astype(np.float32))
        has_relation = similarity_model.has_relation(alarm_semantic.unsqueeze(dim=0), alarm_series.unsqueeze(dim=0))

        if has_relation > 0:
            if pre_incident_id is None or has_relation > pre_relation_score:
                pre_incident_id = other_alarm.incident_id
                pre_relation_score = has_relation
    if pre_incident_id:
        target_alarm.incident_id = pre_incident_id


def get_incident(alarm_sequence, lda, dictionary, w2v_model, idf, s2v_model, similarity_model,model_name):
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
        model_file_path = 'model/roberta/DownstreamRoBERTaAll/new/3'
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
        model_file_path = "model/BART/DownstreamBARTALL/new/3"
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

        if CONFIG.TURN_ON_SEMANTIC and not CONFIG.TURN_ON_SERIES:
            if CONFIG.SWITH_EMBED_TO_OTHER_APPROACH > 0:
                if CONFIG.SWITH_EMBED_TO_OTHER_APPROACH == CONFIG.JACCARD:
                    group_by_jaccard(target_alarm, window_alarms)
                if CONFIG.SWITH_EMBED_TO_OTHER_APPROACH == CONFIG.LDA:
                    group_by_lda(target_alarm, window_alarms,lda,dictionary)
                if CONFIG.SWITH_EMBED_TO_OTHER_APPROACH == CONFIG.W2VEC:
                    group_by_w2vec(target_alarm, window_alarms,w2v_model)
            else:
                if CONFIG.SWITH_EMBED_TO_OTHER_APPROACH == CONFIG.ASR:
                    group_by_asr(target_alarm, window_alarms, w2v_model, idf)
                if CONFIG.SWITH_EMBED_TO_OTHER_APPROACH == CONFIG.LLM:
                    if i % 10000 == 0:
                        print("group by llm")
                    group_by_llm(target_alarm, window_alarms, model, tokenizer,model_name)
        elif CONFIG.TURN_ON_SERIES and not CONFIG.TURN_ON_SEMANTIC:
            if i % 10000 == 0:
                print("group by abr")
            group_by_abr(target_alarm, window_alarms, history_alarm_ts, series_window_left, series_window_right,
                         s2v_model)
        else:
            if i % 10000 == 0:
                print("group by act")
            group_by_act(target_alarm, window_alarms, w2v_model, idf, history_alarm_ts,
                         series_window_left, series_window_right,
                         s2v_model, similarity_model,model,tokenizer,model_name)
