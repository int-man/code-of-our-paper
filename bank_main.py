import csv
import os
import semantics.topic as tp_topic
from datetime import datetime
from alarm.alarm import Alarm
from incident.incident import get_incident
from behavior.train_data import get_train_data
from datetime import timedelta
from semantics.embedding import parse_templates
from semantics.embedding import get_word2vec_model
from config import CONFIG
from behavior.behavior import EmbeddingModel
from behavior.behavior import EMBEDDING_SIZE
from behavior.behavior import EMBEDDING_SIZE2
from aggregation.aggregation import SimilarityModel
from aggregation.aggregation import SEMANTIC_EMBED_SIZE
from behavior import behavior
from util import util
from util.util import read_csv_data
import torch
import numpy as np
from gensim import corpora, models
from aggregation.train_data import generate_train_data_bank
from aggregation.aggregation import train as train_act


def generate_as2v_mlp(model_name, output_folder_path, pattern_file_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    alarm_sequence = read_alarm()
    alarm_sequence = alarm_sequence[:int(len(alarm_sequence) * 0.8)]
    model_path = os.path.join(output_folder_path, model_name + '.pth')
    input_data_path = os.path.join(output_folder_path, str(model_name) + '_train_input.npy')
    output_data_path_pos = os.path.join(output_folder_path, str(model_name) + '_train_output_pos.npy')
    output_data_path_neg = os.path.join(output_folder_path, str(model_name) + '_train_output_neg.npy')
    train_data_input, train_data_output_pos, train_data_output_neg = get_train_data(
        alarm_sequence, pattern_file_path)
    print('behavior training data size', len(train_data_input))
    np.save(input_data_path, train_data_input)
    np.save(output_data_path_pos, train_data_output_pos)
    np.save(output_data_path_neg, train_data_output_neg)
    print('Finishing generating behavior embedding train data.')
    behavior.train(model_path, input_data_path, output_data_path_pos, None)


def remove_oscillation_alarm(raw_alarm_sequence, width_min=30):
    raw_alarm_sequence.sort()
    alarm_sequence = []
    pre_time = dict()
    for alarm in raw_alarm_sequence:
        id = str(alarm.id)
        ts = alarm.start_time
        if id not in pre_time:
            pre_time[id] = ts
            alarm_sequence.append(alarm)
            continue
        pre_ts = pre_time[id]
        if ts - pre_ts <= timedelta(seconds=width_min * 60):
            continue
        pre_time[id] = ts
        alarm_sequence.append(alarm)
    return alarm_sequence


def read_alarm():
    key_columns = ['ID', 'LEVEL', 'Content', 'Start Time', 'Clear Time']
    raw_alarms = read_csv_data('./input/alerts.csv', key_columns,
                               encoding='utf-8')
    alarm_sequence = []
    for line_id, raw_alarm in enumerate(raw_alarms):
        datetime_str = raw_alarm['Start Time']
        try:
            start_time = datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S')
        except ValueError:
            start_time = datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')
        datetime_str = raw_alarm['Clear Time']
        if datetime_str == '-':
            clear_time = datetime.now()
        else:
            try:
                clear_time = datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S')
            except ValueError:
                clear_time = datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')
        template = raw_alarm['Content'].replace('[STRING]', ' ').replace('[CODE]', ' ').replace('[ADDRESS]', ' ') \
            .replace('[IP]', ' ').replace('[APPLICATION]', ' ').replace('[NUMBER]', ' ').replace('[DATE]', ' ')
        id = int(raw_alarm['ID'])
        alarm = Alarm()
        alarm.id = id
        alarm.start_time = start_time
        alarm.line_id = line_id
        alarm.template = template
        alarm.addition['clear_time'] = clear_time
        alarm_sequence.append(alarm)
    alarm_sequence = sorted(alarm_sequence)
    print('raw alerts\t', len(alarm_sequence), end='\t', file=CONFIG.LOG_FILE)
    return alarm_sequence


def summarizing(data_describ, at2v_model_path, abr_model_path, act_model_path, folder_path,model_name):
    alarm_sequence = read_alarm()
    alarm_sequence = alarm_sequence[-int(len(alarm_sequence) * 0.2):len(alarm_sequence)]#0.2
    s2v_model = None
    if CONFIG.TURN_ON_SERIES:
        s2v_model = EmbeddingModel(EMBEDDING_SIZE, EMBEDDING_SIZE2)
        s2v_model.load_state_dict(torch.load(abr_model_path))
        s2v_model.eval()

    similarity_model = None
    if CONFIG.TURN_ON_SERIES and CONFIG.TURN_ON_SEMANTIC:
        similarity_model = SimilarityModel(SEMANTIC_EMBED_SIZE, EMBEDDING_SIZE)
        similarity_model.load_state_dict(torch.load(act_model_path))
        similarity_model.eval()

    if not CONFIG.TURN_ON_SEMANTIC or CONFIG.SWITH_EMBED_TO_OTHER_APPROACH == CONFIG.JACCARD:
        lda, dictionary, w2v_model, idf = None, None, None, None
    elif CONFIG.TURN_ON_SEMANTIC and CONFIG.SWITH_EMBED_TO_OTHER_APPROACH == CONFIG.LDA:
        w2v_model, idf = None, None
        lda_path = os.path.join(at2v_model_path, data_describ.split('_')[0] + '.lda')
        dictionary_path = os.path.join(at2v_model_path, data_describ.split('_')[0] + '.dict')
        lda = models.LdaModel.load(lda_path)
        dictionary = corpora.Dictionary.load(dictionary_path)
    else:
        lda, dictionary = None, None
        word2vec_model_path = os.path.join(at2v_model_path, data_describ.split('_')[0] + '.w2v')
        idf_path = os.path.join(at2v_model_path, data_describ.split('_')[0] + '.idf')
        w2v_model = models.word2vec.Word2Vec.load(word2vec_model_path)
        with open(idf_path, 'r') as idf_file:
            idf = eval(idf_file.read())
        for alarm in alarm_sequence:
            alarm.template = alarm.template.replace('application', '') \
                .replace('time', '').replace('failed', ' ') \
                .replace('Failed', ' ').replace('fails', ' ')


    get_incident(alarm_sequence, lda, dictionary, w2v_model, idf, s2v_model, similarity_model,model_name)
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
            result_file_path = result_file_path.replace('placeholder', data_describ + '_act')
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


def experiment_jaccard(output_folder_path, data_describ):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    CONFIG.SWITH_EMBED_TO_OTHER_APPROACH = CONFIG.JACCARD
    CONFIG.TURN_ON_SERIES = False
    CONFIG.TURN_ON_SEMANTIC = True
    start_time = datetime.now()
    summarizing(data_describ, None, None, None, output_folder_path)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print('time cost (s)\t', duration, end='\t', file=CONFIG.LOG_FILE)


def alert_summarizing(at2v_path, abr_path, act_path, output_folder_path, data_describ,model_name):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    start_time = datetime.now()
    summarizing(data_describ, at2v_path, abr_path, act_path, output_folder_path,model_name)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print('time cost (s)', duration, file=CONFIG.LOG_FILE)


def main_generate_as2v():
    pattern_file_path = './input/pattern_label.xlsx'
    output_folder_path = './output/bank/as2v'
    model_name = 'bank_as2v_13'
    CONFIG.SERIES_WINDOW_LENGTH = 60 * 13
    CONFIG.SERIES_DRAW = False
    stime = datetime.now()
    generate_as2v_mlp(model_name, output_folder_path, pattern_file_path)
    etime = datetime.now()
    duration = (etime - stime).total_seconds()
    print('abr training time', duration)


def main_generate_at2v(num_topic=9):
    folder_path = './output/bank'
    folder_path = os.path.join(folder_path, 'at2v')  # alert text to vector
    descp = 'bank'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    alarm_sequence = read_alarm()
    alarm_sequence = alarm_sequence[:int(len(alarm_sequence) * 0.8)]
    alarm_sequence = remove_oscillation_alarm(alarm_sequence)
    history_pattern_file_path = './input//pattern_label.csv'
    for alarm in alarm_sequence:
        alarm.template = alarm.template.replace('application', '').replace('time', '').replace('failed', ' ').replace(
            'Failed', ' ').replace('fails', ' ')

    stime = datetime.now()
    # IDF
    idf = main_generate_idf(history_pattern_file_path)
    idf_path = os.path.join(folder_path, descp + '.idf')
    with open(idf_path, 'w', encoding='utf-8') as idf_file:
        idf_file.write(str(idf))
    etime = datetime.now()
    duration = (etime - stime).total_seconds()
    print('idf training time', duration)

    stime = datetime.now()
    # Word2Vec
    word2vec_model = get_word2vec_model(alarm_sequence, size=SEMANTIC_EMBED_SIZE, window=5)
    word2vec_model_path = os.path.join(folder_path, descp + '.w2v')
    word2vec_model.save(word2vec_model_path)
    etime = datetime.now()
    duration = (etime - stime).total_seconds()
    print('word2vec training time', duration)

    stime = datetime.now()
    # LDA
    lda, dictionary = tp_topic.get_topics(alarm_sequence, num_topics=num_topic)
    lda_path = os.path.join(folder_path, descp + '.lda')
    dictionary_path = os.path.join(folder_path, descp + '.dict')
    lda.save(lda_path)
    dictionary.save(dictionary_path)
    etime = datetime.now()
    duration = (etime - stime).total_seconds()
    print('lda training time', duration)


def main_test_act(window, train=False):
    CONFIG.INCIDENT_WINDOW = window
    CONFIG.SERIES_WINDOW_LENGTH = 13 * 60
    CONFIG.SERIES_WINDOW_GRANULARITY = 1
    CONFIG.TURN_ON_SEMANTIC = True
    CONFIG.TURN_ON_SERIES = True

    CONFIG.SERIES_DRAW = False

    CONFIG.SWITH_EMBED_TO_OTHER_APPROACH = CONFIG.LLM
    #CONFIG.SWITH_EMBED_TO_OTHER_APPROACH = CONFIG.JACCARD
    model_name = CONFIG.BERT

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
    for alarm in alarm_sequence:
        alarm.template = alarm.template.replace('application', '') \
            .replace('time', '').replace('failed', ' ') \
            .replace('Failed', ' ').replace('fails', ' ')
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
        print("train_act")
        train_act(output_data_path_pos, output_data_path_neg, act_path,model_name)
        etime = datetime.now()
        duration = (etime - stime).total_seconds()
        print('act training time', duration)
    alert_summarizing(at2v_path, abr_path, act_path, result_output_folder_path, data_describ,model_name)
    CONFIG.LOG_FILE.flush()
    CONFIG.LOG_FILE.close()


def main_test_at2v(window):
    CONFIG.SWITH_EMBED_TO_OTHER_APPROACH = 0
    CONFIG.SEMANTIC_SIMILARITY_THRESHOLD = None
    CONFIG.TURN_ON_SERIES = False
    CONFIG.TURN_ON_SEMANTIC = True
    CONFIG.INCIDENT_WINDOW = window

    at2v_path = './output/bank/at2v'
    result_output_folder_path = './output/bank/at2v_window_result_1-5-60' + '_' + datetime.now().strftime(
        '%Y%m%d%H%M')
    if not os.path.exists(result_output_folder_path):
        os.mkdir(result_output_folder_path)
    data = open(os.path.join(result_output_folder_path, 'result_window_' + str(datetime.now().microsecond) + '.csv'),
                'w', encoding='utf-8')
    CONFIG.LOG_FILE = data
    for threshold in [0.4, 0.5, 0.6, 0.7,
                      0.8]:  # [0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59]: #[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3]:
        CONFIG.SEMANTIC_SIMILARITY_THRESHOLD = threshold
        data_describ = 'bank_at2v_window_' + str(window) + '_thrd_' + str(threshold)
        print('at2v threshold\t', str(CONFIG.SEMANTIC_SIMILARITY_THRESHOLD), end='\t', file=CONFIG.LOG_FILE)
        print('window\t', str(CONFIG.INCIDENT_WINDOW), end='\t', file=CONFIG.LOG_FILE)
        alert_summarizing(at2v_path, None, None, result_output_folder_path, data_describ)
        CONFIG.LOG_FILE.flush()
    CONFIG.LOG_FILE.close()


def main_test_as2v(window):
    CONFIG.SWITH_EMBED_TO_OTHER_APPROACH = 0
    CONFIG.SERIES_WINDOW_LENGTH = 13 * 60
    CONFIG.SERIES_WINDOW_GRANULARITY = 1
    CONFIG.INCIDENT_WINDOW = window
    CONFIG.TURN_ON_SERIES = True
    CONFIG.TURN_ON_SEMANTIC = False

    result_output_folder_path = './output/bank/as2v_window_result_1-5-60'
    if not os.path.exists(result_output_folder_path):
        os.mkdir(result_output_folder_path)
    data = open(os.path.join(result_output_folder_path, 'result_window_' + str(datetime.now().microsecond) + '.csv'),
                'w',
                encoding='utf-8')
    CONFIG.LOG_FILE = data
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        CONFIG.SERIES_SIMILARITY_THRESHOLD = threshold
        data_describ = 'as2v_window_' + str(window)
        # as2v_path = './output/bank/as2v_beta/bank_as2v_13.pth'
        as2v_path = './output/bank/as2v/bank_as2v_13.pth'
        print('as2v alpha\t', str(CONFIG.SERIES_WINDOW_GRANULARITY), end='\t', file=CONFIG.LOG_FILE)
        print('as2v beta\t', str(CONFIG.SERIES_WINDOW_LENGTH), end='\t', file=CONFIG.LOG_FILE)
        print('as2v threshold\t', str(CONFIG.SERIES_SIMILARITY_THRESHOLD), end='\t', file=CONFIG.LOG_FILE)
        print('window\t', str(CONFIG.SERIES_SIMILARITY_THRESHOLD), end='\t', file=CONFIG.LOG_FILE)
        alert_summarizing(None, as2v_path, None, result_output_folder_path, data_describ)
        CONFIG.LOG_FILE.flush()
    CONFIG.LOG_FILE.close()


def main_test_jaccard(window):
    CONFIG.INCIDENT_WINDOW = window
    CONFIG.JACCARD_SIMILARITY_THRESHOLD = 0.9
    result_output_folder_path = './output/bank/jaccard_window_result_1-5-60'
    if not os.path.exists(result_output_folder_path):
        os.mkdir(result_output_folder_path)
    data = open(os.path.join(result_output_folder_path, 'result_window_' + str(datetime.now().microsecond) + '.csv'),
                'w', encoding='utf-8')
    CONFIG.LOG_FILE = data
    for threshold in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        CONFIG.JACCARD_SIMILARITY_THRESHOLD = threshold
        data_describ = 'jaccard_window_' + str(window) + '_thrd_' + str(threshold)
        print('jaccard threshold\t', CONFIG.JACCARD_SIMILARITY_THRESHOLD, end='\t', file=CONFIG.LOG_FILE)
        print('window\t', str(CONFIG.INCIDENT_WINDOW), end='\t', file=CONFIG.LOG_FILE)
        experiment_jaccard(result_output_folder_path, data_describ)
        CONFIG.LOG_FILE.flush()
    CONFIG.LOG_FILE.close()


def main_test_lda_window(by_incident):
    canidates = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    CONFIG.TURN_ON_SERIES = False
    CONFIG.TURN_ON_SEMANTIC = True
    CONFIG.SWITH_EMBED_TO_OTHER_APPROACH = CONFIG.LDA
    at2v_path = './output/bank/lda'
    result_output_folder_path = './output/bank/lda_window_result_1-5-60'
    if not os.path.exists(at2v_path):
        os.mkdir(at2v_path)
    if not os.path.exists(result_output_folder_path):
        os.mkdir(result_output_folder_path)
    data = open(os.path.join(result_output_folder_path, 'result_window_' + str(datetime.now().microsecond) + '.csv'),
                'w',
                encoding='utf-8')
    CONFIG.LOG_FILE = data
    for window in canidates:
        CONFIG.INCIDENT_WINDOW = window
        for num_topic in range(27, 53):
            data_describ = 'bank-topic-' + str(num_topic) + '-window-' + str(window) + '-byincident-' + str(by_incident)
            stime = datetime.now()
            main_generate_lda(num_topic, at2v_path, data_describ, by_incident)
            etime = datetime.now()
            duration = (etime - stime).total_seconds()
            print('training time', duration)
            exit(1)
            for threshold in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
                CONFIG.LDA_SIMILARITY_THRESHOLD = threshold
                new_data_describ = data_describ + '_thrd_' + str(threshold)
                print('training time\t', duration, end='\t', file=CONFIG.LOG_FILE)
                print('threshold\t', CONFIG.LDA_SIMILARITY_THRESHOLD, end='\t', file=CONFIG.LOG_FILE)
                print('window\t', str(CONFIG.INCIDENT_WINDOW), end='\t', file=CONFIG.LOG_FILE)
                print('num topic\t', num_topic, end='\t', file=CONFIG.LOG_FILE)
                print('by incident\t', by_incident, end='\t', file=CONFIG.LOG_FILE)
                alert_summarizing(at2v_path, None, None, result_output_folder_path, new_data_describ)
                CONFIG.LOG_FILE.flush()
    CONFIG.LOG_FILE.close()


def main_generate_lda(num_topic, folder_path, descp, by_incident=True):
    fmt_file_path = './input/pattern_label.csv'
    reader = csv.DictReader(open(fmt_file_path, encoding='utf8'))
    patterns = dict()
    alert_types = set()
    alert_contents = []
    for line_id, line in enumerate(reader):
        content = line['Template']
        key_words = parse_templates([content])[0]
        alert_contents.append(key_words)
        alert_id = int(line['ID'])
        alert_types.add(alert_id)
        pattern_id = int(line['Pattern'])
        if pattern_id not in patterns:
            patterns[pattern_id] = []
        for word in key_words:
            patterns[pattern_id].append(word)
    related_alert_contents = []
    for id in patterns:
        related_alert_contents.append(patterns[id])
    if by_incident:
        lda, dictionary = tp_topic.get_topics_by_templates(related_alert_contents, num_topics=num_topic)
    else:
        lda, dictionary = tp_topic.get_topics_by_templates(alert_contents, num_topics=num_topic)
    lda_path = os.path.join(folder_path, descp + '.lda')
    dictionary_path = os.path.join(folder_path, descp + '.dict')
    lda.save(lda_path)
    dictionary.save(dictionary_path)
    return lda, dictionary


def main_generate_idf(pattern_file_path):
    reader = csv.DictReader(open(pattern_file_path, encoding='utf8'))
    patterns = dict()
    for line_id, line in enumerate(reader):
        content = line['Template']
        pattern_id = int(line['Pattern'])
        key_words = parse_templates([content])[0]
        if pattern_id not in patterns:
            patterns[pattern_id] = []
        for word in key_words:
            patterns[pattern_id].append(word)
    idf = dict()
    for pattern_id in patterns:
        for word in set(patterns[pattern_id]):
            if word not in idf:
                idf[word] = 0
            idf[word] += 1
    for word in idf:
        idf[word] = len(patterns) * 1.0 / (idf[word] + 1)
    return idf


if __name__ == '__main__':
    #main_generate_at2v(num_topic=9)
    #main_test_at2v(5)
    #main_generate_as2v()
    #main_test_as2v(5)
    #for thread in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]:#0.4
    #for thread in [0.95]:
        # print("SEMANTIC_SIMILARITY_THRESHOLD:", thread)
        # CONFIG.SEMANTIC_SIMILARITY_THRESHOLD = thread
        # main_test_act(window=5, train=False)
    for epoch in range(5):
        main_test_act(window=5, train=False)

    # CONFIG.SEMANTIC_SIMILARITY_THRESHOLD = 0.999
    # print(CONFIG.SEMANTIC_SIMILARITY_THRESHOLD)
    # main_test_act(window=5, train=False)
    #for i in range(5):
    #main_test_act(window=5, train=True)

    # main_test_lda_window(by_incident=False)
    #main_test_jaccard(5)
