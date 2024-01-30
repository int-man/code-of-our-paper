import sys
import math
import numpy as np
from config import CONFIG
from datetime import timedelta
from util.util import read_pattern_file
from incident.linked_list import LinkedList
sys.path.append(".")


def form_time_series_alarm(start_time, end_time, granularity, timestamp_seq):
    series = [0 for i in range(math.floor((end_time - start_time).total_seconds() / (granularity * 60.0)))]
    cnt_set = set()
    for ts in timestamp_seq:
        if ts > end_time:
            continue
        pos = math.floor((ts - start_time).total_seconds() / (granularity * 60.0)) - 1
        series[pos] += 1
        cnt_set.add(pos)
    return series, cnt_set


def get_train_data(alarm_sequence, pattern_file_path):
    series_length = CONFIG.SERIES_WINDOW_LENGTH
    series_granularity = CONFIG.SERIES_WINDOW_GRANULARITY
    window_width = CONFIG.INCIDENT_WINDOW
    train_data_input = []
    train_data_output_pos = []
    train_data_output_neg = []
    right_patterns, wrong_patterns = read_pattern_file(pattern_file_path)
    right_pattern_dict = dict()
    wrong_pattern_dict = dict()
    alm_template_dict = dict()

    for pattern in right_patterns:
        for alarm_id in pattern:
            if alarm_id not in right_pattern_dict:
                right_pattern_dict[alarm_id] = set()
            right_pattern_dict[alarm_id].update(pattern)
    for pattern in wrong_patterns:
        if len(pattern) > 2:
            continue
        for alarm_id in pattern:
            if alarm_id not in wrong_pattern_dict:
                wrong_pattern_dict[alarm_id] = set()
            wrong_pattern_dict[alarm_id].update(pattern)
    for alm_id in right_pattern_dict:
        right_pattern_dict[alm_id].remove(alm_id)
    for alm_id in wrong_pattern_dict:
        wrong_pattern_dict[alm_id].remove(alm_id)
        if alm_id in right_pattern_dict:
            wrong_pattern_dict[alm_id] = wrong_pattern_dict[alm_id].difference(right_pattern_dict[alm_id])

    alarm_ts = dict()
    alarm_list_position = 0
    window_alarms = set()
    for i, t_alarm in enumerate(alarm_sequence):
        pos_alm_id_series = dict()
        neg_alm_id_series = dict()
        alm_template_dict[t_alarm.id] = t_alarm.template
        incident_window_left = t_alarm.start_time - timedelta(seconds=window_width * 60)
        incident_window_right = t_alarm.start_time
        series_window_left = t_alarm.start_time - timedelta(seconds=series_length * 60)
        series_window_right = t_alarm.start_time

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

        if t_alarm.id not in alarm_ts:
            alarm_ts[t_alarm.id] = LinkedList()
        alarm_ts[t_alarm.id].append(t_alarm.start_time)

        while len(alarm_ts[t_alarm.id]) > 0 and \
                alarm_ts[t_alarm.id].front() <= series_window_left:
            alarm_ts[t_alarm.id].pop()
        if len(alarm_ts[t_alarm.id]) == 0:
            alarm_ts.pop(t_alarm.id)

        t_alarm_series, cnt_set = form_time_series_alarm(series_window_left, series_window_right, series_granularity,
                                                         alarm_ts[t_alarm.id])
        if len(cnt_set) <= 1:
            continue
        t_alarm_series = (t_alarm_series - np.mean(t_alarm_series)) / np.std(t_alarm_series)
        pos_vec = np.zeros(len(t_alarm_series))
        if t_alarm.id in right_pattern_dict:
            related_alarms = {e.id for e in window_alarms} & right_pattern_dict[t_alarm.id]
            if len(related_alarms) == 0:
                continue
            if related_alarms:
                for o_alarm_id in related_alarms:
                    o_alarm_series, cnt_set = form_time_series_alarm(series_window_left, series_window_right,
                                                                     series_granularity,
                                                                     alarm_ts[o_alarm_id])
                    if len(cnt_set) <= 1:
                        continue
                    pos_vec += o_alarm_series
                    pos_alm_id_series[str(o_alarm_id)] = o_alarm_series
                if len(pos_alm_id_series) > 0:
                    pos_vec = (pos_vec - np.mean(pos_vec)) / np.std(pos_vec)
        if len(pos_alm_id_series) == 0:
            continue
        neg_vec = np.zeros(len(t_alarm_series))
        if t_alarm.id in wrong_pattern_dict:
            unrelated_alarms = {e.id for e in window_alarms} & wrong_pattern_dict[t_alarm.id]
            if unrelated_alarms:
                for o_alarm_id in unrelated_alarms:
                    o_alarm_series, cnt_set = form_time_series_alarm(series_window_left, series_window_right,
                                                                     series_granularity,
                                                                     alarm_ts[o_alarm_id])
                    if len(cnt_set) <= 1:
                        continue
                    neg_vec += o_alarm_series
                    neg_alm_id_series[str(o_alarm_id)] = o_alarm_series
                if len(neg_alm_id_series) > 0:
                    neg_vec = (neg_vec - np.mean(neg_vec)) / np.std(neg_vec)
        if len(neg_alm_id_series) == 0:
            neg_vec = None
        train_data_input.append(t_alarm_series)
        train_data_output_pos.append(pos_vec)
        train_data_output_neg.append(neg_vec)
    return train_data_input, train_data_output_pos, train_data_output_neg
