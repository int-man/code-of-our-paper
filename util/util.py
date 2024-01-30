from config import CONFIG
import csv
import xlrd


def read_csv_data(csv_file_path, key_columns, encoding='utf8'):
    reader = csv.DictReader(open(csv_file_path, encoding=encoding))
    raw_alarms = []
    for line_id, line in enumerate(reader):
        raw_alarm = dict()
        for key_column in key_columns:
            if key_column not in line:
                continue
            raw_alarm[key_column] = line[key_column]
        raw_alarms.append(raw_alarm)
    return raw_alarms


def read_pattern_file(label_file_path):
    workbook = xlrd.open_workbook(label_file_path)
    sheet = workbook.sheet_by_index(0)
    right_patterns = []
    wrong_patterns = []
    visited = set()
    for row, row_range, col, col_range in sheet.merged_cells:
        if (row, row_range) in visited:
            continue
        visited.add((row, row_range))
        label = str(sheet.cell_value(row, 4))
        pattern = set()
        while row < row_range:
            log_id = int(sheet.cell_value(row, 1))
            row += 1
            pattern.add(log_id)
        if label == 'Y':
            right_patterns.append(pattern)
        if label == 'N':
            wrong_patterns.append(pattern)
    return right_patterns, wrong_patterns


def statistics(alarm_sequence):
    incident_alm_cnt = dict()
    single_alarm_cnt = 0
    for alarm in alarm_sequence:
        if alarm.incident_id not in incident_alm_cnt:
            incident_alm_cnt[alarm.incident_id] = 0
        incident_alm_cnt[alarm.incident_id] += 1
    for alarm in alarm_sequence:
        if incident_alm_cnt[alarm.incident_id] == 1:
            alarm.is_single = True
            single_alarm_cnt += 1
    print("input alerts\t" + str(len(alarm_sequence)) + "\t#incidents\t" + str(len(incident_alm_cnt)) +
          "\t#isolated incidents\t" + str(single_alarm_cnt) + "\tnaive compression ratio\t" +
          str((1 - len(incident_alm_cnt) * 1.0 / len(alarm_sequence)) * 100), end="\t", file=CONFIG.LOG_FILE)
