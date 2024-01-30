import jieba
import threading

from string import digits
from semantics.stop_words import ENG_STOP_WORDS


class Util(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        self.group_id = 0
        self.series_group_id = 0
        self.pattern_group_id = 0
        self.parameter_group_id = 0
        self.semantic_group_id = 0
        self.incident_id = 0

    def generate_group_id(self):
        old_group_id = self.group_id
        self.group_id += 1
        return old_group_id

    def generate_semantic_group_id(self):
        old_semantic_group_id = self.semantic_group_id
        self.semantic_group_id += 1
        return old_semantic_group_id

    def generate_series_group_id(self):
        old_series_group_id = self.series_group_id
        self.series_group_id += 1
        return old_series_group_id

    def generate_pattern_group_id(self):
        old_pattern_group_id = self.pattern_group_id
        self.pattern_group_id += 1
        return old_pattern_group_id

    def generate_parameter_group_id(self):
        old_parameter_group_id = self.parameter_group_id
        self.parameter_group_id += 1
        return old_parameter_group_id

    def __new__(cls, *args, **kwargs):
        if not hasattr(Util, "_instance"):
            with Util._instance_lock:
                if not hasattr(Util, "_instance"):
                    Util._instance = object.__new__(cls)
        return Util._instance


def parse_templates(raw_templates):
    stop_words = set()
    stop_words.update(ENG_STOP_WORDS)

    templates = []
    for i, template in enumerate(raw_templates):
        remove_digits = str.maketrans('', '', digits)
        template = template.translate(remove_digits)
        key_words = template.replace('-', ' ').replace('_', ' ').replace('+', ' ').replace('/', ' ') \
            .replace('.', ' ').replace('%', ' ').replace('*', ' ').replace('【', ' ').replace('】', ' ') \
            .replace(':', ' ').replace('$', ' ').replace('@', ' ').replace('[', ' ').replace(']', ' ') \
            .replace('!', ' ').replace(')', ' ').replace('(', ' ').replace('（', ' ').replace('）', ' ') \
            .replace('<', ' ').replace('-', ' ').replace('>', ' ').replace('=', ' ').replace('：', ' ') \
            .replace('-', ' ').split()
        tmp = []
        for word in key_words:
            for sub_word in jieba.cut(word, cut_all=False):
                tmp.append(sub_word.lower())
        key_words = tmp
        removed = []
        for word in key_words:
            if word.strip() == '' or word in stop_words or len(word) == 1:
                removed.append(word)
        for word in removed:
            key_words.remove(word)
        if len(key_words) > 0:
            templates.append(key_words)
        else:
            templates.append([" "])
    return templates


UTIL = Util()
