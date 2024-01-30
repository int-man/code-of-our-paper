import threading


class Config(object):
    _instance_lock = threading.Lock()

    INCIDENT_WINDOW = 5

    TURN_ON_SEMANTIC = True
    TURN_ON_SERIES = True
    SWITH_EMBED_TO_OTHER_APPROACH = 0
    ASR = 0
    JACCARD = 1
    LDA = 2
    W2VEC = 3
    LLM = -1

    BERT = 0
    RoBerTa = 1
    GPT2 = 2
    BART = 3

    SERIES_WINDOW_LENGTH = 60 * 6
    SERIES_WINDOW_GRANULARITY = 1
    SERIES_SIMILARITY_THRESHOLD = 0.92
    SERIES_DRAW = True

    SEMANTIC_SIMILARITY_THRESHOLD = 0.003
    JACCARD_SIMILARITY_THRESHOLD = 0.99
    W2VEC_SIMILARITY_THRESHOLD = 0.003
    LDA_SIMILARITY_THRESHOLD = 0.003

    LOG_FILE = None

    MONTH_STRING = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                    'Nov': 11, 'Dec': 12}

    def __new__(cls, *args, **kwargs):
        if not hasattr(Config, "_instance"):
            with Config._instance_lock:
                if not hasattr(Config, "_instance"):
                    Config._instance = object.__new__(cls)
        return Config._instance


CONFIG = Config()
