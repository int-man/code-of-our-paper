from gensim.models import word2vec
from incident.util import parse_templates


def get_word2vec_model(alarms, size, window):
    raw_templates = []
    for alarm in alarms:
        raw_templates.append(alarm.template)
    templates = parse_templates(raw_templates)
    sentence = []
    for words in templates:
        sentence.extend(words)
    #model = word2vec.Word2Vec([sentence], size=size, sg=1, min_count=1, window=window, iter=100)  # 30
    model = word2vec.Word2Vec([sentence], vector_size=size, sg=1, min_count=1, window=window)
    return model
