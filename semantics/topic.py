from gensim import corpora, models
from incident.util import parse_templates


def get_topics(alarms, num_topics):
    raw_templates = []
    for alarm in alarms:
        raw_templates.append(alarm.template)
    templates = parse_templates(raw_templates)
    dictionary = corpora.Dictionary(templates)
    corpus = [dictionary.doc2bow(words) for words in templates]
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, iterations=6000)
    print('log_perplexity', lda.log_perplexity(corpus))
    return lda, dictionary


def get_topics_by_templates(templates, num_topics):
    dictionary = corpora.Dictionary(templates)
    corpus = [dictionary.doc2bow(words) for words in templates]
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, iterations=6000,
                                   minimum_probability=0)
    print('log_perplexity', lda.log_perplexity(corpus))
    return lda, dictionary


def get_topic_by_words(key_words, lda, dictionary):
    bow = dictionary.doc2bow(key_words)
    result = lda.inference([bow])[0][0]
    return result.argsort()[-1]
