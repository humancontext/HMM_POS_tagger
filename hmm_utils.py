from nltk.corpus import *
import numpy as np
import json

# A switcher to select corpus
def choose_corpus(corpus_name, tagset_name):
    if tagset_name == 'default':
        tagset_name =''
    # Error handler
    elif tagset_name != 'universal':
        print('<tagset_name> only support \"universal\" and \"default\"')
        quit()
    switcher = {
        'brown': brown.tagged_sents(tagset = tagset_name),
        'conll2000': conll2000.tagged_sents(tagset = tagset_name),
        'treebank': treebank.tagged_sents(tagset = tagset_name),
    }
    sents = switcher[corpus_name]
    possible_tags = []
    for sent in sents:
        for token in sent:
            if token[1] not in possible_tags:
                possible_tags.append(token[1])
    possible_tags.append('<S>')
    possible_tags.append('</S>')
    return switcher[corpus_name], possible_tags

# Save the trained model
def save_dict(name, table):
    with open(name, 'w') as file_output:
        json.dump({name: table}, file_output, indent = 6)


# Load trained model from the saved file.
def load_model(name):
    with open(name) as file_input:
        data = json.load(file_input)
        return data[name]

# Linear regression
def linear_regression(x, y):
    X = np.array(x)
    Y = np.array(y)
    A = np.vstack([X, np.ones(len(X))]).T
    slope, intercept = np.linalg.lstsq(A, Y)[0]
    return slope, intercept
