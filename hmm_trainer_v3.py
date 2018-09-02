#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:08:36 2018

@author: 170024030
"""
from hmm_utils import *
from nltk.corpus import *
from collections import defaultdict
import json
import copy
import sys
import time



# Initialize the count for all tags with 0
def initialize_count(entry, possible_tags):
    for tag in possible_tags:
        entry[tag] = 0

# Update the emission count matrix
def update_emit_model(word, tag, table, possible_tags):
    if word not in table or tag not in table[word]:
        table[word][tag] = 1
    else:

        table[word][tag] += 1

# Update the transition count matrix
def update_trans_model(cur_tag, prev_tag, table, possible_tags):
    if cur_tag not in table:
        row = dict()
        initialize_count(row, possible_tags)
        table[cur_tag] = row
    table[cur_tag][prev_tag] += 1

# Initialize the transition matrix with all possible tags from the corpus
def initialize_trans_model(table, possible_tags):
    for cur_tag in possible_tags:
        for prev_tag in possible_tags:
            table[cur_tag][prev_tag] = 0

# Apply Laplace smoothing to calculate the probability matrix
def laplace_smoothing(table, counter):
    laplace_prob = copy.deepcopy(table)
    laplace_counter = copy.deepcopy(counter)
    # Add one to both condition counter and total count
    for cur in laplace_prob:
        for pre in laplace_prob[cur]:
            laplace_prob[cur][pre] += 1
            laplace_counter[pre] += 1
    # Calculate the probabilities
    for cur in laplace_prob:
        for pre in laplace_prob[cur]:
            laplace_prob[cur][pre] = 1.0 * laplace_prob[cur][pre] / laplace_counter[pre]
    return laplace_prob

# Apply Good Turing smoothing to calculate the probability matrix
def good_turing_smoothing(table, counter, possible_tags):
    # Count the frequencies
    Nc = dict()
    for prev_tag in possible_tags:
        Nc[prev_tag] = dict()
        for cur_tag in table:
            if table[cur_tag][prev_tag] in Nc[prev_tag]:
                Nc[prev_tag][table[cur_tag][prev_tag]] += 1
            else:
                Nc[prev_tag][table[cur_tag][prev_tag]] = 1
    # Calculate the probabilities
    good_turing_prob = defaultdict(dict)
    for prev_tag in possible_tags:
        x = []
        y = []
        for count in Nc[prev_tag]:
            if Nc[prev_tag][count] != 0:
                x.append(count)
                y.append(Nc[prev_tag][count])
        k, b = linear_regression(x, y)
        for cur_tag in possible_tags:
            count = table[cur_tag][prev_tag]
            good_turing_prob[cur_tag][prev_tag] = (table[cur_tag][prev_tag] + 1) * (k * (count + 1) + b) / (k * (count) + b) / counter[prev_tag]
            if good_turing_prob[cur_tag][prev_tag] < 0:
                good_turing_prob[cur_tag][prev_tag] = 1 / counter[prev_tag]
    return good_turing_prob

# The main method that takes three argument: <corpus_name> <tagset_name> <smoothing_method>
def main():
    time_start = time.time()
    print('=====================')
    print('Loading data...')
    print('=====================')
    # Usability error handling
    try:
        corpus_name = sys.argv[1]
        tagset_name = sys.argv[2]
        smoothing_method = sys.argv[3]
        training_perc = sys.argv[4]
        training_perc = int(training_perc)
        if training_perc >= 100 or training_perc <= 0 or training_perc != int(training_perc):
            print('Percentage must a integer in range (0,100)')
            return
        training_ratio = (1.0 * training_perc) / 100
    except IndexError:
        print('Usability:   python hmm_trainer.py <corpus_name> <tagset_name> <smoothing_method> <traning_ratio')
        print('             if the default tagset is desired, input defalt at <tagset_name>')
        return
    except ValueError:
        print('Ratio must be a valid number in range (0,100)')
        return
    try:
        sents, possible_tags = choose_corpus(corpus_name, tagset_name)
    except KeyError:
        print('Supported corpora: brown, conll2000, treebank')
        return

    # Initialize the matrices and parameters
    len_sents = len(sents)
    emit_count, trans_count, tag_counter = defaultdict(dict), defaultdict(dict), dict()
    initialize_count(tag_counter, possible_tags)
    initialize_trans_model(trans_count, possible_tags)
    print('=====================')
    print('Triaining...')
    print("Corpus: " + corpus_name)
    print("Tagset: " + tagset_name)
    print("Smoothing method: " + smoothing_method)
    print("Training ratio = " + str(training_perc) + "%" )
    print('=====================')

    # Count emission and transition word by word
    for sent in sents[:int(len_sents * training_ratio)]:
        pre = '<S>'
        for token in sent:
            tag_counter['<S>'] += 1
            tag_counter['</S>'] += 1
            word, tag = token[0], token[1]
            # Counting c(tag,word) for each of the word
            update_emit_model(word, tag, emit_count, possible_tags)

            # Counting c(TAGcur preceeded by TAGpre)
            cur = tag
            update_trans_model(cur, pre, trans_count, possible_tags)
            pre = cur
            # Update the tag counter c(tag), which will be used to calculate Pt and Pe
            tag_counter[tag] += 1
        cur = '</S>'

        # Involve the end-of-sentence marker to transition probabilities.
        update_trans_model(cur, pre, trans_count, possible_tags)

    # Calculate Pe(word | tag) = c(word , tag) / c(tag)
    emit_prob = copy.deepcopy(emit_count)
    for word in emit_prob:
        for tag in emit_prob[word]:
            emit_prob[word][tag] = 1.0 * emit_prob[word][tag] / tag_counter[tag]

    # A switcher to store all invokers for smoothing methods
    smoothing_methods = {
        'laplace': laplace_smoothing(trans_count, tag_counter),
        'good_turing': good_turing_smoothing(trans_count, tag_counter, possible_tags),
    }

    # Error handler
    try:
        trans_prob = smoothing_methods[smoothing_method]
    except KeyError:
        print('<smoothing_method> only support \"laplace\" and \"good_turing\"')
        quit()

    # Save the output transition and emission matrices
    save_dict("models/Transition_" + corpus_name + '_' + tagset_name + '_' + smoothing_method + '_' + str(training_perc),  trans_prob)
    save_dict("models/Emission_" + corpus_name + '_' + tagset_name + '_' + str(training_perc), emit_prob)
    time_end = time.time()
    print('=====================')
    print('Training Successful.')
    print('Time: ' + str(time_end - time_start) + 's')
    print('=====================')



if __name__ == "__main__": main()
