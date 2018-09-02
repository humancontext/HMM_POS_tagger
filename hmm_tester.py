#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:54:06 2018

@author: 170024030
"""

from hmm_utils import *
from collections import defaultdict
import csv
import json
import sys
import time
import pandas as pd

# Calculate the probability using viterbi method: Pi(tag) = Max(Pt(tagi|tagi-1) * Pe(wordi | tagi) * Pi-1(tagi-1))
def update_viterbi_matrix(matrix, index, cur_word, sent, j, emit_model, trans_model, possible_tags):
    coe = 1000
    entry = dict()
    prob_set = []
    DEF_THRE = -1
    # Handle the start-of-sentence marker
    if index == 0:
        if cur_word in emit_model:
            for tag in emit_model[cur_word]:
                p = emit_model[cur_word][tag] * trans_model[tag]['<S>'] * coe
                prob_set = [p, '<S>']
                entry[tag] = prob_set
        else:
            for tag in possible_tags:
                p =  trans_model[tag]['<S>'] * coe
                prob_set = [p, '<S>']
                entry[tag] = prob_set
    # Handle the end-of-sentence marker
    elif index == -1:
        bestp = DEF_THRE
        for prev_tag in matrix[len(matrix) - 1]:
            p = trans_model['</S>'][prev_tag] * matrix[len(matrix) - 1][prev_tag][0] * coe
            if p > bestp:
                bestp = p
                prob_set = [p, prev_tag]
            entry['</S>'] = prob_set
    # If the word is seen in trained corpus, use the good old Pi(tag) = Max(Pt(tagi|tagi-1) * Pe(wordi | tagi) * Pi-1(tagi-1))
    elif cur_word in emit_model:
        for tag in emit_model[cur_word]:
            bestp = DEF_THRE
            for prev_tag in matrix[index - 1]:
                p = emit_model[cur_word][tag] * trans_model[tag][prev_tag] * matrix[index - 1][prev_tag][0] * coe
                if p > bestp:
                    bestp = p
                    prob_set = [p, prev_tag]
            entry[tag] = prob_set
    # For words already seen, only consider Pi(tag) = Max(Pt(tagi|tagi-1) * Pi-1(tagi-1))
    else:
        for tag in possible_tags:
            bestp = DEF_THRE
            for prev_tag in matrix[index - 1]:
                p = trans_model[tag][prev_tag] * matrix[index - 1][prev_tag][0] * coe
                if p > bestp:
                    bestp = p
                    prob_set = [p, prev_tag]
            entry[tag] = prob_set
    return entry

# From the viterbi matrix generate the best prediction backwards
def calc_mostlikely_tags(matrix):
    tags = []
    prev_tag = '</S>'
    tags.append(prev_tag)
    for i in range(len(matrix))[::-1]:
        prev_tag = matrix[i][prev_tag][1]
        tags.append(prev_tag)
    return tags[::-1]

# Initialize the confusion matrix
def init_conf_matrix(table, possible_tags):
    for cur_tag in possible_tags:
        for prev_tag in possible_tags:
            table[cur_tag][prev_tag] = 0


def main(): 
    print('=====================')
    print('Loading data...')
    print('=====================')
    time_start = time.time()
    # Error handlers
    try:
        corpus_name = sys.argv[1]
        tagset_name = sys.argv[2]
        smoothing_method = sys.argv[3]
        training_perc = sys.argv[4]
        testing_perc = sys.argv[5]
        sents, possible_tags = choose_corpus(corpus_name, tagset_name)
        trans_model = load_model("Transition_" + corpus_name + '_' + tagset_name + '_' + smoothing_method + '_' + training_perc)
        emit_model = load_model("Emission_" + corpus_name + '_' + tagset_name + '_' + training_perc)
        testing_perc = int(testing_perc)
        if testing_perc >= 100 or testing_perc <= 0 or testing_perc != int(testing_perc):
            print('Percentage must a integer in range (0,100)')
            return
        testing_ratio = (1.0 * testing_perc) / 100
    except IndexError:
        print('Usability:   python hmm_tester.py <corpus_name> <tagset_name> <smoothing_method> <training_perc> <testing_perc>')
        print('             if the default tagset is desired, input defalt at <tagset_name>')
        return
    except KeyError:
        print('Supported corpora: brown, conll2000, treebank')
        return
    except FileNotFoundError:
        print('Model not found. Please train first.')
        return
    except ValueError:
        print('Percentage must a integer in range (0,100)')
        return

    # Initialize prameters and matrices
    possible_tags.remove('<S>')
    len_sents = len(sents)
    total_word = 0
    correct_word = 0
    try:
        sents, possible_tags = choose_corpus(corpus_name, tagset_name)
    except KeyError:
        print('Supported corpora: brown, conll2000, masc_tagged, treebank')
        return
    conf_matrix = defaultdict(dict)
    init_conf_matrix(conf_matrix, possible_tags)

    print('=====================')
    print('Testing...')
    print('Triaining...')
    print("Corpus: " + corpus_name)
    print("Tagset: " + tagset_name)
    print("Smoothing method: " + smoothing_method)
    print("Training ratio = " + str(training_perc) + "%" )
    print("Testing ratio = " + str(testing_perc) + "%" )
    print('=====================')
    for sent in sents[int(len_sents * (1 - testing_ratio)):]:
        word_seq = [w for (w, _) in sent]
        tag_seq = [t for (_, t) in sent]
        decode_map = dict()
        for i in range(len(word_seq)):
            cur_word = word_seq[i]
            decode_map[i] = update_viterbi_matrix(decode_map, i, cur_word, sent, i, emit_model, trans_model, possible_tags)
        # Involve end-of-sentence marker
        decode_map[len(word_seq)] = update_viterbi_matrix(decode_map, -1, "", sent, i, emit_model, trans_model, possible_tags)
        predicted_tags = calc_mostlikely_tags(decode_map)
        word_tag_only = predicted_tags[1:len(predicted_tags) - 1]
        total_word += len(word_seq)
        for i in range(len(tag_seq)):
            conf_matrix[tag_seq[i]][word_tag_only[i]] += 1
            if tag_seq[i] == word_tag_only[i]:
                correct_word += 1
    time_end = time.time()
    print('=====================')
    print('Accuracy: ' + str(100.0 * correct_word / total_word) + '%')
    print('Time: ' + str(time_end - time_start) + 's')
    print('=====================')

    data = pd.DataFrame(conf_matrix)
    data.to_csv("conf_matrix_" + corpus_name + '_' + tagset_name + '_' + smoothing_method + '_' + str(training_perc)+".csv")


if __name__ == "__main__": main()
