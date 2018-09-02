CS5012-P1
POS tagging and smoothing
170024030
10/03/2018
--------------------------------------------------------------------------------------------------
GENERAL USAGE NOTES
--------------------------------------------------------------------------------------------------
- To train the HMM POS tagger, use
	python hmm_trainer.py <corpus_name> <tagset_name> <smoothing_method> <training_ratio>

- To test the HMM POS tagger, use
	python hmm_tester.py <corpus_name> <tagset_name> <smoothing_method> <training_ratio> <testing_ratio>

    valid options for   <corpus_name>:      "bown", "conll2000", "treebank"
                        <tagset_name>:      "default", "universal"
                        <smoothing_method>: "laplace", "good_turing"
                        <training_ratio> and <testing_ratio> must be integers in range(0,100)
- The code is programmed for python3 but compatible for python2.

- Required packages:  nltk (with corpura downloaded), json, sys, time, numpy, copy, collections.
--------------------------------------------------------------------------------------------------
