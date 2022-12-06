#!/usr/bin/env python

import sys
import json


#run tfidf and word2vector file
from TFIDF import *
from Word2Vector import *


class TFIDF_runner:

	def __init__(self, dataset, cla):

		print("Task 1/2: creating tf-idf model:")
		self.dataset = dataset

		object_token = tfidf_Tokenization('./test/testdata/{}/{}'.format(dataset, cla))

		token = object_token.token_X()
		seeds = object_token.modify_seeds()

		self.object_tfidf = tfidf('./test/testdata/{}/{}'.format(dataset, cla), token, seeds)

		print ("micro and macro f1 scores on test data for " + self.dataset + " are " + str(self.object_tfidf.get_accuracy())) 
		print ("finished task 1/2")

class W2V_Runner:

	def __init__(self, dataset, cla):

		print("Task 2/2: creating word2vec model: ")
		self.dataset = dataset

		object_token = w2v_Tokenization('./test/testdata/{}/{}'.format(dataset, cla), 100, 5, 1, 4)

		token, word_dic = object_token.get_token_wordDic()
		seeds = object_token.modify_seeds()

		self.object_w2v = Word2vector('./test/testdata/{}/{}'.format(dataset, cla), token, word_dic, seeds)

		print ( "micro and macro f1 scores on test data for " + self.dataset + " are " + str(self.object_w2v.get_accuracy()))
		print ("finished task2/2")



def main(targets):
	TFIDF_runner('20news', 'coarse')
	W2V_Runner('20news', 'coarse')

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)