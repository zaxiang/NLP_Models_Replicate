#run tfidf and word2vector file
from TFIDF import *
from Word2Vector import *


class TFIDF_runner:

	def __init__(self, dataset):

		self.dataset = dataset

		object_token = Tokenization('./data/{}/coarse'.format(dataset))

		token = object_token.token_X()
		seeds = tfidf_token.modify_seeds()

		self.object_tfidf = tfidf('./data/{}/coarse'.format(dataset), token, seeds)

		
		def get_accuracy(self):

			return "micro and macro f1 scores for " + self.dataset + "are" + str(self.object_tfidf.get_accuracy())

class W2V_Runner:

	def __init__(self, dataset):

		self.dataset = dataset

		object_token = Tokenization('./data/{}/coarse'.format(dataset), 100, 5, 1, 4)

		token, word_dic = object_token.get_token_wordDic()
		seeds = object_token.modify_seeds()

		self.object_w2v = Word2vector('./data/{}/coarse'.format(dataset), token, word_dic, seeds)

		
	def get_accuracy(self):

		return "micro and macro f1 scores for " + self.dataset + "are" + str(self.object_w2v.get_accuracy())


tfidf_runner_nyt = TFIDF_runner('nyt')
tfidf_runner_nyt.get_accuracy()

tfidf_runner_20news = TFIDF_runner('20news')
tfidf_runner_20news.get_accuracy()
