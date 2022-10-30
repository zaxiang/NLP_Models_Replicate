#TFIDF

import json
import pandas as pd
import numpy
import pickle
import string
from nltk.stem.porter import *

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from numpy.linalg import norm

class Tokenization:

	def __init__(self, dataset_path):

		f = open(dataset_path + "/seedwords.json")

		self.seeds_dic = json.load(f)

		with open(dataset_path + "/df.pkl", 'rb') as f:
			self.df = pickle.load(f)

		self.X_train = self.df['sentence'].tolist()


	def token_X(self):

		print('tokenizing...')

		tfidf_vectorizer = TfidfVectorizer()
		tokenizer = tfidf_vectorizer.build_tokenizer()
	    
		punct = string.punctuation
		stemmer = PorterStemmer()

		X_token = []
		for doc in self.X_train:
			doc = doc.lower()
			doc = [i for i in doc if not (i in punct)] # non-punct characters
			doc = ''.join(doc) # convert back to string
			words = tokenizer(doc) # tokenizes
			for i in range(len(words)): #stemmer
				words[i] = stemmer.stem(words[i])
			X_token.append(words)
	    
		return X_token

	def modify_seeds(self):

		for clas in self.seeds_dic:
			cla_lis = self.seeds_dic[clas]

			token_input = [' '.join(cla_lis)]

			#repeat the token_X function
			tfidf_vectorizer = TfidfVectorizer()
			tokenizer = tfidf_vectorizer.build_tokenizer()
		    
			punct = string.punctuation
			stemmer = PorterStemmer()

			X_token = []
			for doc in token_input:
				doc = doc.lower()
				doc = [i for i in doc if not (i in punct)] # non-punct characters
				doc = ''.join(doc) # convert back to string
				words = tokenizer(doc) # tokenizes
				for i in range(len(words)): #stemmer
					words[i] = stemmer.stem(words[i])
				X_token.append(words)

			new_lis = X_token[0]
			self.seeds_dic[clas] = new_lis

		return self.seeds_dic


class tfidf:

	def __init__(self, dataset_path, token, seeds_dic):

		with open(dataset_path + "/df.pkl", 'rb') as f:
			self.df = pickle.load(f)

		self.X_train = self.df['sentence'].tolist()

		self.token = token
		self.seeds_dic = seeds_dic


	def get_idf(self):

		print('get_idf')

		dic_idf = defaultdict(int)
		for doc in self.token:
			unique_token = set(doc)
			for w in unique_token:
				dic_idf[w] += 1

		self.idf_dic = dic_idf

	def get_tfidf_stat(self, doc, seeds): #the passed in doc is already tokenized

		sum_tfidf = 0
    
		dic_tfidf = defaultdict(int)
    
		for w in doc:
			dic_tfidf[w] += 1
        
		counter = 0
		for s in seeds:
			if s in self.idf_dic:
				#print("yeahhh")
				counter += 1
				dic_tfidf[s] = dic_tfidf[s] * numpy.log((len(self.X_train) / self.idf_dic[s]))
			else:
				#print("no way..")
				dic_tfidf[s] = 0
			sum_tfidf += dic_tfidf[s] 
        
		return sum_tfidf / counter

	def get_class(self, doc):

		dic_scores = defaultdict(int)

		for c in list(self.seeds_dic.keys()):
			tfidf = self.get_tfidf_stat(doc, self.seeds_dic[c])
			dic_scores[c] = tfidf

		if len(set(list(dic_scores.values()))) == 1: 
			if int(list(set(list(dic_scores.values())))[0]) == 0: #having tie: all 0 case
				return "no_label"
			else: #having tie: 
				return list(dic_scores.keys())[0]

		return max(dic_scores, key=dic_scores.get)


	def get_prediction(self):

		prediction = []
		print("getting predictions")
		for doc in self.token:
			prediction.append(self.get_class(doc))

		return prediction

	def get_accuracy(self):

		self.get_idf() #get idf attribute only once

		prediction = self.get_prediction()
    
		print('calculating accuracy')
		#print('Accuracy: ', accuracy_score(news_df.label.tolist(), news_prediction, normalize=False))
		micro = f1_score(self.df.label, prediction, average='micro')
		macro = f1_score(self.df.label, prediction, average='macro')
		# print('F1-score micro: ', micro)
		# print('F1-score macro: ', macro)

		return micro, macro
    
