from __future__ import division
from math import floor
from math import ceil
from pandas import *
from sklearn.cross_validation import StratifiedShuffleSplit

class PrepData:
	'''Takes df of asin:likes'''
	def __init__(self, data, features, training_sizes, folds, balance=True):
		self.data = data
		self.features = features
		self.training_sizes = training_sizes
		self.folds = folds
		self.form_dataset()
		self.training_sets, self.test_sets = self.test_train_split()
		if balance == True:
			self.balance_all_training_sets()
		self.unlabeled = self.form_unlabeled_dataset()

	def balance_training_set(self, n):
		dfs = self.training_sets[n]
		for fold in dfs.keys():
			df = dfs[fold]

			dislike_examples = df[df.like == 0]
			like_examples = df[df.like == 1]

			tmp = df.like.value_counts()
			dislikes = tmp[0]
			if len(tmp) == 1:
				print 'Only 1 class represented for training size n={0}. Ignoring.'.format(n)
				return {}
			likes = tmp[1]

			if dislikes > likes:
				# how many times ot append each positive example
				mult = int(ceil(dislikes/likes))
				for _ in range(mult):
					# append 1 of each positive example to data
					df = df.append(like_examples)
			elif dislikes < likes:
				# how many times ot append each negative example
				mult = int(ceil(likes/dislikes))
				for _ in range(mult):
					# append 1 of each negative example to data
					df = df.append(dislike_examples)
			# print df.value_counts
			dfs[fold] = df
		return dfs

	# oversample the smaller class to balance dataset
	# not exactly 50/50 since it's oversampled in batches
	# (all examples from minority class are oversampled
	# the same n times)
	def balance_all_training_sets(self):
		for size in self.training_sizes:
			self.training_sets[size] = self.balance_training_set(size)

	def form_dataset(self):
		self.data =  self.data.merge(self.features, 
						left_index = True,
						right_index = True)

	def test_train_split(self):	
		training_sets = {}
		test_sets = {}
		for n in self.training_sizes:
			print n, 1-n/len(self.data)
			sss = StratifiedShuffleSplit(
						self.data.like.values, 
						self.folds, 
						test_size=1-n/len(self.data), 
						random_state=45)
			training_sets[n] = {}
			test_sets[n] = {}

			fold = 0
			for train_index, test_index in sss:
				print len(train_index), len(test_index)
				training_sets[n][fold] = self.data.copy().iloc[train_index, :]
				test_sets[n][fold] = self.data.copy().iloc[test_index, :]     

				if fold == 0:
					test_sets[n]['whole'] = test_sets[n][fold].drop_duplicates()
					training_sets[n]['whole'] = training_sets[n][fold].drop_duplicates()
				else:
					test_sets[n]['x`whole'] = test_sets[n]['whole']\
												.append(test_sets[n][fold]).drop_duplicates()
					training_sets[n]['whole'] = training_sets[n]['whole']\
												.append(training_sets[n][fold]).drop_duplicates()
				fold += 1

		return training_sets, test_sets

	def form_unlabeled_dataset(self):
		return self.features[
					self.features\
						.index\
						.isin(self.data.index) == False]

