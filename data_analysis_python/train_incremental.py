from pandas import *
from sklearn.linear_model import SGDRegressor

class IncrementalClassifier:
	def __init__(features, cluster_dict):
		self.features = features
		self.unseen_features = self.features.copy()
		self.cluster_dict = cluster_dict
		self.model = None

	def extract_previous(self, user):
		self.asin_prev, self.like_prev = \
				open('/root/classifier/{0}.txt'\
			    		.format(user))\
					.read()\
					.split('\t')

	def update_unseen(self):
		if self.asin_prev not in self.unseen_features.index:
			return self.unseen_features

		self.unseen_features.drop(self.asin_prev, 
								  axis = 0,
								  inplace = True)

	def get_candidates_cluster(self, cluster_num=None):
		return self.unseen_features.loc[
					self.cluster_dict[cluster_num]
					].dropna()

	def new_model(self, class_weight=None):
		return SGDClassifier(loss='hinge', penalty='elasticnet', 
	         alpha=0.1, l1_ratio=0.85, fit_intercept=True, n_iter=5, 
	         shuffle=True, verbose=0, epsilon=0.1, 
	         n_jobs=1, random_state=None, learning_rate='optimal', 
	         eta0=0.0, power_t=0.5, class_weight=class_weight, 
	         warm_start=False, average=False)

	def score_unseen(self, candidates):
		scores = self.model.decision_function(self.unseen_features.values)
		return (scores).argsort()[:n]

	def batch_train(self, likes, dislikes, n=1):
		self.model = self.new_model({0: len(likes),
									 1: len(dislikes)})

		all_reviewed = [l for l in likes]
		all_reviewed.extend(dislikes)

		dataset = self.features.loc[all_reviewed, :]
		self.model.fit(dataset.drop('like', axis = 1), 
						dataset.like.values)

		return self.score_unseen(self.get_candidates_cluster())

	def online_train(self):
		self.model.partial_fit(self.asin_prev.values.reshape(1, -1), 
                  np.array(int(self.like_prev)).reshape(-1, 1))
		return self.score_unseen()

## in applications.py, send to similarity until like and dislike are both > 0
## also call 
ic = IncrementalClassifier(features, cluster_dict)

CLUSTER_NUM = 1
ic.extract_previous('Josh')
ic.update_unseen()
candidates = ic.get_candidates_cluster()

if len(like) == 0 or len(dislike) == 0:
	## similarity
else:
	if len(like) + len(dislike) % 10 == 0 \
		## batch train
		or ic.model is None:
		ic.batch_train(likes, dislikes)
	else:
		## online train
		ic.online_train()


## balance?/class weight?

## TODO : recommendations
