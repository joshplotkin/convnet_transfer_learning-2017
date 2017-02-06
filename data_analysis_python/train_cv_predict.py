from math import floor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from operator import itemgetter
import os
from pandas import *
from scipy.stats import randint as sp_randint
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from time import time
import xgboost as xgb
from XGBoostClassifier import XGBoostClassifier

class TrainModel:
	def __init__(self, training, test, algo, params, folds):
		self.training = training
		# make sure no training examples in test set
		# (possible due to oversampling)
		self.test = test[test\
						   .index
						   .isin(self.training) == False]\
						 .drop_duplicates()
		self.algo = algo
		self.params = params
		self.folds = folds

		self.roc_auc = None
		self.recs = None

		self.train_model()
		self.score_test_set()

	def make_recommendations(self, new_images, n):
		if self.algo == 'xgb_no_cv':
			X = xgb.DMatrix(new_images)
		else:
			X = new_images

		new_images.loc[:, 'predict'] = np.array(self.model.predict(X))
		self.recs = new_images.sort_values(by = 'predict', 
								ascending = False, 
								inplace = False)\
							  .index[:n]\
							  .values

	def display_recommendations(self, new_images, n):
		if self.recs is None:
			self.make_recommendations(new_images, n)

		for img in ['../boots_dataset/pics_boots/{0}.jpg'.format(asin) 
					for asin in self.recs]:
			if os.path.isfile(img) and open(img).read() != '':
				plt.imshow(mpimg.imread(img))
			plt.grid(False)
			plt.xticks([], [])
			plt.yticks([], [])
			plt.show()

	def roc_metrics(self, other_predictions=None):
		if other_predictions is not None:
			pred = other_predictions
		else:
			pred = self.predictions

		fpr, tpr, _ = roc_curve(pred.actual, 
								pred.predicted, 
								pos_label = 1)
		roc_auc = auc(fpr, tpr) 
		return fpr, tpr, roc_auc

	def get_auc(self):
		if self.roc_auc is None: 
			self.roc_auc = self.roc_metrics()[2]
		return self.roc_auc

	def balance_oversample(self, df):

		dislike_examples = df[df.like == 0]
		like_examples = df[df.like == 1]

		tmp = df.like.value_counts()
		dislikes = tmp[0]
		likes = tmp[1]

		if dislikes > likes:
			# how many times ot append each positive example
			mult = int(floor(dislikes/likes))
			for _ in range(mult):
				# append 1 of each positive example to data
				df = df.append(like_examples)
		elif dislikes < likes:
			# how many times ot append each negative example
			mult = int(floor(likes/dislikes))
			for _ in range(mult):
				# append 1 of each negative example to data
				self.data = df.append(dislike_examples)
		return df

	def plot_cv_roc(self, folds):
		training_dedup = self.training.drop_duplicates(inplace = False)

		fig = plt.figure(figsize = (16,12))
		ax = plt.subplot(111)

		cv = StratifiedKFold(training_dedup.like, 
							 n_folds=folds)

		for i, (train, test) in enumerate(cv):
			fold_train_unbal = training_dedup.iloc[train, :]
			# oversample
			fold_train = self.balance_oversample(fold_train_unbal)
			X_fold_train = fold_train.drop('like', axis = 1)
			y_fold_train = fold_train.like
			
			fold_test = training_dedup.iloc[test, :]
			# only continue if there's a positive example
			# in training or test set
			# TODO: make this handle negative case also
			if 1 in y_fold_train.values and \
			  1 in fold_test.like.values:	    
				X_fold_test = fold_test.drop('like', axis = 1)
				y_fold_test = fold_test.like

				pred = self.model.fit(X_fold_train, y_fold_train)\
								 .predict(X_fold_test)
				pred_df = DataFrame(zip(pred, y_fold_test))
				pred_df.columns = 'predicted','actual'
				
				fpr, tpr, roc_auc = self.roc_metrics(pred_df)
				plt.plot(fpr, tpr, 
					label='Fold{0} (area = {1})'\
						.format(i, '{:.2f}'.format(roc_auc)))

		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC Curve')
		plt.legend(loc="lower right")
		plt.show()

	def plot_roc(self):
		fpr, tpr, roc_auc = self.roc_metrics()
		self.roc_auc = roc_auc

		plt.figure()
		plt.plot(fpr, tpr, 
			label='ROC curve (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC Curve')
		plt.legend(loc="lower right")
		plt.show()

	def report(self, n_top=3):
		top_scores = sorted(self.grid_scores, 
							key=itemgetter(1), 
							reverse=True)[:n_top]
		for i, score in enumerate(top_scores):
			print("Model with rank: {0}".format(i + 1))
			print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
				  score.mean_validation_score,
				  np.std(score.cv_validation_scores)))
			print("Parameters: {0}".format(score.parameters))
			print("")

	def score_test_set(self):
		print self.model
		if self.algo == 'xgb_no_cv':
			dm = xgb.DMatrix(self.test.drop('like', axis = 1))
			pred = self.model.predict(dm)
		else:
			pred = self.model.predict(self.test.drop('like', axis = 1))
			
		self.predictions = DataFrame(zip(pred, self.test.like))
		self.predictions.columns = 'predicted', 'actual'		

	def fit_model(self, model):
		model.fit(self.training.drop('like', axis = 1), 
				  self.training.like)
		return model
		
	def grid_search(self, model):
		model_cv = GridSearchCV(cv=5, 
					   estimator=model,
					   param_grid=self.params)    
		self.fit_model(model_cv)

		self.grid_scores = model_cv.grid_scores_

		return model_cv.best_estimator_

	def train_xgb_no_cv(self):
		dtrain = xgb.DMatrix(self.training.drop('like', axis = 1), 
							 label = self.training.like)

		evallist  = [(dtrain, 'train')]#[(dtest,'eval'), (dtrain,'train')]
		num_round = 10
		bst = xgb.train(self.params, dtrain, num_round, evallist)
		return bst


	def train_xgb(self):
		# https://www.kaggle.com/tanitter/introducing-kaggle-scripts/grid-search-xgboost-with-scikit-learn/run/23363

		model = XGBoostClassifier(
			eval_metric = 'auc',
			num_class = 2,
			nthread = 4,
			eta = 0.1,
			num_boost_round = 80,
			max_depth = 12,
			subsample = 0.5,
			colsample_bytree = 1.0,
			silent = 1,
			)

		X = self.training.drop('like', axis = 1)
		y = self.training.like

		model = GridSearchCV(model, self.params, n_jobs=1, cv=5)

		model.fit(X, y)
		best_parameters, score, _ = max(model.grid_scores_, key=lambda x: x[1])
		print score
		for param_name in sorted(best_parameters.keys()):
			print("%s: %r" % (param_name, best_parameters[param_name]))

		print 'best', best_parameters

		return model.best_estimator_

	def train_model(self):
		# xgboost is a special case
		if self.algo == 'xgb':
			self.model = self.train_xgb()
			return

		if self.algo == 'xgb_no_cv':
			self.model = self.train_xgb_no_cv()
			return

		if self.algo == 'rf':
			base_model = RandomForestClassifier(n_estimators=2)

		elif self.algo == 'nb':
			base_model = BernoulliNB(binarize=0.0, fit_prior=True, 
								class_prior=None)

		elif self.algo == 'lr':
			base_model = LogisticRegression(penalty='l1', dual=False, 
					   tol=0.0001, C=5.0, fit_intercept=True, 
					   intercept_scaling=1, class_weight=None, 
					   random_state=None, solver='liblinear', max_iter=100, 
					   multi_class='ovr', verbose=0, 
					   warm_start=False, n_jobs=1)

		elif self.algo == 'enet':
			base_model = ElasticNet(alpha=1.0, l1_ratio=0.3, 
				fit_intercept=True, normalize=False, precompute=False, 
				max_iter=1000, copy_X=True, tol=0.0001, warm_start=False,
				positive=False, random_state=None, selection='cyclic')

		elif self.algo == 'svm':
			base_model = SVC(gamma='auto', 
					  coef0=0.0, shrinking=True, probability=True, 
					  tol=0.001, cache_size=200,  
					  verbose=False, max_iter=-1, decision_function_shape=None, 
					  random_state=None)

		elif self.algo == 'lr_enet':
			base_model = SGDClassifier(loss='squared_loss', penalty='elasticnet', 
									   fit_intercept=True, n_iter=5, 
									   shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, 
									   random_state=None, learning_rate='optimal', 
									   eta0=0.0, power_t=0.5, class_weight=None, 
									   warm_start=False, average=False)			

		elif self.algo == 'sgd':
			base_model = SGDClassifier(penalty='elasticnet', fit_intercept=True, 
										n_iter=5, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, 
										learning_rate='optimal', eta0=0.0, power_t=0.5, warm_start=False,
										average=False)

		self.model = self.grid_search(base_model)