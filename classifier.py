from pandas import *
from sklearn.linear_model import SGDClassifier

class IncrementalClassifier:
    def __init__(self, features, cluster_dict):
        self.features = features
        self.unseen_features = self.features.copy()
        self.cluster_dict = cluster_dict
        self.model = None

    def extract_previous(self, user):
        with open('./classifier/{0}.txt'.format(user)) as f:
            if f.read() == '':
                self.asin_prev = ''
                self.like_prev = ''
            else:
                self.asin_prev, self.like_prev = \
                   open('./classifier/{0}.txt'\
                        .format(user))\
                    .read()\
                    .split('\t')

    def update_unseen(self):
        if self.asin_prev not in self.unseen_features.index:
            return self.unseen_features

        self.unseen_features.drop(self.asin_prev, axis = 0,
            inplace = True)

    def get_candidates_cluster(self, cluster_num=None):
        if cluster_num:
            return self.unseen_features.loc[
                    self.cluster_dict[cluster_num], :].dropna()
        return self.unseen_features

    def new_model(self, class_weight=None):
        return SGDClassifier(loss='hinge', penalty='elasticnet', 
             alpha=0.1, l1_ratio=0.85, fit_intercept=True, n_iter=5, 
             shuffle=True, verbose=0, epsilon=0.1, 
             n_jobs=1, random_state=None, learning_rate='optimal', 
             eta0=0.0, power_t=0.5, class_weight=class_weight, 
             warm_start=False, average=False)

    def score_unseen(self, candidates, n=1):
        scores = self.model.decision_function(self.unseen_features.values)
        top_n = (-scores).argsort()[:n]
        return list(self.unseen_features.iloc[top_n, :].index)

    def batch_train(self, likes, dislikes, n=1):
        self.model = self.new_model({0: len(likes),
                                     1: len(dislikes)})
    
        all_reviewed = [l for l in likes]
        all_reviewed.extend(dislikes)
        dataset = self.features.loc[all_reviewed, :]
        
        print 'trained batch'
        self.model.fit(dataset.values,
                [1 if l in likes else 0 for l in dataset.index])

    def online_train(self, n=1):
        print 'trained 1 datapoint'
        row = self.features.loc[self.asin_prev, :]
        self.model.partial_fit(row.values.reshape(1, -1), 
                  np.array(int(self.like_prev)).reshape(-1, 1))

