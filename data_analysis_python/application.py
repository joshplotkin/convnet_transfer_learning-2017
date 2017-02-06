
from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from os import walk
import csv
import random
import redis
import json
import pickle
import numpy 

from beta_bandit import BetaBandit 
from distance import *
from classifier import *
from clustering_sparse import *

application = Flask(__name__)

USERS=['Erin','Dave','Josh','Priya']

@application.route('/')
def homepage():
	#Home path requires local version of csv and images
	args=request.args
	if 'user' not in args:
		return redirect("?user="+USERS[0], code=302)
	likes = redis.Redis(db=1).smembers(args['user'])
	dislikes = redis.Redis(db=0).smembers(args['user'])

	if len(likes) + len(dislikes) == 0:
		cluster_dict = pickle.load(open("outputs/clusters.txt","rb"))
		bb = BetaBandit(num_options=int(len(cluster_dict.keys())))
		ic = IncrementalClassifier(features, cluster_dict)

	cluster_key = bb.get_recommendation()

	ic.extract_previous(args['user'])
	ic.update_unseen()
	candidates = ic.get_candidates_cluster(cluster_key)

	#logic for empty set vs set with process
	if len(likes) == 0 or len(dislikes) == 0:
		boot = calc_sim(dist_df,likes,dislikes,cluster_dict,cluster_key)
	else:
		if (len(likes) + len(dislikes)) % 10 == 0 \
		  or ic.model is None:
			ic.batch_train(likes, dislikes)
			# every 10 iterations, update clusters based on L1 subset selection
			clustering = ClusteringImages(features, 
						    ic.model.coef_,
						    'VBGMM',
						    1,
						    25)
			cluster_dict = clustering.cluster_dict
			bb = clustering.count_votes_by_cluster(likes, dislikes)
		else:
			ic.online_train()
		boot = ic.score_unseen(ic.get_candidates_cluster(cluster_key))[0] 

	impath=data.loc[boot, 'imUrl']
	return render_template('index.html',string=impath,asin=boot,users=USERS,activeuser=args['user'],cluster=cluster_key)

def get_image(path):
	f = []
	for (dirpath, dirnames, filenames) in walk(path):
		f.extend(filenames)
	impath=random.choice(f)
	return impath[0:-4]

def load_data(path):
	lookup_dict={}
	with open(path, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in spamreader:
			lookup_dict[row[4]]={"url":row[6]}
	return lookup_dict

@application.route('/userchoices')
#diplay user choices page
def choices():
	args=request.args
	if 'user' not in args:
		return redirect("userchoices?user="+USERS[0], code=302)
	conn = redis.Redis(db=1)
	likes = conn.smembers(args['user'])
	links=[data.loc[x, 'imUrl'] for x in likes]

	return render_template('choices.html',vals=links,users=USERS,activeuser=args['user'])

@application.route('/groupview')
#diplay group images, based on existing HTML template pages
def groupview():
	args=request.args
	if 'user' not in args:
		return redirect("groupview?user="+USERS[0]+"&groupnum=1", code=302)
	if 'groupnum' not in args:
		return redirect("groupview?user="+USERS[0]+"&groupnum=1", code=302)
	num=args['groupnum']
	groups=["Group "+ str(x) for x in range(0,25)]
	return render_template('./groups/group'+str(num)+'.html',users=USERS,activeuser=args['user'],showgroups="Yes",groups=groups)

@application.route('/suggestions')
#show suggestions page
def suggestions():
	args=request.args
	if 'user' not in args:
		return redirect("suggestions?user="+USERS[0], code=302)
	conn = redis.Redis(db=1)
	likes = conn.smembers(args['user'])
	conn_ = redis.Redis(db=0)
	dislikes = conn_.smembers(args['user'])

	## JOSH
	if len(likes) == 0 or len(dislikes) == 0:
		recs=calc_sim_rec(dist_df,likes,dislikes,15,False)
		recs=list(recs)
	else:
		try:
			recs = ic.score_unseen(ic.get_candidates_cluster(), 15)
		except:
			recs=list(calc_sim_rec(dist_df,likes,dislikes,15,False))
			

	rec_urls=[data.loc[boot, 'imUrl'] for boot in recs]
	clusters=[]
	for rec in recs:
		for key in cluster_dict:
			if rec in cluster_dict[key]:
				clusters.append(key)
	return render_template('suggestions.html',users=USERS,activeuser=args['user'],recs=rec_urls,clusters=clusters)

@application.route('/flush')
# reset databases
def flush():
	conn = redis.Redis(db=0)
	conn2 = redis.Redis(db=1)
	conn.flushdb()
	conn2.flushdb()
	return redirect("/?user="+USERS[0], code=302)

@application.route('/labeler')
# test application for labeling for classification - not used in core project
def labeler():
	args=request.args
	if 'user' not in args:
		return redirect("labeler?user="+USERS[0], code=302)

	data=load_data('./data/boots_aws.csv')
	rand = random.choice(list(data.keys()))
	impath=data.loc[rand, 'imUrl']

	return render_template('labeler.html',users=USERS,activeuser=args['user'],string=impath,asin=rand,labels=LABS)

@application.route('/submit')
# API route for submitting choices
def submit():
	args=request.args
	bb.add_result(int(args['cluster']),int(args['like']))
	if float(args['like'])==0:
		conn = redis.Redis(db=0)
		conn.sadd(args['user'],args['asin'])

	if float(args['like'])==1:
		conn = redis.Redis(db=1)
		conn.sadd(args['user'],args['asin'])
	
	## for classifier
	with open('/root/classifier/{0}.txt'.format(args['user']),'w') as w:
		w.write('{0}\t{1}'.format(args['asin'], args['like']))

	return json.dumps({"stored":True})

@application.route('/submitLabel')
# for label classifier test
def submit_label():
	args=request.args
	label=args['label']
	asin=args['asin']
	f=open('./data/labels.txt','a+')
	f.write('{"'+asin+'":"'+label+'"}\n')
	return json.dumps({"stored":True})

if __name__ == '__main__':
	# preload data distance matrix and clusters  for faster run time
	data = pd.read_csv('./data/metadata_women_042016.csv',
					index_col = 'asin') 
	f = open('./data/dist_df.txt','rb')
	dist_df  = pickle.load(f)
	women_asin = data.index.values
	dist_df = dist_df.loc[women_asin, women_asin]
	features = pd.read_csv('./data/features_women.csv',
			index_col = 'asin')
	
	application.run(host='0.0.0.0', debug=True, port=5555)
