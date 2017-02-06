from pandas import *
import subprocess
from subprocess import Popen, PIPE
import os
os.chdir('/data1/amazon/data')

boots = read_csv('boots.csv', index_col = 'Unnamed: 0')
cols = [c.replace("'",'') for c in boots.columns]
boots.columns = cols

def run(cmd, output = False):
    if output == True:
        return Popen([cmd],stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True).communicate()[0].split('\n')[:-1]
    out = Popen([cmd], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True).communicate()

run('mkdir pics_boots')

pic_locations = boots.loc[:, 'asin'].apply(lambda x: 'pics/{0}.jpg'.format(x)).values

mv = [run('mv {0} pics_boots'.format(l)) for l in pic_locations]

def sex_of_boots(x):
    for woman_word in ['women', 'woman', 'ladies', 'lady', 'girl']:
        if woman_word in x:
            return 'women'
    for man_word in ['men', 'man', 'boy']:
        if man_word in x:
            return 'men'
    for child_word in ['kid', 'child', 'youth', 'baby', 'infant']:
        if child_word in x:
            return 'children'
    for unisex_word in ['unisex', 'adult']:
        if unisex_word in x:
            return 'unisex'
    else:
        return 'unknown'

boots.loc[:, 'sex'] = boots.loc[:, 'title'].apply(lambda x: sex_of_boots(x.lower()))

print boots.sex.value_counts()

boots.loc[:, 'image_path'] = boots.loc[:, 'asin'].apply(lambda x: 'pics_boots/{0}.jpg'.format(x)).values

# boots.to_csv('boots_aws.csv')