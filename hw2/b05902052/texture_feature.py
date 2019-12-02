#!/usr/bin/env python
# coding: utf-8

# In[1]:
'''
Get list of pictures.
'''

import glob

data_dir = 'data/'
pics = glob.glob(data_dir + '*/*/*.jpg')


# In[2]:
'''
Define the Picture namedtuple, and set the reshape size.
'''

from collections import namedtuple
from PIL import Image
import numpy as np
import cv2

Picture = namedtuple('Picture', ['category', 'camera', 'index', 'feature'])
maxsize = 200


# In[3]:
'''
Convert filename into its histogram-type feature.
Using FFT method.
'''

def filename_to_hist_feat(filename):
    image = Image.open(filename)
    minsize = min(image.size[0], image.size[1])
    ymin, xmin = int((image.size[0] - minsize) / 2), int((image.size[1] - minsize) / 2)
    image = np.array(image)[xmin:xmin+minsize, ymin:ymin+minsize, :]
    image = Image.fromarray(image)
    image = image.resize((maxsize, maxsize), Image.ANTIALIAS)
    image = np.array(image)
    
    chans = cv2.split(image)
    feat = []
    for chan in chans:
        f = np.fft.fft2(chan)
        fshift = np.fft.fftshift(f)
        spectrum = np.log(np.abs(fshift))
        feat.append(spectrum.reshape(-1))
    feat = np.concatenate(feat)
    names = filename.split('/')
    return Picture(category=names[1], camera=names[2], index=names[3], feature=feat)


# In[4]:
'''
Given filename of picture, get its histogram-type feature. (Multiprocess)
'''

import concurrent.futures
import time

start = time.time()
chunksize = 32
ref_pics, query_pics = [], []
with concurrent.futures.ProcessPoolExecutor() as executor:
    for filename, feat_pic in zip(pics, executor.map(filename_to_hist_feat, pics, chunksize=chunksize)):
        if feat_pic.camera == 'Reference':
            ref_pics.append(feat_pic)
        else:
            query_pics.append(feat_pic)

feature_mat = np.array([pic.feature.reshape(-1) for pic in ref_pics])
id2info = {idx:(pic.category, pic.index) for idx, pic in enumerate(ref_pics)}            

print('Calculate Features in %.4f seconds...' % (time.time() - start))


# In[5]:
'''
Implement several distance / similarity metrics.
'''

class ScoreFunctions(object):
    def cosine_similarity(u, v):
        return u @ v / np.linalg.norm(u) / np.linalg.norm(v)
    def l1_dist(u, v):
        return -np.linalg.norm((u - v), 1)
    def l2_dist(u, v):
        return -np.linalg.norm((u - v), 2)


# In[6]:
'''
Given a query, returning the reference list sorted by similarity.
'''

def retrieval(query, func, top=5):
    score_list = [func(feat, query) for feat in feature_mat]
    score_list = sorted(list(enumerate(score_list)), key=lambda x:x[1], reverse=True)[:top]
    index_list = [index for index, score in score_list]
    info_list = [id2info[index] for index in index_list]
    return info_list


# In[7]:
'''
Check whether the retrieval result hits the correct reference.
'''

def retrieval_correct(query, ref):
    return (query.category, query.index) == ref


# In[8]:
'''
Given query, return two booleans:
(successful retrieval or not at top1, successful retrieval or not at top5)
'''

def Sat1and5(q, func=ScoreFunctions.cosine_similarity):
    rankings = retrieval(q.feature.reshape(-1), func=func)
    Sat1 = any([retrieval_correct(q, r) for r in rankings[:1]])
    Sat5 = any([retrieval_correct(q, r) for r in rankings[:5]])
    return Sat1, Sat5


# In[9]:
'''
Given distance metric, calculate the S@1 and S@5 score of the computed features. (Multiprocess)
'''

from functools import partial
from collections import defaultdict

chunksize = 32
Sat1s, Sat5s = defaultdict(list), defaultdict(list)
with concurrent.futures.ProcessPoolExecutor() as executor:
    for q, (Sat1, Sat5) in zip(query_pics, executor.map(partial(Sat1and5, func=ScoreFunctions.l2_dist), query_pics, chunksize=chunksize)):
        Sat1s[q.category].append(Sat1)
        Sat5s[q.category].append(Sat5)

for (c1, sat1), (c2, sat5) in zip(Sat1s.items(), Sat5s.items()):
    assert c1 == c2
    print('Category:', c1)
    print('S@1: %.4f' % (sum(sat1) / len(sat1)), 'S@5: %.4f' % (sum(sat5) / len(sat5)))