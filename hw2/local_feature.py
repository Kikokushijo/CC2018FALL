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
Define the Picture namedtuple.
'''

from collections import namedtuple
from PIL import Image
import numpy as np
import cv2

Picture = namedtuple('Picture', ['category', 'camera', 'index', 'feature'])


# In[3]:
'''
Convert filename into its keypoints feature.
Using DoG + SIFT method.
'''

def filename_to_keypoints(filename):
    image = Image.open(filename)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    names = filename.split('/')
    return Picture(category=names[1], camera=names[2], index=names[3], feature=des)


# In[4]:
'''
Given filename of picture, get its keypoints feature. (Multiprocess)
If there exists pre-computed keypoints data, skip the DoG + SIFT process.
'''

import concurrent.futures
import time
import pickle
import os

if os.path.isfile('key_ref_pics.pickle') and os.path.isfile('key_query_pics.pickle'):
    with open('key_ref_pics.pickle', 'rb') as f, open('key_query_pics.pickle', 'rb') as g:
        ref_pics_key = pickle.load(f)
        query_pics_key = pickle.load(g)
    key_feature_mat = np.load('key_feature_mat.npy')
    
else:
    start = time.time()
    chunksize = 32
    ref_pics_key, query_pics_key = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        for filename, feat_pic in zip(pics, executor.map(filename_to_keypoints, pics, chunksize=chunksize)):
            if feat_pic.camera == 'Reference':
                ref_pics_key.append(feat_pic)
            else:
                query_pics_key.append(feat_pic)

    key_feature_mat = np.concatenate([pic.feature for pic in ref_pics_key])

    print('Calculate Features in %.4f seconds...' % (time.time() - start))
    
    with open('key_ref_pics.pickle', 'wb') as f, open('key_query_pics.pickle', 'wb') as g:
        pickle.dump(ref_pics_key, f)
        pickle.dump(query_pics_key, g)

id2info = {idx:(pic.category, pic.index) for idx, pic in enumerate(ref_pics_key)}


# In[27]:
'''
Use K-Means clustering method transform ~1500k keypoints into 4096 clusters.
'''

from sklearn.cluster import MiniBatchKMeans
n_clusters = 4096
kmeans = MiniBatchKMeans(init_size=n_clusters*3, n_clusters=n_clusters, random_state=0, verbose=0).fit(key_feature_mat)


# In[18]:
'''
Given the keypoints of a picture, calculate the histogram of the codewords.
'''

def key_to_SIFT_hist_feat(picture, threshold=0.75):
    feat = picture.feature
    cluster_dist = kmeans.transform(feat)
    cluster_pred = kmeans.predict(feat)
    hist = np.zeros((n_clusters,))
    for d, pred in zip(cluster_dist, cluster_pred):
        first, second = sorted(d)[:2]
        if first <= second * threshold:
            hist[pred] += 1
    return Picture(category=picture.category, camera=picture.camera, index=picture.index, feature=hist)


# In[28]:
'''
Given filename of picture, get its Histogram of SIFT codewords feature. (Multiprocess)
'''

import concurrent.futures
import time
from functools import partial

start = time.time()
chunksize = 32
ref_pics, query_pics = [], []
with concurrent.futures.ProcessPoolExecutor() as executor:
    for filename, feat_pic in zip(pics, executor.map(partial(key_to_SIFT_hist_feat, threshold=1.0), ref_pics_key+query_pics_key, chunksize=chunksize)):
        if feat_pic.camera == 'Reference':
            ref_pics.append(feat_pic)
        else:
            query_pics.append(feat_pic)

feature_mat = np.concatenate([[pic.feature] for pic in ref_pics])

print('Calculate Features in %.4f seconds...' % (time.time() - start))


# In[34]:
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
    def chi_square(u, v):
        return -cv2.compareHist(u.astype(np.float32), v.astype(np.float32), cv2.HISTCMP_CHISQR_ALT)
    def intersect(u, v):
        return cv2.compareHist(u.astype(np.float32), v.astype(np.float32), cv2.HISTCMP_INTERSECT)


# In[12]:
'''
Given a query, returning the reference list sorted by similarity.
'''

def retrieval(query, func, top=5):
    score_list = [func(feat, query) for feat in feature_mat]
    score_list = sorted(list(enumerate(score_list)), key=lambda x:x[1], reverse=True)[:top]
    index_list = [index for index, score in score_list]
    info_list = [id2info[index] for index in index_list]
    return info_list


# In[13]:
'''
Check whether the retrieval result hits the correct reference.
'''

def retrieval_correct(query, ref):
    return (query.category, query.index) == ref


# In[14]:
'''
Given query, return two booleans:
(successful retrieval or not at top1, successful retrieval or not at top5)
'''

def Sat1and5(q, func=ScoreFunctions.cosine_similarity):
    rankings = retrieval(q.feature.reshape(-1), func=func)
    Sat1 = any([retrieval_correct(q, r) for r in rankings[:1]])
    Sat5 = any([retrieval_correct(q, r) for r in rankings[:5]])
    return Sat1, Sat5


# In[36]:
'''
Given distance metric, calculate the S@1 and S@5 score of the computed features. (Multiprocess)
'''

from functools import partial
from collections import defaultdict

chunksize = 32
Sat1s, Sat5s = defaultdict(list), defaultdict(list)
with concurrent.futures.ProcessPoolExecutor() as executor:
    for q, (Sat1, Sat5) in zip(query_pics, executor.map(partial(Sat1and5, func=ScoreFunctions.chi_square), query_pics, chunksize=chunksize)):
        Sat1s[q.category].append(Sat1)
        Sat5s[q.category].append(Sat5)

for (c1, sat1), (c2, sat5) in zip(Sat1s.items(), Sat5s.items()):
    assert c1 == c2
    print('Category:', c1)
    print('S@1: %.4f' % (sum(sat1) / len(sat1)), 'S@5: %.4f' % (sum(sat5) / len(sat5)))