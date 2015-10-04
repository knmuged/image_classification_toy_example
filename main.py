#!/usr/bin/env python
#-*-coding:utf-8-*-

# https://www.youtube.com/watch?v=-Ja-vLbHWLc

import sys
import mahotas
from mahotas.features.lbp import lbp
import numpy as np
from sklearn import svm

# settings for LBP
radius = 2
n_points = 16
# 4116 dimensional lbp histogram
X = []
Y = []

with open(sys.argv[1]) as f_in:
    for filename in f_in:
        img = mahotas.imread(filename.rstrip(), as_grey = True)
        img.resize(100,100)
        lbp_hist = lbp(img, radius, n_points)
        lbp_hist /= np.linalg.norm(lbp_hist)
        X.append(lbp_hist)
        if 'background' in filename:
            Y.append(0)
        else:
            Y.append(1)

clf = svm.SVC()
clf.fit(X, Y)  
rate = 0.0
counter = 0

with open(sys.argv[2]) as f_in:
    for filename in f_in:
        img = mahotas.imread(filename.rstrip(), as_grey = True)
        img.resize(100,100)
        lbp_hist = lbp(img, radius, n_points)
        lbp_hist /= np.linalg.norm(lbp_hist)
        pred = clf.predict(lbp_hist)[0]
        if 'background' in filename and pred == 0.0:
            rate += 1
        if 'watch' in filename and pred == 1.0:
            rate += 1
        counter += 1

print 'accuracy: ', rate/counter


