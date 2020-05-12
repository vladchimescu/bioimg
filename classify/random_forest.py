#!/usr/bin/env python
"""
Image classification with random forest
"""
import numpy as np
import matplotlib.pyplot as plt
import re
import h5py
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from plotly.offline import iplot
from ..extra.viz import plotly_viz, plotly_predictions


class IncrementalClassifier:
    def __init__(self):
        # initialize with 'None' something to be loaded later
        self.imgx = None

        self.newlabels = None
        # training data
        self.Xtrain = None
        self.ytrain = None
        # inialize classifier as 'None'
        self.clf = None
        self.classes = None

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        # if a new ImgX object is passed,
        # compute its features
        if name == 'imgx':
            self._compute_imgx_data()
            self.newlabels = None

    # function for setting individual class parameters
    def set_param(self, **kwargs):
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

    # internal function checks if the embedded
    # imgx object has the features computed
    def _compute_imgx_data(self):
        if self.imgx is not None and len(self.imgx.data) == 0:
            self.imgx.compute_props()

    # plot predictions overlaid with the original image
    # plot is a 'void' function (returns 'None')
    def plot_predictions(self):
        if self.imgx.y is not None:
            layout, feats = plotly_predictions(img=self.imgx.img,
                                               bb=self.imgx.bbox,
                                               ypred=self.imgx.y,
                                               labels=self.classes
                                               )
        else:
            layout, feats = plotly_viz(img=self.imgx.img,
                                       bb=self.imgx.bbox)
        iplot(dict(data=feats, layout=layout))

    def add_instances(self, newlabels):
        newlabels = np.unique(newlabels, axis=0)
        if self.newlabels is None:
            self.newlabels = newlabels
        else:
            a1_rows = newlabels.view(
                [('', newlabels.dtype)] * newlabels.shape[1])
            a2_rows = self.newlabels.view(
                [('', self.newlabels.dtype)] * self.newlabels.shape[1])

            newlabels = (np.setdiff1d(a1_rows, a2_rows).
                         view(newlabels.dtype).
                         reshape(-1, newlabels.shape[1]))
            self.newlabels = np.append(self.newlabels, newlabels, axis=0)
        # if 'newlabels' array is not empty
        if len(newlabels) > 0:
            self._push_traindata(newlabels=newlabels)
        return self

    def _push_traindata(self, newlabels):
        ids = newlabels[:, 0]
        if self.Xtrain is None:
            self.Xtrain = self.imgx.data.iloc[ids, :]
            self.ytrain = label_binarize(newlabels[:, 1],
                                         classes=range(len(self.classes)))
        else:
            self.Xtrain = pd.concat(
                [self.Xtrain, self.imgx.data.iloc[ids, :]], axis=0)
            self.ytrain = np.append(self.ytrain, label_binarize(newlabels[:, 1],
                                                                classes=range(len(self.classes))), axis=0)

    def set_classifier(self, clf=None):
        self.clf = clf
        # if 'None' then some reasonable default
        if clf is None:
            self.clf = OneVsRestClassifier(RandomForestClassifier(bootstrap=True,
                                                                  class_weight="balanced",
                                                                  n_estimators=500,
                                                                  random_state=123,
                                                                  n_jobs=-1))
        return self

    def train_classifier(self):
        self.clf.fit(self.Xtrain, self.ytrain)
        return self

    # print the confusion matrix on the existing training set
    def train_error(self):
        ypred = self.clf.predict(self.Xtrain)
        print(classification_report(self.ytrain.argmax(axis=1),
                                    ypred.argmax(axis=1),
                                    target_names=self.classes))
        # print(confusion_matrix(self.ytrain.argmax(axis=1), ypred.argmax(axis=1),
        #                       labels=range(len(self.classes))))

    # generate predictions and pass them to self.imgx.y
    def generate_predictions(self, prob=False):
        Xtest = self.imgx.data
        ypred = self.clf.predict(Xtest)
        # set labels to these
        self.imgx.y = ypred.argmax(axis=1)

    def h5_write(self, fname):
        hf = h5py.File(fname, 'w')
        hf.create_dataset('Xtrain', data=self.Xtrain)
        hf.create_dataset('ytrain', data=self.ytrain.argmax(axis=1))
        hf.close()
