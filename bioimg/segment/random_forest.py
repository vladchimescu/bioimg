#!/usr/bin/env python3
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
from ..base.viz import plotly_viz, plotly_predictions


class IncrementalClassifier:
    '''Classifier for image patch classification
       -----------------------------------------
       This class is an interface for sklearn classifiers,
       adapted for classification of image patches in a labelled
       image. The user can accumulate training instances and
       visualize interactively the predictions on new images.
       Misclassified instances can be added as new training instances
       and the classifier can be re-trained and predictions re-run.

       Attributes
       ----------
       imgx : ImgX object
            Labelled image instance with the intensity image and
            segmentation (bounding boxes of ROIs)
       newlabels : array-like
           Array or list of new training instances
       Xtrain : np.array
           Train data with observations (e.g. cells) in rows
           and morphological features in columns
       ytrain : np.array
           List or array of labels
       clf : sklearn classifier
           Classifier model. Default is RandomForestClassifier in
           one-vs-rest mode. For details see the documentation:
           (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
       classes : array-like
           List of class names, e.g. ['cell A', 'cell B', 'cell C']

       Methods
       -------
       set_param(**kwargs)
           Change any of the class attributes that have been set
       set_classifier(clf=None)
           Initializes a classifier object

       add_instances(newlabels)
           Add new training instances

       train_classifier()
           Train a classifier with the current training data

       generate_predictions()
           Generate predictions after the classifier has been
           trained.

       plot_predictions()
           Plot predictions using plotly interactive
           visualization

       h5_write(fname, group)
           Write the train set as HDF5 file
       
    '''
    def __init__(self):
        '''
        Parameters
        ----------
        imgx : ImgX object
            Labelled image instance with the intensity image and
            segmentation (bounding boxes of ROIs)
        newlabels : array-like
            Array or list of new training instances
        Xtrain : np.array
            Train data with observations (e.g. cells) in rows
            and morphological features in columns
        ytrain : np.array
            List or array of labels
        clf : sklearn classifier
            Classifier model. Default is RandomForestClassifier in
            one-vs-rest mode. For details see the documentation:
            (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
        classes : array-like
            List of class names, e.g. ['cell A', 'cell B', 'cell C']
        '''
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
        '''Change class attribute values
           -----------------------------
           Using this function we can change the current
           labelled image loaded (`imgx` attribute)
        '''
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

    # internal function checks if the embedded
    # imgx object has the features computed
    def _compute_imgx_data(self):
        if self.imgx is not None and len(self.imgx.data) == 0:
            self.imgx.compute_props()

    

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
        '''Initialize classifier
           ---------------------
           
           Parameters
           ----------
           clf : sklearn classifier (optional)
               By default a RandomForestClassifier with 500 estimators
               is initialized in one-vs-rest mode (for multi-class settings).
               Users can initialize any of the sklearn classifiers
               externally and provide as the `clf` argument.
        '''
        self.clf = clf
        # if 'None' then some reasonable default
        if clf is None:
            self.clf = OneVsRestClassifier(RandomForestClassifier(bootstrap=True,
                                                                  class_weight="balanced",
                                                                  n_estimators=500,
                                                                  random_state=123,
                                                                  n_jobs=-1))
        return self


    def add_instances(self, newlabels):
        '''Add new training instances
           --------------------------
           The function accepts a 2D array that for each new
           instance provides the numeric index in `imgx` instance
           and user defined class (as integer), e.g. 
           np.array([[12,0], [36,1]])
           
           Parameters
           ----------
           newlabels : array
               A 2D numpy.array is expected with an index of
               the labelled region and the class (as integer)
        '''
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

    def train_classifier(self):
        '''Fit a supervised model to the current training
           data. The function runs sklearn.clf.fit() on 
           Xtrain and ytrain that the user provided
        '''
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
        '''Generate predictions after the model was fit
           The function runs sklearn.clf.predict() on
           the currently loaded `imgx` instance
        '''
        Xtest = self.imgx.data
        ypred = self.clf.predict(Xtest)
        # set labels to these
        self.imgx.y = ypred.argmax(axis=1)

    # plot predictions overlaid with the original image
    # plot is a 'void' function (returns 'None')
    def plot_predictions(self):
        '''Plot predictions over the original image
           The function generates an interactive visualization
           with plotly. The user can hover over a labelled region
           (e.g. a cell) and the class of the region will be shown
           (if .generate_predictions() has been run prior to the 
           function call)
        '''
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

    def h5_write(self, fname, group):
        '''Write current train set as HDF5 file
          
           Parameters
           ----------
               fname : string
                   Path and file name (e.g. trainset.h5)
               group : string
                   Dataset / group name under which the
                   train data will be saved. For more details
                   see the documentation on HDF5 groups:
                   (http://docs.h5py.org/en/stable/high/group.html)
        '''
        hf = h5py.File(fname, 'w')
        hf.create_dataset(group + '/Xtrain', data=self.Xtrain)
        hf.create_dataset(group + '/ytrain', data=self.ytrain.argmax(axis=1))
        hf.create_dataset(group + '/columns',
                          data=self.Xtrain.columns.values.astype('S').tolist())
        hf.close()
