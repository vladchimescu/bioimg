#!/usr/bin/env python
"""
Image classification with random forest
"""
import numpy as np
import functions as fn
import visualize as vi
from plotly.offline import iplot
from skimage.exposure import equalize_adapthist
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import h5py

from _internals import get_regionprop_feats


def convert_well_name(fname):
    pass


exclude = []


class IncrementalClassifier:
    def __init__(self, **kwargs):

        # width and height of a standardised image
        self.w = 40
        self.h = 40

        # some default arguments
        self.path = kwargs.get("path")   # image path
        # directory where the pre-computed instance bounding boxes are stored
        self.featdir = kwargs.get("featdir")
        # selected well in the view
        self.select_well = kwargs.get("select_well")
        self.target_names = kwargs.get("target_names")  # class names
        self.X_train_norm = kwargs.get("X_train_norm")
        self.X_train_prop = kwargs.get("X_train_prop")
        self.y_train = kwargs.get("y_train")

        # initialize with 'None' something to be loaded later
        self.img = None
        self.bb = None
        self.cellbb = None
        self.y_pred = None
        self.y_prob = None
        self.X_test_norm = None
        self.X_test_prop = None
        self.layout = None
        self.feats = None
        self.newlabels = None
        # inialize classifier as 'None'
        self.clf = None
        self.pca = None

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    # function for setting individual class parameters
    def set_param(self, **kwargs):
        for k in kwargs.keys():
            if k in ['path', 'featdir', 'select_well',
                     'target_names', 'w', 'h']:
                self.__setattr__(k, kwargs[k])

    # load well image and feature data
    def load_img(self):
        fl = self.featdir + self.select_well
        self.bb, self.cellbb = fn.get_instances_from_well(
            fname=fl, imgpath=self.path)
        return self

    # EDIT this one
    # generate predictions with the loaded feature data
    def generate_predictions(self, prob=False):
        if self.X_test_norm is None:
            cellbb_norm = [resize(cb, (self.w, self.h), anti_aliasing=True)
                           for cb in self.cellbb]
            self.X_test_norm = np.array([cbn.ravel() for cbn in cellbb_norm])

        if self.X_test_prop is None:
            X_prop_list = [get_regionprop_feats(
                mip_rgb=cbb, exclude=exclude) for cbb in self.cellbb]
            self.X_test_prop = np.vstack(X_prop_list)

        X_test_pca = self.pca.transform(self.X_test_norm)
        X_test_all = np.append(X_test_pca, self.X_test_prop, axis=1)

        self.y_pred = self.clf.predict(X_test_all)
        if prob:
            self.y_prob = self.clf.predict_proba(X_test_all)
        return self

    # set plotly graphical layers
    def set_scene(self):
        # derived object attributes
        wellpos = convert_well_name(self.select_well)

        mip = fn.get_mip(path=self.path, wellpos=wellpos)
        mip_rgb = equalize_adapthist(np.dstack((mip[:, :, 1],
                                                mip[:, :, 2],
                                                mip[:, :, 0])))
        self.img = mip_rgb
        if self.y_pred is not None:
            self.layout, self.feats = vi.plotly_predictions(img=mip_rgb,
                                                            bb=self.bb,
                                                            y_pred=self.y_pred,
                                                            target_names=self.target_names)
        else:
            self.layout, self.feats = vi.plotly_viz(img=mip_rgb,
                                                    bb=self.bb)
        return self

    # plot predictions overlaid with the original image
    # plot is a 'void' function (returns 'None')
    def plot(self):
        iplot(dict(data=self.feats, layout=self.layout))

    # update scene after refitting the pipeline
    def update_scene(self):
        self.feats = vi.update_feats(img=self.img,
                                     bb=self.bb,
                                     y_pred=self.y_pred,
                                     target_names=self.target_names)
        return self

    def show_subset(self, inds):
        self.feats = vi.update_feats(img=self.img,
                                     bb=[self.bb[i] for i in inds],
                                     y_pred=self.y_pred[inds],
                                     target_names=self.target_names)
        return self

    def add_training_set(self, newlabels):
        if newlabels.size:
            cellbb_new = [self.cellbb[i] for i in newlabels[:, 0]]
            cellbb_new_norm = [resize(cb, (self.w, self.h), anti_aliasing=True)
                               for cb in cellbb_new]

            self.X_train_norm = np.array(
                [cbn.ravel() for cbn in cellbb_new_norm])
            self.y_train = newlabels[:, 1]

            X_prop_list = [get_regionprop_feats(
                mip_rgb=cbb, exclude=exclude) for cbb in cellbb_new]
            self.X_train_prop = np.vstack(X_prop_list)

            return self

    def add_instances(self, newlabels):
        if self.newlabels is None:
            self.newlabels = newlabels
        else:
            a1_rows = newlabels.view(
                [('', newlabels.dtype)] * newlabels.shape[1])
            a2_rows = self.newlabels.view(
                [('', self.newlabels.dtype)] * self.newlabels.shape[1])

            self.newlabels = newlabels
            newlabels = (np.setdiff1d(a1_rows, a2_rows).
                         view(newlabels.dtype).
                         reshape(-1, newlabels.shape[1]))

        if newlabels.size:
            cellbb_new = [self.cellbb[i] for i in newlabels[:, 0]]
            cellbb_new_norm = [resize(cb, (self.w, self.h), anti_aliasing=True)
                               for cb in cellbb_new]
            X_new_norm = np.array([cbn.ravel() for cbn in cellbb_new_norm])

            self.X_train_norm = np.concatenate((self.X_train_norm, X_new_norm))
            self.y_train = np.append(self.y_train, newlabels[:, 1])

            X_prop_list = [get_regionprop_feats(
                mip_rgb=cbb, exclude=exclude) for cbb in cellbb_new]
            X_new_prop = np.vstack(X_prop_list)
            self.X_train_prop = np.concatenate((self.X_train_prop, X_new_prop))
        return self

    def train_classifier(self, n_components=150,
                         bootstrap=True,
                         max_depth=None,
                         n_estimators=100,
                         max_features='sqrt',
                         min_samples_split=5,
                         min_samples_leaf=5,
                         random_state=100):

        self.pca = PCA(n_components=n_components, svd_solver='randomized',
                       whiten=True, random_state=random_state).fit(self.X_train_norm)

        # project the train data
        X_train_pca = self.pca.transform(self.X_train_norm)
        X_train_all = np.append(X_train_pca, self.X_train_prop, axis=1)

        self.clf = RandomForestClassifier(bootstrap=bootstrap,
                                          class_weight="balanced",
                                          max_depth=max_depth,
                                          n_estimators=n_estimators,
                                          max_features=max_features,
                                          min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf,
                                          random_state=random_state,
                                          n_jobs=-1)

        self.clf.fit(X_train_all, self.y_train)
        return self

    # print the confusion matrix on the existing training set
    def get_classifiction_report(self):
        X_train_pca = self.pca.transform(self.X_train_norm)
        X_train_all = np.append(X_train_pca, self.X_train_prop, axis=1)
        y_pred = self.clf.predict(X_train_all)
        print(classification_report(self.y_train,
                                    y_pred, target_names=self.target_names))
        print(confusion_matrix(self.y_train, y_pred,
                               labels=range(len(self.target_names))))
        return self

    def get_feat_importance(self, max_feats=20):
        importances = self.clf.feature_importances_

        std = np.std([tree.feature_importances_ for tree in self.clf.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(max_feats):
            print("%d. feature %d (%f)" %
                  (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(self.X_train_prop.shape[1] + 150), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(self.X_train_prop.shape[1] + 150), indices)
        plt.xlim([-1, max_feats])

        return self

    def get_cross_val_score(self, kfold=5):
        X_train_pca = self.pca.transform(self.X_train_norm)
        X_train_all = np.append(X_train_pca, self.X_train_prop, axis=1)
        scores = cross_val_score(self.clf, X_train_all, self.y_train, cv=kfold)
        print scores

        return self

    def get_low_score(self, cl=0, n=1):
        if self.y_prob is not None:
            maxprob = np.max(self.y_prob, axis=1)
            probrank = maxprob.argsort().argsort()
            minrank = np.sort(probrank[self.y_pred == cl])[:n]
            return np.where(np.isin(probrank, minrank))[0]

    def reset(self):
        self.newlabels = None
        self.X_test_norm = None
        self.X_test_prop = None
        return self

    def h5_write(self, fname):
        hf = h5py.File(fname, 'w')
        hf.create_dataset('X_train_norm', data=self.X_train_norm)
        hf.create_dataset('X_train_prop', data=self.X_train_prop)
        hf.create_dataset('y_train', data=self.y_train)
        hf.close()
