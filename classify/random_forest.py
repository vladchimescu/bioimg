#!/usr/bin/env python
"""
Image classification with random forest
"""
import numpy as np
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


class IncrementalClassifier:
    def __init__(self, **kwargs):
        # initialize with 'None' something to be loaded later
        self.img = None
        self.bb = None
        self.newlabels = None
        # inialize classifier as 'None'
        self.clf = None
        self.pca = None

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    # function for setting individual class parameters
    def set_param(self, **kwargs):
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

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

    def get_cross_val_score(self, kfold=5):
        X_train_pca = self.pca.transform(self.X_train_norm)
        X_train_all = np.append(X_train_pca, self.X_train_prop, axis=1)
        scores = cross_val_score(self.clf, X_train_all, self.y_train, cv=kfold)
        print scores

        return self

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
