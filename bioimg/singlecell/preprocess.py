#!/usr/bin/env python3
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def glog_transform(df, eps=1e-8):
    '''Generalized log-transform
       -------------------------
       
       Parameters
       ---------
       df : DataFrame
           Input data frame
       eps : float (optional)
           Small constant added before log-transformation
           Default value: 1e-8

       Returns
       -------
       df_out : DataFrame
           Log-transformed features. The generalized log
           transformation is log((x + sqrt(x**2 + x.min()**2))/2 + eps)
           Parameter eps  is introduced to avoid log(0)
    '''
    return df.apply(lambda x: np.log((x + np.sqrt(x**2 + x.min()**2))/2 + eps))

def scale_data(df, scaler):
    '''Scale data columns
       ------------------
       Scale columns of a DataFrame by applying the
       sklearn.preprocessing scaler (e.g. StandardScaler,
       MinMaxScaler, etc). If `scaler=None`, mean and std
       are estimated for each column and the data is standardized
       using these values. It is, however, preferable to
       initialize and fit `scaler` object, so that the same
       standardization is applied within the plate or for
       both train and test sets for supervised learning

       Parameters
       ----------
       df : DataFrame
           Input DataFrame with morphological features in
           columns
       scaler : scaler object or None
           If `scaler=None`, (df - df.mean())/df.std() is
           returned. It is better to fit StandardScaler
           using e.g. plate control wells and apply the
           same scaling to all wells in the same plate


       Returns
       -------
       df_scaled : DataFrame
           Scaled DataFrame with all columns 
           approximately on the same scale
    '''
    if scaler is None:
        return (df-df.mean())/df.std()
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df_scaled

def check_data(df, return_indices=False):
    '''Check rows and columns of a DataFrame
       -------------------------------------
       The function prints the number of rows and columns
       with missing values (NaN or Inf)
       Parameters
       ----------
       df : DataFrame
           Input DataFrame to check for NaN or Inf values
       return_indices : bool (optional)
           If `True` return dictionary of na_rows and
           na_cols (row and column identifiers with 
           missing values). Default: False
    '''
    pd.set_option('use_inf_as_na', True)
    ncol_na = np.sum(df.isna().sum(axis=0) > 0)
    print("Number of columns with NaN/Inf values: %d" % ncol_na)
    nrow_na = np.sum(df.isna().sum(axis=1) > 0)
    print("Number of rows with NaN/Inf values: %d" % nrow_na)
    if return_indices:
        colind = df.columns[(df.isna().any(axis=0))]
        rowind = df.index[(df.isna().any(axis=1))]
        indices = dict(na_cols = colind, na_rows= rowind)
        return indices

def preprocess_data(df, sel=None, glog=True):
    '''Preprocess and transform data
       -----------------------------
       Run initial feature selection and/or
       transform the data using generalized log

       Parameters
       ----------
       df : DataFrame
           Input data with observations in rows and
           features in columns
       sel : feature selection method (optional)
           Variable selection method, e.g. VarianceThreshold
           that removes features with low variance
       glog : bool
           Transform the data using glog
       
       Returns
       -------
       df_out : DataFrame
           Transformed data
    '''
    if sel is not None:
        df = select_features(df, sel=sel)
    if glog:
        df = glog_transform(df)
    check_data(df)
    return df

def get_residuals(df, y):
    '''Compute linear model residuals
    '''
    X = sm.add_constant(df)
    lm = sm.OLS(y, X).fit()
    return lm.resid.values

def get_cor_residuals(rep1, rep2, sel, col):
    '''Correlation between replicate lm residuals
    '''
    resid1 = get_residuals(df=rep1[sel], y=rep1[col])
    resid2 = get_residuals(df=rep2[sel], y=rep2[col])
    return np.corrcoef(x=resid1, y=resid2)[0,1]

def select_residcor(prof1, prof2, sel):
    '''Iterative feature selection based on regression residuals
       ---------------------------------------------------------
       Provided the initial list of selected features (`sel`)
       fit linear models for all other features based on the
       selected set, compute residuals and at each step choose
       a feature with the highest replicate correlation between
       the residuals
       

       Parameters
       ----------
       prof1 : DataFrame
           Mean (or median) profile of replicate 1
       prof2 : DataFrame
           Mean (or median) profile of replicate 2
       sel : array like
           Initial list of features against which all
           other features are regressed

       Returns
       -------
       sel : array like
           List of selected features based on iterative
           regression residual correlation
    '''
    assert(prof1.shape == prof2.shape)
    all_feats = prof1.columns.values
    stop_criterion = 1
    while stop_criterion > 0.5:
        feats_to_check = np.setdiff1d(all_feats, sel)
        # correlations of residuals
        resid_cor = np.array([get_cor_residuals(rep1=prof1, rep2=prof2,
                          sel=sel, col=col) for col in feats_to_check])
        sel = sel + [feats_to_check[np.argmax(resid_cor)]]
        stop_criterion = np.sum(resid_cor > 0) / len(resid_cor)
    return sel

def select_uncorrelated(df, sel, cor_th=0.5):
    '''Select uncorrelated features
       ----------------------------
       Provided the initial list of selected features (`sel`)
       add iteratively uncorrelated features, i.e. those that have 
       maximum absolute correlation with any of the selected 
       features less than the correlation threshold (`cor_th`)
    
       Parameters
       ----------
       df : DataFrame
           Input data with features in columns
       sel : array like
           Initial list of selected features
       cor_th : float (optional)
           Correlation threshold

       Returns
       -------
       sel : array like
           List of featuers selected to be uncorrelated with
           the initially provided list
    '''
    cor_df = df.corr()
    
    all_feats = df.columns.values
    stop_criterion = 0
    while stop_criterion < cor_th:
        feats_to_check = np.setdiff1d(all_feats, sel)
        cand_cor = cor_df[sel]
        cand_cor = cand_cor[np.isin(cand_cor.index, feats_to_check)]
        max_cor = cand_cor.abs().max(axis=1)
        stop_criterion = np.min(max_cor)
        sel = sel + [cand_cor.index[np.argmin(max_cor)]]
    return sel

def aggregate_profiles(df, annot, how='mean',
                       by='well', as_index=False):
    '''Aggregate single-cell profiles
       ------------------------------

       Parameters
       ----------
       df : DataFrame
           Single-cell data (cells in rows and features
           in columns)
       annot : DataFrame
           Annotation data
       how : string
           Aggregation method passed to .groupby.agg()
           Default: 'mean'
       by : string
           Annotation level (column in annot) that
           groups cells. Default: 'well'
       as_index : bool
           Remove the grouping variable in the output
           DataFrame. Default: False

       Returns
       -------
       df_out : DataFrame with aggregated profiles
           By default, the grouping variable 'well'
           is kept in the `df_out`. Set `as_index=True`
           to return aggregated profiles without the
           grouping variable
    '''
    df[by] = annot[by]
    return df.groupby(by, as_index=as_index).agg(how)

# feature selection
def select_features(df, sel):
    '''Perform feature selection
       -------------------------
       Perform feature selection and return a
       DataFrame with the selected subset of features. 
       Any of the sklearn.feature_selection
       methods can be used such as SelectKBest,
       SelectFdr, VarianceThreshold, etc.
       For details see skimage.feature_selection documentation:
       https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

       Parameters
       ----------
       df : DataFrame
           Input DataFrame with features in columns and
           observations in rows
       sel : feature selector
           Primary feature selection methods that can be used
           are SelectKBest and SelectFdr. The user initializes
           sel object and passes it to the function (see Examples)

       Returns
       -------
       df_out : DataFrame
           DataFrame with subset features based on selection
           criteria

       Examples
       --------
       >>> from sklearn.feature_selection import SelectKBest, f_classif
       >>> sel = SelectKBest(f_classif, k=100)
       >>> # X, y are feature data and response vector, respectively
       >>> sel = sel.fit(X,y)
       >>> X_subset = select_features(df=X,sel=sel)
    '''
    df_out = sel.transform(df)
    return pd.DataFrame(df_out, columns=df.columns[sel.get_support()])

def recursive_elim(df, y, n_feat, elim_step=50, estim=None):
    '''Recursive feature elimination
       -----------------------------
       Perform recursive feature elimination using
       random forest classifier (or regression if y
       is continuous). At each iteration `n=elim_step`
       features are removed based on feature importance,
       until the number of features exceeds `n_feat`
       

       Parameters
       ----------
       df : DataFrame
           Input DataFrame with features in columns and
           observations in rows
       y : array-like
           Response vector, may be continuous or discrete
       n_feat : int
           Number of features to select
       elim_step : int
           Number of features to eliminate at each step (default=50)
       estim : estimator (optional)
           sklearn estimator. By default sklearn.ensemble.RandomForestClassifier
           or sklearn.ensemble.RandomForestRegressor. The user can initialize
           a different model (e.g. sklearn.svm.SVC) and pass as `estim` argument

       Returns
       -------
       rfe : sklearn.feature_selection.RFE object
           Recursive feature eliminate object fit to the data.
           The returned object should be used with `select_features` function
    '''
    if estim is None:
        if y.dtype == np.float:
            estim = RandomForestClassifier(n_estimators=500,
                                           max_depth=7,
                                           random_state=93,
                                           n_jobs=-1)
        else:
            estim = RandomForestClassifier(n_estimators=500,
                                 max_depth=7,
                                 random_state=3,
                                 n_jobs=-1)
    rfe = RFE(estimator=estim,
              n_features_to_select=n_feat, step=elim_step)
    rfe = rfe.fit(df, y)
    return rfe
