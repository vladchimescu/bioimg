#!/usr/bin/env python3

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE, SelectKBest, SelectFdr

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

# quality control for rows and columns
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

# feature selection
# also implement feature selection based on replicate correlation
