#Problem 1
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

def toBoolean(outliers):
    #turns an array of 1's and -1's into True if 1 and False if -1
    out = pd.Series(True,index=range(len(outliers)))
    for i in range(len(outliers)):
        if outliers[i] == -1:
            out.iat[i] = False
    return(out)

def LOF(df):
    #runs LOF on DataFrame as a whole
    #takes in DataFrame and returns DataFrame without outliers
    clf = LocalOutlierFactor()
    outliers = clf.fit_predict(df.iloc[:,1:9])
    return(df[toBoolean(outliers)])

def iso_forest(df):
    #runs Isolation Forest on DataFrame as a whole
    #takes in DataFrame and returns DataFrame without outliers
    iso_f = IsolationForest()
    iso_f.fit(df.iloc[:,1:9])
    outliers = iso_f.predict(df.iloc[:,1:9])
    return(df[toBoolean(outliers)])