#Q2
import numpy as np
from sklearn.utils import resample
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
#library for progress bar
#import pyprind

def Q2_bootstrap(df):
    # load dataset
    data = df.iloc[:,5:]
    mean_expr = df.iloc[:,6:].agg("mean", axis="rows")
    growth_mean = [df.iloc[:,5].mean()]

    values = data.values
    n_iterations = 100

    #bar = pyprind.ProgBar(n_iterations)


    # configure bootstrap
    n_size = int(len(data))
    # run bootstrap
    stats = list()
    for i in range(n_iterations):
        # prepare train and test sets
        train = resample(values, n_samples=n_size)
        #test = np.array([x for x in values if x.tolist() not in train.tolist()])
        # fit model
        #reg = Lasso(max_iter=10000, tol=0.01, random_state=0, alpha=q1_alpha)
        reg = Lasso(max_iter=1000, tol=0.01, random_state=0)
        reg.fit(train[:,1:], train[:,0])
        # evaluate model
        predictions = reg.predict(mean_expr.values.reshape(1,-1))
        mse = mean_squared_error(growth_mean, predictions)
        stats.append(mse)
        #bar.update()
    return stats

def conf_int(mse):
    alpha = 0.95
    p = ((1.0-alpha)/2.0)*100
    l = max(0.0, np.percentile(mse, p))
    p = (alpha+((1.0-alpha)/2.0))*100
    u = min(1.0, np.percentile(mse, p))
    print('With %.1f confidence, the MSE is likely to be between %.3f and %.3f' % (alpha*100, l, u))