#Q1
import numpy as np
from sklearn.linear_model import LassoCV

def Q1_fit(data):
    genes = data.iloc[:,6:]
    growth = data.iloc[:,5]
    reg = LassoCV(cv=10, max_iter=300, tol=0.01, random_state=0,verbose=True, n_jobs=-1)
    reg.fit(genes, growth)
    #Find best lambda
    print('10-fold CV lambda: ', reg.alpha_)
    #Report non-zero coeffs
    print('Number of non-zero coefficients: ', np.count_nonzero(reg.coef_))
    #Find mean MSE error for last run
    print('Generalization Error: ', np.mean(reg.mse_path_[99]))
    return reg
