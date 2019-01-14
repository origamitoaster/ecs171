#Problem 3
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt

    
def single_ols(dep, indep, order):
    #create mat with 1's in first col and indep in second col
    ones = np.ones(len(indep))
    indep = indep.to_frame()
    indep.insert(0, 'ones', value=ones)
    indep['y_sq'] = indep.iloc[:,1]**2
    indep['y_cu'] = indep.iloc[:,1]**3
    
    if order == 0:  
        indep = indep.drop(indep.columns[1:4], axis=1)
    elif order == 1:
        indep = indep.iloc[:,0:2]
    elif order == 2:
        indep = indep.iloc[:,0:3]
    elif order == 3:
        indep = indep.iloc[:,0:4]
        
     
    #weights = [w0, w1, w2]
    x_trans = np.transpose(indep)
    p1 = np.dot(x_trans, indep)
    p1 = pinv(p1)
    p2 = np.dot(x_trans, dep)
    weights = np.dot(p1, p2)
    
    return weights