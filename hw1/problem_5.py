#Problem 5
import numpy as np
from numpy.linalg import pinv
from sklearn.metrics import mean_squared_error as mse

#solve ols w/ all indep variables (w0+w1x1+w2x2+w3x3+..+w7x7+w8x1^2+w9x2^2+...+w15x7^2)
def full_ols(data, order):
    dep = data['mpg']
    indep = data.iloc[:,1:8]
    
    #create mat with 1's in first col and indep in second col
    ones = np.ones(len(indep))
    indep.insert(0, 'ones', value=ones)
    
    if order == 0:  
        indep = indep.drop(indep.columns[1:8], axis=1)
    elif order == 2:
        for i in range(1,8):
            indep[i+7] = indep.iloc[:,i]**2       
     
    x_trans = np.transpose(indep)
    p1 = np.dot(x_trans, indep)
    p1 = pinv(p1)
    p2 = np.dot(x_trans, dep)
    weights = np.dot(p1, p2)
    
    #returns weights depending on data and order given (1 weight for 0 order, 7 weights for 1st order, and 15 weights for 2nd order)
    return weights

def full_train(data):
    #assign data to var
    y = data['mpg']
    x1 = data['cyl']
    x2 = data['disp']
    x3 = data['hp']
    x4 = data['wt']
    x5 = data['accel']
    x6 = data['mdl_year']
    x7 = data['origin']
    
    #run ols on all indep var for each order
    t_0 = full_ols(data, 0)
    t_1 = full_ols(data, 1)
    t_2 = full_ols(data, 2)
    
    #calculate training mse for each order
    tr_mse_0 = mse(y, np.full((len(y),1),t_0[0]))
    tr_mse_1 = mse(y, t_1[0] + t_1[1]*x1 + t_1[2]*x2 + t_1[3]*x3 + t_1[4]*x4 + t_1[5]*x5 + t_1[6]*x6 + t_1[7]*x7)
    tr_mse_2 = mse(y, t_2[0] + t_2[1]*x1 + t_2[2]*x2 + t_2[3]*x3 + t_2[4]*x4 + t_2[5]*x5 + t_2[6]*x6 + t_2[7]*x7 + t_2[8]*(x1**2) + t_2[9]*(x2**2) + t_2[10]*(x3**2) + t_2[11]*(x4**2) + t_2[12]*(x5**2) + t_2[13]*(x6**2) + t_2[14]*(x7**2))
    
    #return array of weights for each order (t_0 is 0 order, etc) and training mse for each order (tr_mse_0 is for 0 order)
    return [t_0, t_1, t_2, tr_mse_0, tr_mse_1, tr_mse_2]

def full_test(data, sln):
    #assign data to var
    y = data['mpg']
    x1 = data['cyl']
    x2 = data['disp']
    x3 = data['hp']
    x4 = data['wt']
    x5 = data['accel']
    x6 = data['mdl_year']
    x7 = data['origin']

    sln_0 = sln[0]
    sln_1 = sln[1]
    sln_2 = sln[2]

    #predict mpg for each order using weights from trained model
    pred_0 = sln_0[0]
    pred_1 = sln_1[0] + sln_1[1]*x1 + sln_1[2]*x2 + sln_1[3]*x3 + sln_1[4]*x4 + sln_1[5]*x5 + sln_1[6]*x6 + sln_1[7]*x7
    pred_2 = sln_2[0] + sln_2[1]*x1 + sln_2[2]*x2 + sln_2[3]*x3 + sln_2[4]*x4 + sln_2[5]*x5 + sln_2[6]*x6 + sln_2[7]*x7 + sln_2[8]*(x1**2) + sln_2[9]*(x2**2) + sln_2[10]*(x3**2) + sln_2[11]*(x4**2) + sln_2[12]*(x5**2) + sln_2[13]*(x6**2) + sln_2[14]*(x7**2)
    
    #calculate testing mse for each order
    tt_mse_0 = mse(y,  np.full((len(y),1),sln_0[0]))
    tt_mse_1 = mse(y, pred_1)
    tt_mse_2 = mse(y, pred_2)

    #return array of training mse for each order (tt_mse_0 is for 0 order, etc) and predictions of mpg for each order (pred_0 is the prediction using 0 order)
    return [tt_mse_0, tt_mse_1, tt_mse_2, pred_0, pred_1, pred_2]