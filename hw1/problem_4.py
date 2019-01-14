#Problem 4
from problem_3 import single_ols
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse

def train(data, indep):
    y = data['mpg']
    x = data[indep]

    #run ols solver for each order
    t_0 = single_ols(y, x, 0)
    t_1 = single_ols(y, x, 1)
    t_2 = single_ols(y, x, 2)
    t_3 = single_ols(y, x, 3)
    
    #calculate training mse for each order
    tr_mse_0 = mse(y, np.full((len(x),1),t_0[0]))
    tr_mse_1 = mse(y, t_1[0] + t_1[1]*x)
    tr_mse_2 = mse(y, t_2[0] + (t_2[1]*x) + (t_2[2]*(x**2)))
    tr_mse_3 = mse(y, t_3[0] + (t_3[1]*x) + (t_3[2]*(x**2)) + (t_3[3]*(x**3)))

    #returns array with values of weights for each order (t_0 is 0 order, etc) and the training mse for each order (tr_mse_0 is for 0 order)
    return [t_0, t_1, t_2, t_3, tr_mse_0, tr_mse_1, tr_mse_2, tr_mse_3]

def test(data, indep, sln):
    y = data['mpg']
    x = data[indep]
    sln_0 = sln[0]
    sln_1 = sln[1]
    sln_2 = sln[2]
    sln_3 = sln[3]
    
    #calculate testing mse for each order
    tt_mse_0 = mse(y,  np.full((len(x),1),sln_0[0]))
    tt_mse_1 = mse(y, sln_1[0] + sln_1[1]*x)
    tt_mse_2 = mse(y, sln_2[0] + (sln_2[1]*x) + (sln_2[2]*(x**2)))
    tt_mse_3 = mse(y, sln_3[0] + (sln_3[1]*x) + (sln_3[2]*(x**2)) + (sln_3[3]*(x**3)))

    #plot scatterplot
    sns.scatterplot(data=data, x=indep, y='mpg')
    x = x.sort_values()
    
    #plot 0 order
    plt.hlines(sln_0[0], xmin=min(x), xmax=max(x))
    #plot 1st order
    plt.plot(x, sln_1[0] + sln_1[1]*x)
    #plot 2nd order
    sq = sln_2[0] + (sln_2[1]*x) + (sln_2[2]*(x**2))
    plt.plot(x, sq)
    #plot 3rd order
    cu = sln_3[0] + (sln_3[1]*x) + (sln_3[2]*(x**2)) + (sln_3[3]*(x**3))
    plt.plot(x, cu)

    plt.savefig(indep + ".png")
    plt.show()
    
    return [tt_mse_0, tt_mse_1, tt_mse_2, tt_mse_3]
    
