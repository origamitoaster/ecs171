#add current folder to path to use functions from other files
import os
import sys
#  csfp - current_script_folder_path
csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.insert(0, csfp)
# import it and invoke it by one of the ways described above

from problem_3 import single_ols
from problem_4 import train
from problem_4 import test
from problem_5 import full_train
from problem_5 import full_test
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#silence pandas warning
pd.options.mode.chained_assignment = None

#Read in data, add headers, fix datatypes, drop missing values
cars_df = pd.read_fwf('auto-mpg.data', header=None)
cars_df.columns = ["mpg", "cyl", "disp", "hp", "wt", "accel", "mdl_year", "origin", "car_name"]
cars_df = cars_df.replace(to_replace="?", value=np.nan)
cars = cars_df.dropna(axis=0)
cars["hp"] = cars["hp"].apply(pd.to_numeric, errors='ignore')

#Problem 1
#Sort mpg values and then assign low/med/high based on mpg boundaries found through looking at the array
cars = cars.sort_values("mpg") 
#cars[:131] #9 to 18.6
#cars[131:262] #19.0 to 27.0
#cars[262:] #27 to 46.6
cars['mpg_cat'] = pd.cut(cars['mpg'], [9, 18.6, 27 ,46.6], labels=['low', 'med', 'high'])

#Problem 2
#Create scatterplot matrix based on mpg categories
sns.set(style="ticks")
sns.pairplot(cars, hue="mpg_cat")
plt.savefig("scatter_mat.png")
plt.show()

#Problem 3
#Look at problem_3.py

#Problem 4
#"cyl", "disp", "hp", "wt", "accel", "mdl_year", "origin"
#I set random_state to 10 to create a random yet repeatable result to verify my functions were working properly.
#This value/constraint can be removed to create a more random set of data
car_train, car_test = train_test_split(cars, test_size=192, train_size=200, random_state=10, shuffle=True)

#twm stands for train_weight_mse
#tm stands for train_mse
#calculate test/train mse for each indep var and also plot the graphs
twm_cyl = train(car_train, 'cyl')
tm_cyl = test(car_test, 'cyl', twm_cyl)
print("Training MSE (cyl):", twm_cyl[4:8])
print("Testing MSE (cyl):", tm_cyl)

twm_disp = train(car_train, 'disp')
tm_disp = test(car_test, 'disp', twm_disp)
print("Training MSE (disp):", twm_disp[4:8])
print("Testing MSE (disp):", tm_disp)

twm_hp = train(car_train, 'hp')
tm_hp = test(car_test, 'hp', twm_hp)
print("Training MSE (hp):", twm_hp[4:8])
print("Testing MSE (hp):", tm_hp)

twm_wt = train(car_train, 'wt')
tm_wt = test(car_test, 'wt', twm_wt)
print("Training MSE (wt):", twm_wt[4:8])
print("Testing MSE (wt):", tm_wt)

twm_accel = train(car_train, 'accel')
tm_accel = test(car_test, 'accel', twm_accel)
print("Training MSE (accel):", twm_accel[4:8])
print("Testing MSE (accel):", tm_accel)

twm_mdl_year = train(car_train, 'mdl_year')
tm_mdl_year = test(car_test, 'mdl_year', twm_mdl_year)
print("Training MSE (mdl_year):", twm_mdl_year[4:8])
print("Testing MSE (mdl_year):", tm_mdl_year)

twm_origin = train(car_train, 'origin')
tm_origin = test(car_test, 'origin', twm_origin)
print("Training MSE (origin):", twm_origin[4:8])
print("Testing MSE (origin):", tm_origin)

#Problem 5
all_train = full_train(car_train)
all_test = full_test(car_test, all_train)
print("Training MSE (7 indep):", all_train[3:6])
print("Testing MSE (7 indep):", all_test[0:3])

#Problem 6
#use scikit-learn LogisticRegression model and precision_score function
log = linear_model.LogisticRegression()
log.fit(car_train.iloc[:,0:8], car_train['mpg_cat'])  

y_train =  car_train['mpg_cat']
y_p_train = log.predict(car_train.iloc[:,0:8])
y_train = y_train.astype('str')
y_p_train = y_p_train.astype('str')

train_prec = precision_score(y_train, y_p_train, labels=['low', 'med', 'high'], pos_label=None, average='micro')

y_test =  car_test['mpg_cat']
y_p_test = log.predict(car_test.iloc[:,0:8])
y_test = y_test.astype('str')
y_p_test = y_p_test.astype('str')

test_prec = precision_score(y_test, y_p_test, labels=['low', 'med', 'high'], pos_label=None, average='micro')

print("Training Precision:", train_prec)
print("Testing Precision:", test_prec)

#Problem 7
#We would have expected a low mpg
#Create new dataframe to hold the single car
new_car = np.array([0.0, 6, 350.0, 180.0, 3700.0, 9.0, 80, 1, "new_car"])
n_car = pd.DataFrame(new_car.reshape(-1, len(new_car)))
n_car.columns= ["mpg", "cyl", "disp", "hp", "wt", "accel", "mdl_year", "origin", "car_name"]
n_car["mpg"] = n_car["mpg"].apply(pd.to_numeric)
n_car["cyl"] = n_car["cyl"].apply(pd.to_numeric)
n_car["disp"] = n_car["disp"].apply(pd.to_numeric)
n_car["hp"] = n_car["hp"].apply(pd.to_numeric)
n_car["wt"] = n_car["wt"].apply(pd.to_numeric)
n_car["accel"] = n_car["accel"].apply(pd.to_numeric)
n_car["mdl_year"] = n_car["mdl_year"].apply(pd.to_numeric)
n_car["origin"] = n_car["origin"].apply(pd.to_numeric)

#Predict the mpg and the category based on previously trained model
n_car_test = full_test(n_car, all_train)
n_car.iloc[0,0] = n_car_test[5].iloc[0]
n_car_pred = log.predict(n_car.iloc[:,0:8])
print("Predicted MPG:", n_car['mpg'].to_string(index=False), " Predicted mpg category:", n_car_pred)