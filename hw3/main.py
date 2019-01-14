#add current folder to path to use functions from other files
import os
import sys
#  csfp - current_script_folder_path
csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.insert(0, csfp)
# import it and invoke it by one of the ways described above

#Ignore sklearn warnings about class imbalance
import warnings
warnings.simplefilter("ignore", UserWarning)

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
#Q1
from problem_1 import Q1_fit
#Q2
from problem_2 import Q2_bootstrap
from problem_2 import conf_int
#Q4
from problem_4 import createSVM
#Q5
from problem_5 import createCompositeSVM
from scipy import interp
from sklearn.metrics import auc
#Q6
from problem_6 import fitPCA


def load_data():
    df = pd.read_csv('ecs171.dataset.txt', delim_whitespace=True)
    return df

#Load Dataset
df = load_data()

#Question 1
Q1_CV = Q1_fit(df)

#Question 3
mses = Q2_bootstrap(df)
conf_int(mses)

#Old Question 3
#mean_expr = df.iloc[:,6:].agg("mean", axis="rows")
#print('Predicted Mean Growth: ', Q1_CV.predict(mean_expr.values.reshape(1,-1)))
#print('Actual Mean Growth: ', df.iloc[:,5].mean())

#Question 4
genes = df.iloc[:,6:]
genes = genes[genes.columns[Q1_CV.coef_ != 0]]

#Strain
strain = createSVM(genes, df['Strain'], 'Strain')
#Medium
med = createSVM(genes, df['Medium'], 'Medium')
#Stress
stress = createSVM(genes, df['Stress'], 'Stress')
#GenePerturbed
geneP = createSVM(genes, df['GenePerturbed'], 'GenePerturbed')

#Question 5
#mean_fpr, mean_tpr, mean_rec, mean_prec
def compare():
    #Plot ROC
    x = np.linspace(0,1,1000)
    med_i = interp(x, med[0], med[1])
    geneP_i = interp(x, geneP[0], geneP[1])
    combo_roc_auc = auc(x, (med_i+geneP_i)/2.0)
    plt.plot(x, (med_i+geneP_i)/2.0, label='Combined AUC: %0.2f' % combo_roc_auc)
    #Train Composite SVM
    composite = createCompositeSVM(genes, df[['Medium', 'GenePerturbed']], 'Composite')
    #Plot PR
    x_med = np.flip(med[2])
    x_geneP = np.flip(geneP[2])
    y_med = np.flip(med[3])
    y_geneP =np.flip(geneP[3])
    pr_med = interp(x, x_med, y_med)
    pr_geneP = interp(x, x_geneP, y_geneP)
    combo_pr_auc = auc(x, (pr_med+pr_geneP)/2.0)
    plt.plot(x, (pr_med+pr_geneP)/2.0, label='Combined AUC: %0.2f' % combo_pr_auc)
    plt.legend(loc='best')
    plt.savefig('PR_' + 'combo' + '.png')
    plt.show()
compare()

#Question 6
full_genes = df.iloc[:,6:]
pca_genes = pd.DataFrame(fitPCA(full_genes))
#Strain
pca_strain = createSVM(pca_genes, df['Strain'], 'Strain_PCA')
#Medium
pca_med = createSVM(pca_genes, df['Medium'], 'Medium_PCA')
#Stress
pca_stress = createSVM(pca_genes, df['Stress'], 'Stress_PCA')
#GenePerturbed
pca_geneP = createSVM(pca_genes, df['GenePerturbed'], 'GenePerturbed_PCA')