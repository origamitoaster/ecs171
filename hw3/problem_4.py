#Q4
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt

def createSVM(x, y, name):
    y_true = []
    y_dec = []
    mean_fpr = np.linspace(0,1,100)
    #mean_rec = np.linspace(0,1,100)
    
    cv = KFold(n_splits=10)
    #classifier = OneVsRestClassifier(SVC(C=0.01, kernel='rbf', gamma='scale', random_state=0))
    classifier = OneVsRestClassifier(SVC(C=1, kernel='rbf', gamma='scale', random_state=0))
    #classifier = OneVsRestClassifier(LinearSVC(random_state=0, max_iter=10000))
    y = label_binarize(y, classes=np.unique(y))
    
    i = 0
    for train, test in cv.split(x, y):
        score = classifier.fit(x.iloc[train], y[train]).decision_function(x.iloc[test])
        fpr, tpr, _ = roc_curve(y[test].ravel(), score.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, alpha=0.2, label='%d fold (AUC: %0.2f)' % (i, roc_auc))
        i += 1

        y_true.append(y[test].ravel())
        y_dec.append(score.ravel())

    y_true = np.concatenate(y_true)
    y_dec = np.concatenate(y_dec)

    mean_fpr, mean_tpr, _ = roc_curve(y_true, y_dec)
    mean_roc_auc = auc(mean_fpr, mean_tpr)

    mean_prec, mean_rec, _ = precision_recall_curve(y_true, y_dec)
    mean_pr_auc = auc(mean_rec, mean_prec)
    #Plot
    #plt.figure()
    plt.plot(mean_fpr, mean_tpr, label='Mean AUC: %0.2f' % mean_roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve ' + '(' + name + ')')
    plt.legend(bbox_to_anchor=(1, 0), loc="lower right")
    plt.savefig('ROC_'+ name + '.png')
    plt.show()

    plt.figure()
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.plot([0, 1], [0.5, 0.5], 'k--')
    plt.plot(mean_rec, mean_prec, label='Mean AUC = {:.3f})'.format(mean_pr_auc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve' + '(' + name + ')')
    plt.legend(loc='best')
    plt.savefig('PR_' + name + '.png')
    plt.show()
    return (mean_fpr, mean_tpr, mean_rec, mean_prec)