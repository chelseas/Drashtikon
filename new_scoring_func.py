import sklearn.metrics as skm
from sklearn.preprocessing import label_binarize
import numpy as np
from scipy import interp

#def multi_roc(trained_estimator, X, y):
#    probas_ = trained_estimator.predict_proba(X)
#    fpr, tpr, thresholds = skm.roc_curve(y, probas_[:,1])
#    return skm.auc(fpr,tpr)

def multi_roc(trained_estimator, X, y):
    y = label_binarize(y, classes=[0, 1, 2, 3, 4])
    y_score = trained_estimator.decision_function(X)
    print("got inside scoring function")
    auc = skm.roc_auc_score(y, y_score, average='macro', sample_weight=None)
    return auc

def multi_roc2(trained_estimator, X, y):
    y = label_binarize(y, classes=[0, 1, 2, 3, 4])
    n_classes = y.shape[1]
    y_score = trained_estimator.decision_function(X)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = skm.roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = skm.auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = skm.auc(fpr["macro"], tpr["macro"])
    return roc_auc["macro"]
