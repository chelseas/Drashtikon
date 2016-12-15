# Learning Method: SVM
from scipy import *
from sklearn import svm
from sklearn import linear_model, decomposition
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
import numpy as np

# RBF SVMs
def rbfSVM(train_data, train_labels):
    model = svm.SVC(kernel='rbf')
    model.fit(train_data, train_labels)
    return model

def rbfSVMBalanced(train_data, train_labels):
    model = svm.SVC(kernel='rbf', class_weight='balanced')
    model.fit(train_data, train_labels)
    return model

# Linear SVMs
def linearSVM(train_data, train_labels):
    model = svm.SVC(kernel='linear')
    model.fit(train_data, train_labels)
    return model

def linearSVMBalanced(train_data, train_labels):
    model = svm.SVC(kernel='linear', class_weight='balanced')
    model.fit(train_data, train_labels)
    return model

# Logistic Regression
def logisticRegression(train_data, train_labels):
    model = linear_model.LogisticRegression()
    model.fit(train_data, train_labels)
    return model

def logisticRegressionBalanced(train_data, train_labels):
    model = linear_model.LogisticRegression(class_weight='balanced')
    model.fit(train_data, train_labels)
    return model

## Cross Validation Hyperparameter Setting


def rbfCVSVM(train_data, train_labels, test_data):
    # Model Initializer
    svc = svm.SVC()

    #Initialize transform Pipeline (PCA)
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('svc', svc)])
    N = len(train_data[0])
    n_components = [N/100, N/50,  N/22, N]
    n_components = [10]

    # Initialize range of SVM params
    C_range = np.logspace(-2, 10, 10)
    gamma_range = np.logspace(-9, 3, 10)
    class_weight_range = [None]

    # Intialize param grid for each type of classifier
    rbf_param_grid = dict(pca__n_components=n_components, svc__gamma=gamma_range, svc__C=C_range, svc__kernel=['rbf'], svc__class_weight=class_weight_range)

    # CV Params
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(pipe, param_grid=rbf_param_grid, cv=cv, n_jobs=-1, verbose=3)

    # Search for hyper paramters
    grid.fit(train_data, train_labels)
    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
    
    # Return a model trained on these parameters.
    best_pca = decomposition.PCA(n_components=grid.best_params_["pca__n_components"])
    best_pca.fit(train_data) 
    fit_train_data = best_pca.transform(train_data)
    fit_test_data = best_pca.transform(test_data) 
    model = svm.SVC(kernel=grid.best_params_["svc__kernel"], C=grid.best_params_["svc__C"], gamma=grid.best_params_["svc__gamma"], class_weight=grid.best_params_["svc__class_weight"])
    model.fit(fit_train_data, train_labels)
    return dict(model=model, test_data=fit_test_data, train_data=fit_train_data, params=str(grid.best_params_))

def linearCVSVM(train_data, train_labels, test_data):
    # Model Initializer
    svc = svm.LinearSVC()

    #Initialize transform Pipeline (PCA)
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('svc', svc)])
    N = len(train_data[0])
    n_components = [N/100, N/50,  N/22, N]

    # Initialize range of SVM params
    C_range = np.logspace(-2, 10, 10)
    class_weight_range = [None]

    # Intialize param grid for each type of classifier
    linear_param_grid = dict(pca__n_components=n_components, svc__C=C_range, svc__class_weight=class_weight_range)

    # CV Params
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(pipe, param_grid=linear_param_grid, cv=cv, n_jobs=-1, verbose=3)

    # Search for hyper paramters
    grid.fit(train_data, train_labels)
    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
    
    # Return a model trained on these parameters.
    best_pca = decomposition.PCA(n_components=grid.best_params_["pca__n_components"])
    best_pca.fit(train_data) 
    fit_train_data = best_pca.transform(train_data)
    fit_test_data = best_pca.transform(test_data) 
    model = svm.SVC(C=grid.best_params_["svc__C"], class_weight=grid.best_params_["svc__class_weight"])
    model.fit(fit_train_data, train_labels)
    return dict(model=model, test_data=fit_test_data, train_data=fit_train_data, params=str(grid.best_params_))

def CVLogisticRegression(train_data, train_labels, test_data):
    # Model Initializer
    logistic = linear_model.LogisticRegression()

    #Initialize transform Pipeline (PCA)
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    N = len(train_data[0])
    n_components = [N/100, N/50,  N/22, N]

    # Initialize range of SVM params
    C_range = np.logspace(1, 10, 10)
    class_weight_range = [None]

    # Intialize param grid for each type of classifier
    logistic_param_grid = dict(pca__n_components=n_components, logistic__C=C_range, logistic__class_weight=class_weight_range)

    # CV Params
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(pipe, param_grid=logistic_param_grid, cv=cv, n_jobs=-1, verbose=3)

    # Search for hyper paramters
    grid.fit(train_data, train_labels)
    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
    
    # Return a model trained on these parameters.
    best_pca = decomposition.PCA(n_components=grid.best_params_["pca__n_components"])
    best_pca.fit(train_data) 
    fit_train_data = best_pca.transform(train_data)
    fit_test_data = best_pca.transform(test_data) 
    model = linear_model.LogisticRegression(C=grid.best_params_["logistic__C"], class_weight=grid.best_params_["logistic__class_weight"])
    model.fit(fit_train_data, train_labels)
    return dict(model=model, test_data=fit_test_data, train_data=fit_train_data, params=str(grid.best_params_))



CV_MODELS = [rbfCVSVM, CVLogisticRegression, linearCVSVM]
MODELS = [rbfSVM, rbfSVMBalanced, linearSVM,  linearSVMBalanced, logisticRegression, logisticRegressionBalanced]
