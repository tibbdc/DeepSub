'''
Author: DengRui
Date: 2023-09-10 12:12:49
LastEditors: OBKoro1
LastEditTime: 2023-09-11 07:19:29
FilePath: /DeepSub/utlis.py
Description: 
Copyright (c) 2023 by DengRui, All Rights Reserved. 
'''

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn import linear_model
import pickle
import os
import joblib

def lrmain(X_train_std, Y_train, X_test_std, Y_test, type='binary'):
    '''
    description: 
    Args: 
    Returns: 
    '''      
    logreg = linear_model.LogisticRegression(
                                            # solver = 'saga',
                                            # multi_class='auto',
                                            verbose=False,
                                            n_jobs=-1,
                                            # max_iter=10000
                                        )
    logreg.fit(X_train_std, Y_train)
    predict = logreg.predict(X_test_std)
    lrpredpro = logreg.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, lrpredpro, logreg

def xgmain(X_train_std, Y_train, X_test_std, Y_test, type='binary', vali=True):

    x_train, x_vali, y_train, y_vali = train_test_split(X_train_std, Y_train, test_size=0.2, random_state=1)
    eval_set = [(x_train, y_train), (x_vali, y_vali)]

    if type=='binary':
        model = XGBClassifier(
                                objective='binary:logistic', 
                                random_state=15, 
                                use_label_encoder=False, 
                                n_jobs=-1, 
                                # eval_metric='mlogloss',
                                min_child_weight=15, 
                                max_depth=15, 
                                n_estimators=300
                            )
        model.fit(x_train, y_train.ravel(), eval_metric="logloss", eval_set=eval_set, verbose=False)
    if type=='multi':
        model = XGBClassifier(
                        min_child_weight=6, 
                        max_depth=6, 
                        objective='multi:softprob', 
                        num_class=len(set(Y_train)), 
                        use_label_encoder=False,
                        n_estimators=120
                    )
        if vali:
            model.fit(x_train, y_train,  eval_set=eval_set, verbose=False)
        else:
            model.fit(X_train_std, Y_train, eval_set=None, verbose=False)
    
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob, model


def knnmain(X_train_std, Y_train, X_test_std, Y_test, type='binary'):
    knn=KNeighborsClassifier(n_neighbors=5, n_jobs=16)
    knn.fit(X_train_std, Y_train)
    predict = knn.predict(X_test_std)
    lrpredpro = knn.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, lrpredpro, knn


def dtmain(X_train_std, Y_train, X_test_std, Y_test):
    model = tree.DecisionTreeClassifier()
    model.fit(X_train_std, Y_train.ravel())
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob,model


def rfmain(X_train_std, Y_train, X_test_std, Y_test):
    model = RandomForestClassifier(oob_score=True, random_state=10, n_jobs=-2)
    model.fit(X_train_std, Y_train.ravel())
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob,model


def gbdtmain(X_train_std, Y_train, X_test_std, Y_test):
    model = GradientBoostingClassifier(random_state=10)
    model.fit(X_train_std, Y_train.ravel())
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob, model


def svmmain(X_train_std, Y_train, X_test_std, Y_test):
    svcmodel = SVC(probability=True, kernel='rbf', tol=0.001)
    svcmodel.fit(X_train_std, Y_train.ravel(), sample_weight=None)
    predict = svcmodel.predict(X_test_std)
    predictprob =svcmodel.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob,svcmodel

def evauation_classification(groundtruth, predict, baselineName, type='binary', title=True):
    
    acc = metrics.accuracy_score(groundtruth, predict)
    if type == 'binary':
        precision = metrics.precision_score(groundtruth, predict, zero_division=True )
        recall = metrics.recall_score(groundtruth, predict,  zero_division=True)
        f1 = metrics.f1_score(groundtruth, predict, zero_division=True)
        tn, fp, fn, tp = metrics.confusion_matrix(groundtruth, predict).ravel()
        npv = tn/(fn+tn+1.4E-45)
        if title:
            print('{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}'.format('ITEM', 'ACC','Precision','NPV' ,'Recall', 'F1'))
        print('{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}tp:{}fp:{}tn:{}fn:{}'.format(baselineName, 
                                                                                round(acc,6), 
                                                                                round(precision,6), 
                                                                                round(npv,6), 
                                                                                round(recall,6), 
                                                                                round(f1,6)),
                                                                                tp,fp,tn,fn
                                                                                )
    
    if type == 'multi':
        precision = metrics.precision_score(groundtruth, predict, average='macro', zero_division=True )
        recall = metrics.recall_score(groundtruth, predict, average='macro', zero_division=True)
        f1 = metrics.f1_score(groundtruth, predict, average='macro', zero_division=True)
        if title:
            print('{:<20}{:<15}{:<15}{:<15}{:<15}'.format('ITEM', 'ACC','Precision', 'Recall', 'F1'))
        print('{:<20}{:<15}{:<15}{:<15}{:<15}'.format(baselineName, round(acc,6), round(precision,6), round(recall,6), round(f1,6)))
                     

def evaluate(baslineName, X_train_std, Y_train, X_test_std, Y_test, type='binary'):

    if baslineName == 'lr':
        groundtruth, predict, predictprob, model = lrmain (X_train_std, Y_train, X_test_std, Y_test, type=type)
    elif baslineName == 'svm':
        groundtruth, predict, predictprob, model = svmmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='xg':
        groundtruth, predict, predictprob, model = xgmain(X_train_std, Y_train, X_test_std, Y_test, type=type)
    elif baslineName =='dt':
        groundtruth, predict, predictprob, model = dtmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='rf':
        groundtruth, predict, predictprob, model = rfmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='gbdt':
        groundtruth, predict, predictprob, model = gbdtmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='knn':
        groundtruth, predict, predictprob, model = knnmain(X_train_std, Y_train, X_test_std, Y_test, type=type)
    else:
        print('error')
    print('*****DeepSub*****')
    
    # joblib.dump(model, '../model/baseline_model/'+baslineName+'.pkl')

    return predict, model
    
def run_baseline(X_train, Y_train, X_test, Y_test, type='binary',method='knn'):

    return evaluate(method, X_train, Y_train, X_test, Y_test, type=type)