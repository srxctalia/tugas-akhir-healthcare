import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# load metode klasifikasi
from sklearn.neighbors import KNeighborsClassifier


#==================================Pemanggilan Dataset===========================================#

#filename_glass = 'data-set/sample9000dataset.csv'
filename_glass = 'data_train/all_train_hp.csv'
df_glass = pd.read_csv(filename_glass)


#=================================Split Data========================================#
def get_train_test(df, y_col, x_cols, ratio):
    mask = np.random.rand(len(df)) < ratio
    df_train = df[mask]
    df_test = df[~mask]
       
    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    return df_train, df_test, X_train, Y_train, X_test, Y_test

y_col_glass = 'label'
x_cols_glass = list(df_glass.columns.values)
x_cols_glass.remove(y_col_glass)

train_test_ratio = 0.6
df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df_glass, y_col_glass, x_cols_glass, train_test_ratio)

#=============================================================Metode Klasifikasi====================================================================#


dict_classifiers = {
    "Nearest Neighbors": KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, metric='chebyshev', metric_params=None, n_jobs=1),
}

def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 7, verbose = True):

    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        joblib.dump(classifier,"model.pkl") #simpen model
        t_end = time.clock()

        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)

        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        # if verbose:
            # print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))

        print (classifier_name)
        predicted = classifier.predict(X_test)
        report = classification_report(Y_test, predicted)
        print(Y_test)
        print(report)
        print("Actual", Y_test)
        print("Prediction", predicted)

        print (Y_test, predicted)

        cm = confusion_matrix(Y_test, predicted)
        # print(cm)
        TN = cm[1][1] * 1.0
        FN = cm[1][0] * 1.0
        TP = cm[0][0] * 1.0
        FP = cm[0][1] * 1.0

        total = TN + FN + TP + FP

        print('TP', TP, 'TN', TN, 'FP', FP, 'FN', FN)
        print('Total',total)

        acc = (TP+TN)/(total) * 100
        sens = TN/(TN+FP) * 100
        spec = TP/(TP+FN) * 100

        print ('Accuracy : '+ str(acc))
        print('Sensitivity : '+ str(sens))
        print('Specificity : '+ str(spec))

    return dict_models

def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]

    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]

    # print(df_.sort_values(by=sort_by, ascending=False))


dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 8, verbose=False)
display_dict_models(dict_models)
