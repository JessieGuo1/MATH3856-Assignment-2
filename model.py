from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA

import xgboost as xgb
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.utils import set_random_seed

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score 
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz
from sklearn.tree import export_text

from subprocess import call

import numpy as np

import pandas as pd
from pandas import DataFrame

def process_data(df):
    values = np.array(df.iloc[:, 0])
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    onehot_encoded = pd.DataFrame(onehot_encoded, columns = ["F", "I", "M"])
    print(onehot_encoded)
    df = df.drop(columns = "sex")
    df = pd.concat([onehot_encoded, df], axis=1)
    df = np.array(df)
    df[:, -1] = np.where(df[:, -1] <= 7, 0, np.where(df[:, -1] <= 10, 1, np.where(df[:, -1] <= 15, 2, 3)))
    return df 
    
def distribution(df):
    for i in np.unique(df[:, -1]):
        clas = df[df[:, -1] == i]

        plt.hist(clas[:, 1+2], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85, density = True)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel(f'Length for {int(i)}')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(f'Length for {int(i)}.png')
        plt.clf()

        plt.hist(clas[:, 2+2], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85, density = True)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel(f'Diameter for {int(i)}')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(f'Diameter for {int(i)}.png')
        plt.clf()

        plt.hist(clas[:, 3+2], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85, density = True)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel(f'Height for {int(i)}')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(f'Height for {int(i)}.png')
        plt.clf()

def pca(df):
    
    X = df[:, 0:-1]
    y = df[:, -1]

    pca = PCA(n_components = 2)
    X_reduced = pca.fit_transform(X)

    colors = ['blue', 'red', 'green', 'darkorange']
    labels = ['0-7 years', '8-10 years', '11-15 years', '> 15 years']
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2, 3], labels):
        plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1],
                    color=color, lw=2, label=target_name)
    plt.title("PCA of Abalone Dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.savefig("PCA of Abalone Dataset.png")
    plt.clf()

def prepare(df, i):
    X = df[:, 0:-1]
    y = df[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = i)
    
    transformer = StandardScaler()
    x_train[:, 3:] = transformer.fit_transform(x_train[:, 3:])
    x_test[:, 3:] = transformer.transform(x_test[:, 3:])

    return x_train, x_test, y_train, y_test

def decision(x_train, x_test, y_train, y_test, depth):

    model = DecisionTreeClassifier(random_state = 0, max_depth = depth)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    acc = accuracy_score(y_test, y_pred)
    acc_train = accuracy_score(y_train, y_pred_train)

    f1 = f1_score(y_test, y_pred, average = 'weighted')
    f1_train = f1_score(y_train, y_pred_train, average = 'weighted') 
    
    return acc, acc_train, f1, f1_train
    
def postprun(x_train, x_test, y_train, y_test):

    clf = DecisionTreeClassifier(random_state=0, max_depth = 6)
    path = clf.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities #alphas, total leaf impurities at each step
    
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, max_depth = 6, ccp_alpha=ccp_alpha)
        clf.fit(x_train, y_train)
        clfs.append(clf)
    
    train_scores = [clf.score(x_train, y_train) for clf in clfs] #training accuracy of model at the specified alpha
    test_scores = [clf.score(x_test, y_test) for clf in clfs]
    
    f1train_scores = []
    f1_scores = []
    for clf in clfs:
        y_pred = clf.predict(x_test)
        y_pred_train = clf.predict(x_train)
        f1 = f1_score(y_test, y_pred, average = 'weighted')
        f1_train = f1_score(y_train, y_pred_train, average = 'weighted') 
 
        f1train_scores.append(f1_train)
        f1_scores.append(f1)

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()
    plt.savefig("Accuracy vs alpha.png")

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("f1 score")
    ax.set_title("F1 Score vs alpha for training and testing sets")
    ax.plot(ccp_alphas, f1train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, f1_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()
    plt.savefig("F1 Score vs alpha.png")

    return ccp_alphas, test_scores, f1_scores

def randfor(x_train, x_test, y_train, y_test, trees):
    model = RandomForestClassifier(n_estimators=trees, random_state = 0)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    accr = accuracy_score(y_test, y_pred)
    acc_trainr = accuracy_score(y_train, y_pred_train)

    f1 = f1_score(y_test, y_pred, average = 'weighted')
    f1_train = f1_score(y_train, y_pred_train, average = 'weighted') 

    return accr, acc_trainr, f1, f1_train

def boost(x_train, x_test, y_train, y_test, xggrad, trees):
    
    if xggrad == 'Gradient':
        model = GradientBoostingClassifier(n_estimators=trees, random_state = 0)

    elif xggrad == 'XG':
        model = xgb.XGBClassifier(n_estimators=trees, random_state = 0)

    model.fit(x_train, y_train)    
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    accb = accuracy_score(y_test, y_pred)
    acc_trainb = accuracy_score(y_train, y_pred_train)

    f1 = f1_score(y_test, y_pred, average = 'weighted')
    f1_train = f1_score(y_train, y_pred_train, average = 'weighted') 

    return accb, acc_trainb, f1, f1_train

def pca2(df, var, i):
    
    if var == 0.95:
        pca = PCA(n_components=0.95)
    elif var == 0.98:
        pca = PCA(n_components=0.98)
    
    X = df[:, 0:-1]
    y = df[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = i)
    pca.fit(x_train)
    x_reduced_train = pca.transform(x_train)
    x_reduced_test = pca.transform(x_test)
    
    transformer = StandardScaler()
    x_reduced_train = transformer.fit_transform(x_reduced_train)
    x_reduced_test = transformer.transform(x_reduced_test)

    model = GradientBoostingClassifier(n_estimators=100, random_state = 0)
    model.fit(x_reduced_train, y_train)    
    y_pred = model.predict(x_reduced_test)
    y_pred_train = model.predict(x_reduced_train)

    acc = accuracy_score(y_test, y_pred)
    acc_train = accuracy_score(y_train, y_pred_train)

    f1 = f1_score(y_test, y_pred, average = 'weighted')
    f1_train = f1_score(y_train, y_pred_train, average = 'weighted') 

    return acc, acc_train, f1, f1_train

def neural(x_train, x_test, y_train, y_test, sol):
    if sol == 'Adam':
        model = MLPClassifier(random_state = 0, solver = 'adam')
    elif sol == 'SGD':
        model = MLPClassifier(random_state = 0, solver= 'sgd')

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    accnn = accuracy_score(y_test, y_pred)
    acc_trainnn = accuracy_score(y_train, y_pred_train)

    f1 = f1_score(y_test, y_pred, average = 'weighted')
    f1_train = f1_score(y_train, y_pred_train, average = 'weighted') 

    return accnn, acc_trainnn, f1, f1_train

def reg(x_train, x_test, y_train, y_test):
    set_random_seed(0)
    model = Sequential()
    model.add(Dense(100, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(4, activation = 'softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, verbose=0)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    _, acc_train = model.evaluate(x_train, y_train, verbose=1)
    
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred_train = model.predict(x_train)
    y_pred_train = np.argmax(y_pred_train, axis=1)
    
    f1 = f1_score(y_test, y_pred, average = 'weighted')
    f1_train = f1_score(y_train, y_pred_train, average = 'weighted')
    K.clear_session()

    return acc, acc_train, f1, f1_train

def regul(x_train, x_test, y_train, y_test, weight):
    set_random_seed(0)
    model = Sequential()
    model.add(Dense(100, input_dim=x_train.shape[1], activation='relu', kernel_regularizer=l2(weight)))
    model.add(Dense(4, activation = 'softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, verbose=0)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    _, acc_train = model.evaluate(x_train, y_train, verbose=1)
    
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred_train = model.predict(x_train)
    y_pred_train = np.argmax(y_pred_train, axis=1)

    f1 = f1_score(y_test, y_pred, average = 'weighted')
    f1_train = f1_score(y_train, y_pred_train, average = 'weighted')
    K.clear_session()

    return acc, acc_train, f1, f1_train

def main():
    df = pd.read_csv('abalone.data', header=None, names=["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"])
    df = process_data(df)
    print(df)

    distribution(df)

    pca(df)

    #DecisionTree: accuracy, f1score for each maxdepth
    accmean = []
    accstd = []
    acc_trainmean = []
    acc_trainstd = []

    f1mean = []
    f1std = []
    f1_trainmean = []
    f1_trainstd = []
    for depth in range(1, 11):
        acc_all = np.zeros(10)
        acc_train_all = np.zeros(10)
        f1_all = np.zeros(10)
        f1_train_all = np.zeros(10)
        for i in range(10):
            x_train, x_test, y_train, y_test = prepare(df, i)
            acc, acc_train, f1, f1_train = decision(x_train, x_test, y_train, y_test, depth)        
            acc_all[i] = acc
            acc_train_all[i] = acc_train
            f1_all[i] = f1
            f1_train_all[i] = f1_train

        accmean.append(round(acc_all.mean(), 4)), accstd.append(round(acc_all.std(), 4)) #accuracy for each minleaf
        acc_trainmean.append(round(acc_train_all.mean(), 4)), acc_trainstd.append(round(acc_train_all.std(), 4))
        f1mean.append(round(f1_all.mean(), 4)), f1std.append(round(f1_all.std(), 4)) 
        f1_trainmean.append(round(f1_train_all.mean(), 4)), f1_trainstd.append(round(f1_train_all.std(), 4))
    print(accmean, 'Decision Tree: Accuracy Mean') 
    print(acc_trainmean, 'Decision Tree: Training Accuracy Mean') 
    print(accstd, 'Decision Tree: Accuracy Standard Deviation (SD)')
    print(acc_trainstd, 'Decision Tree: Training Accuracy SD')
    print(f1mean, 'Decision Tree: F1 Mean') 
    print(f1_trainmean, 'Decision Tree: F1 Training Mean') 
    print(f1std, 'Decision Tree: F1 SD')
    print(f1_trainstd, 'Decision Tree: F1 Training SD')

    index = np.argmax(accmean)
    print(index, max(accmean), accstd[index], acc_trainmean[index], acc_trainstd[index], 'Maximum Accuracies')
    index2 = np.argmax(f1mean)
    print(index2, max(f1mean), f1std[index2], f1_trainmean[index2], f1_trainstd[index2], 'Maximum F1 Scores')

    x_train, x_test, y_train, y_test = prepare(df, 10)
    best_tree = DecisionTreeClassifier(random_state = 0, max_depth = index + 1)
    best_tree.fit(x_train, y_train)

    export_graphviz(best_tree, out_file='tree.dot', 
                    feature_names = ("F", "I", "M", "Length", "Diameter", "Height", "Whole weight", "Schucked weight", "Viscera weight", "Shell weight"),
                    class_names = ("0-7 years", "8-10 years", "11-15 years", ">15 years"),
                    rounded = True, proportion = False, 
                    precision = 3, filled = True)

    call(['dot', '-Tpdf', 'tree.dot', '-o', 'best_tree_acc.pdf', '-Gdpi=600'])
    visual = export_text(best_tree, feature_names=("F", "I", "M", "Length", "Diameter", "Height", "Whole weight", "Schucked weight", "Viscera weight", "Shell weight"))
    print(visual, 'based on Accuracy')

    best_tree = DecisionTreeClassifier(random_state = 0, max_depth = index2 + 1)
    best_tree.fit(x_train, y_train)
    export_graphviz(best_tree, out_file='tree.dot', 
                    feature_names = ("F", "I", "M", "Length", "Diameter", "Height", "Whole weight", "Schucked weight", "Viscera weight", "Shell weight"),
                    class_names = ("0-7 years", "8-10 years", "11-15 years", ">15 years"),
                    rounded = True, proportion = False, 
                    precision = 3, filled = True)
    call(['dot', '-Tpdf', 'tree.dot', '-o', 'best_tree_F1.pdf', '-Gdpi=600'])
    visual = export_text(best_tree, feature_names=("F", "I", "M", "Length", "Diameter", "Height", "Whole weight", "Schucked weight", "Viscera weight", "Shell weight"))
    print(visual, 'based on F1 Score')

    #Postpruning
    x_train, x_test, y_train, y_test = prepare(df, 10)
    ccp_alphas, test_scores, f1_scores = postprun(x_train, x_test, y_train, y_test)
    index = np.argmax(test_scores)
    index2 = np.argmax(f1_scores) 
    optimal_ccp = ccp_alphas[index]
    optimal_ccp2 = ccp_alphas[index2]

    print(round(optimal_ccp, 5), 'Optimal ccp_alpha based on Accuracy')
    print(round(optimal_ccp2, 5), 'Optimal ccp_alpha based on F1 Score')

    acc_all = np.zeros(10)
    acc_train_all = np.zeros(10)
    f1_all = np.zeros(10)
    f1_train_all = np.zeros(10)
    for i in range(10):
        x_train, x_test, y_train, y_test = prepare(df, i)
        model = DecisionTreeClassifier(random_state = 0, max_depth = 6, ccp_alpha = 0.00133)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_pred_train = model.predict(x_train)

        acc = accuracy_score(y_test, y_pred)
        acc_train = accuracy_score(y_train, y_pred_train)

        f1 = f1_score(y_test, y_pred, average = 'weighted')
        f1_train = f1_score(y_train, y_pred_train, average = 'weighted')        
            
        acc_all[i] = acc
        acc_train_all[i] = acc_train
        f1_all[i] = f1
        f1_train_all[i] = f1_train
    
    print(round(acc_all.mean(), 4), 'Postpruned Decision Tree: Accuracy Mean')
    print(round(acc_all.std(), 4), 'Postpruned Decision Tree: Accuracy SD')
    print(round(acc_train_all.mean(), 4), 'Postpruned Decision Tree: Training Accuracy Mean')
    print(round(acc_train_all.std(), 4), 'Postpruned Decision Tree: Training Accuracy SD')
    print(round(f1_all.mean(), 4), 'Postpruned Decision Tree: F1 Mean')
    print(round(f1_all.std(), 4), 'Postpruned Decision Tree: F1 SD') 
    print(round(f1_train_all.mean(), 4), 'Postpruned Decision Tree: Training F1 Mean')
    print(round(f1_train_all.std(), 4), 'Postpruned Decision Tree: Training F1 SD')

    #RandomForest
    treelist = [100, 110, 120, 130, 140, 150]
    accmean = []
    accstd = []
    acc_trainmean = []
    acc_trainstd = []
    f1mean = []
    f1std = []
    f1_trainmean = []
    f1_trainstd = []
    for trees in treelist:
        acc_all = np.zeros(10)
        acc_train_all = np.zeros(10)
        f1_all = np.zeros(10)
        f1_train_all = np.zeros(10)
        for j in range(10):
            x_train, x_test, y_train, y_test = prepare(df, j)
            acc, acc_train, f1, f1_train = randfor(x_train, x_test, y_train, y_test, trees)
            acc_all[j] = acc
            acc_train_all[j] = acc_train
            f1_all[j] = f1
            f1_train_all[j] = f1_train 

        accmean.append(round(acc_all.mean(), 4)), accstd.append(round(acc_all.std(), 4))
        acc_trainmean.append(round(acc_train_all.mean(), 4)), acc_trainstd.append(round(acc_train_all.std(), 4))
        f1mean.append(round(f1_all.mean(), 4)), f1std.append(round(f1_all.std(), 4)) 
        f1_trainmean.append(round(f1_train_all.mean(), 4)), f1_trainstd.append(round(f1_train_all.std(), 4))
    print(accmean, 'Random Forest: Accuracy Mean') 
    print(acc_trainmean, 'Random Forest: Training Accuracy Mean') 
    print(accstd, 'Random Forest: Accuracy SD')
    print(acc_trainstd, 'Random Forest: Training Accuracy SD')
    print(f1mean, 'Random Forest: F1 Mean') 
    print(f1_trainmean, 'Random Forest: Training F1 Mean') 
    print(f1std, 'Random Forest: F1 SD')
    print(f1_trainstd, 'Random Forest: Training F1 SD')

    index = np.argmax(accmean)
    print(index, max(accmean), accstd[index], acc_trainmean[index], acc_trainstd[index], 'Maximum Accuracies')
    index2 = np.argmax(f1mean)
    print(index2, max(f1mean), f1std[index2], f1_trainmean[index2], f1_trainstd[index2], 'Maximum F1 Scores')

    #Boosting
    treelist = [100, 110, 120, 130, 140, 150]
    accmean = []
    accstd = []
    acc_trainmean = []
    acc_trainstd = []
    f1mean = []
    f1std = []
    f1_trainmean = []
    f1_trainstd = []

    accmeanx = []
    accstdx = []
    acc_trainmeanx = []
    acc_trainstdx = []
    f1meanx = []
    f1stdx = []
    f1_trainmeanx = []
    f1_trainstdx = []
    for trees in treelist:
        acc_all = np.zeros(10)
        acc_train_all = np.zeros(10)
        acc_allx = np.zeros(10)
        acc_train_allx = np.zeros(10)
        f1_all = np.zeros(10)
        f1_train_all = np.zeros(10)
        f1_allx = np.zeros(10)
        f1_train_allx = np.zeros(10)        
        for j in range(10):
            x_train, x_test, y_train, y_test = prepare(df, j)
            acc, acc_train, f1, f1_train = boost(x_train, x_test, y_train, y_test, 'Gradient', trees)
            acc_all[j] = acc
            acc_train_all[j] = acc_train
            f1_all[j] = f1
            f1_train_all[j] = f1_train

            accx, acc_trainx, f1x, f1_trainx = boost(x_train, x_test, y_train, y_test, 'XG', trees)
            acc_allx[j] = accx
            acc_train_allx[j] = acc_trainx
            f1_allx[j] = f1x
            f1_train_allx[j] = f1_trainx
        accmean.append(round(acc_all.mean(), 4)), accstd.append(round(acc_all.std(), 4))
        acc_trainmean.append(round(acc_train_all.mean(), 4)), acc_trainstd.append(round(acc_train_all.std(), 4))
        accmeanx.append(round(acc_allx.mean(), 4)), accstdx.append(round(acc_allx.std(), 4))
        acc_trainmeanx.append(round(acc_train_allx.mean(), 4)), acc_trainstdx.append(round(acc_train_allx.std(), 4))
        f1mean.append(round(f1_all.mean(), 4)), f1std.append(round(f1_all.std(), 4)) 
        f1_trainmean.append(round(f1_train_all.mean(), 4)), f1_trainstd.append(round(f1_train_all.std(), 4))
        f1meanx.append(round(f1_allx.mean(), 4)), f1stdx.append(round(f1_allx.std(), 4)) 
        f1_trainmeanx.append(round(f1_train_allx.mean(), 4)), f1_trainstdx.append(round(f1_train_allx.std(), 4))
 
    print(accmean, 'Gradient Boosting: Accuracy Mean') 
    print(acc_trainmean, 'Gradient Boosting: Training Accuracy Mean') 
    print(accstd, 'Gradient Boosting: Accuracy SD')
    print(acc_trainstd, 'Gradient Boosting: Training Accuracy SD')
    print(f1mean, 'Gradient Boosting: F1 Mean') 
    print(f1_trainmean, 'Gradient Boosting: Training F1 Mean') 
    print(f1std, 'Gradient Boosting: F1 SD')
    print(f1_trainstd, 'Gradient Boosting: Training F1 SD')   

    print(accmeanx, 'XGBoosting: Accuracy Mean') 
    print(acc_trainmeanx, 'XGBoosting: Training Accuracy Mean') 
    print(accstdx, 'XGBoosting: Accuracy SD')
    print(acc_trainstdx, 'XGBoosting: Training Accuracy SD')
    print(f1meanx, 'XGBoosting: F1 Mean') 
    print(f1_trainmeanx, 'XGBoosting: Training F1 Mean') 
    print(f1stdx, 'XGBoosting: F1 SD')
    print(f1_trainstdx, 'XGBoosting: Training F1 SD')

    index = np.argmax(accmean)
    print(index, max(accmean), accstd[index], acc_trainmean[index], acc_trainstd[index], 'Gradient Boosting Maximum Accuracies')
    index2 = np.argmax(f1mean)
    print(index2, max(f1mean), f1std[index2], f1_trainmean[index2], f1_trainstd[index2], 'Gradient Boosting Maximum F1 Scores')
    indexx = np.argmax(accmeanx)
    print(indexx, max(accmeanx), accstdx[index], acc_trainmeanx[index], acc_trainstdx[index], 'XGBoosting Maximum Accuracies')
    index2x = np.argmax(f1meanx)
    print(index2x, max(f1meanx), f1stdx[index2x], f1_trainmeanx[index2x], f1_trainstdx[index2x], 'XGBoosting Maximum F1 Scores')

    #PCA
    acc_all = np.zeros(10)
    acc_train_all = np.zeros(10)
    acc_all2 = np.zeros(10)
    acc_train_all2 = np.zeros(10)
    f1_all = np.zeros(10)
    f1_train_all = np.zeros(10)
    f1_all2 = np.zeros(10)
    f1_train_all2 = np.zeros(10)  

    for i in range(10):

        acc, acc_train, f1, f1_train = pca2(df, 0.95, i)
        acc_all[i] = acc
        acc_train_all[i] = acc_train
        f1_all[i] = f1
        f1_train_all[i] = f1_train

        acc2, acc_train2, f12, f1_train2 = pca2(df, 0.98, i)
        acc_all2[i] = acc2
        acc_train_all2[i] = acc_train2
        f1_all2[i] = f12
        f1_train_all2[i] = f1_train2

    print(round(acc_all.mean(), 4), 'PCA95: Accuracy Mean') 
    print(round(acc_train_all.mean(), 4), 'PCA95: Training Accuracy Mean') 
    print(round(acc_all.std(), 4), 'PCA95: Accuracy SD')
    print(round(acc_train_all.std(), 4), 'PCA95: Training Accuracy SD')
    print(round(f1_all.mean(), 4), 'PCA95: F1 Mean') 
    print(round(f1_train_all.mean(), 4), 'PCA95: Training F1 Mean') 
    print(round(f1_all.std(), 4), 'PCA95: F1 SD')
    print(round(f1_train_all.std(), 4), 'PCA95: Training F1 SD')
        
    print(round(acc_all2.mean(), 4), 'PCA98: Accuracy Mean') 
    print(round(acc_train_all2.mean(), 4), 'PCA98: Training Accuracy Mean') 
    print(round(acc_all2.std(), 4), 'PCA98: Accuracy SD')
    print(round(acc_train_all2.std(), 4), 'PCA98: Training Accuracy SD')
    print(round(f1_all2.mean(), 4), 'PCA98: F1 Mean') 
    print(round(f1_train_all2.mean(), 4), 'PCA98: Training F1 Mean') 
    print(round(f1_all2.std(), 4), 'PCA98: F1 SD')
    print(round(f1_train_all2.std(), 4), 'PCA98: Training F1 SD')

    #NeuralNetworks
    acc_all = np.zeros(10)
    acc_train_all = np.zeros(10)
    acc_all2 = np.zeros(10)
    acc_train_all2 = np.zeros(10)
    f1_all = np.zeros(10)
    f1_train_all = np.zeros(10)
    f1_all2 = np.zeros(10)
    f1_train_all2 = np.zeros(10) 

    for i in range(10):
        x_train, x_test, y_train, y_test = prepare(df, i)

        acc, acc_train, f1, f1_train = neural(x_train, x_test, y_train, y_test, 'Adam')
        acc_all[i] = acc
        acc_train_all[i] = acc_train
        f1_all[i] = f1
        f1_train_all[i] = f1_train

        acc2, acc_train2, f12, f1_train2 = neural(x_train, x_test, y_train, y_test, "SGD")
        acc_all2[i] = acc2
        acc_train_all2[i] = acc_train2
        f1_all2[i] = f12
        f1_train_all2[i] = f1_train2

    print(round(acc_all.mean(), 4), 'Adam Neural Network: Accuracy Mean') 
    print(round(acc_train_all.mean(), 4), 'Adam Neural Network: Training Accuracy Mean') 
    print(round(acc_all.std(), 4), 'Adam Neural Network: Accuracy SD')
    print(round(acc_train_all.std(), 4), 'Adam Neural Network: Training Accuracy SD')
    print(round(f1_all.mean(), 4), 'Adam Neural Network: F1 Mean') 
    print(round(f1_train_all.mean(), 4), 'Adam Neural Network: Training F1 Mean') 
    print(round(f1_all.std(), 4), 'Adam Neural Network: F1 SD')
    print(round(f1_train_all.std(), 4), 'Adam Neural Network: Training F1 SD')

    print(round(acc_all2.mean(), 4), 'SGD Neural Network: Accuracy Mean') 
    print(round(acc_train_all2.mean(), 4), 'SGD Neural Network: Training Accuracy Mean') 
    print(round(acc_all2.std(), 4), 'SGD Neural Network: Accuracy SD')
    print(round(acc_train_all2.std(), 4), 'SGD Neural Network: Training Accuracy SD')
    print(round(f1_all2.mean(), 4), 'SGD Neural Network: F1 Mean') 
    print(round(f1_train_all2.mean(), 4), 'SGD Neural Network: Training F1 Mean') 
    print(round(f1_all2.std(), 4), 'SGD Neural Network: F1 SD')
    print(round(f1_train_all2.std(), 4), 'SGD Neural Network: Training F1 SD')

    #L2Regularisation
    acc_all = np.zeros(5)
    acc_train_all = np.zeros(5)
    f1_all = np.zeros(5)
    f1_train_all = np.zeros(5)

    weights = [0.001, 0.01, 0.05]
    for i in range(5):
        x_train, x_test, y_train, y_test = prepare(df, i)

        acc, acc_train, f1, f1_train = reg(x_train, x_test, y_train, y_test)
        acc_all[i] = acc
        acc_train_all[i] = acc_train
        f1_all[i] = f1
        f1_train_all[i] = f1_train

    acc_all2 = np.zeros(5)
    acc_train_all2 = np.zeros(5)
    f1_all2 = np.zeros(5)
    f1_train_all2 = np.zeros(5)

    acc_all3 = np.zeros(5)
    acc_train_all3 = np.zeros(5)
    f1_all3 = np.zeros(5)
    f1_train_all3 = np.zeros(5)

    acc_all4 = np.zeros(5)
    acc_train_all4 = np.zeros(5)
    f1_all4 = np.zeros(5)
    f1_train_all4 = np.zeros(5) 

    for weight in weights:
        for j in range(5):
            x_train, x_test, y_train, y_test = prepare(df, j)
            acc2, acc_train2, f12, f1_train2 = regul(x_train, x_test, y_train, y_test, weight)
            if weight == 0.001:
                acc_all2[j] = acc2
                acc_train_all2[j] = acc_train2
                f1_all2[j] = f12
                f1_train_all2[j] = f1_train2
            elif weight == 0.01:
                acc_all3[j] = acc2
                acc_train_all3[j] = acc_train2
                f1_all3[j] = f12
                f1_train_all3[j] = f1_train2
            elif weight == 0.05:
                acc_all4[j] = acc2
                acc_train_all4[j] = acc_train2
                f1_all4[j] = f12
                f1_train_all4[j] = f1_train2  
    
    print(round(acc_all.mean(), 4), 'Keras Adam Neural Network: Accuracy Mean') 
    print(round(acc_train_all.mean(), 4), 'Keras Adam Neural Network: Training Accuracy Mean') 
    print(round(acc_all.std(), 4), 'Keras Adam Neural Network: Accuracy SD')
    print(round(acc_train_all.std(), 4), 'Keras Adam Neural Network: Training Accuracy SD')
    print(round(f1_all.mean(), 4), 'Keras Adam Neural Network: F1 Mean') 
    print(round(f1_train_all.mean(), 4), 'Keras Adam Neural Network: Training F1 Mean') 
    print(round(f1_all.std(), 4), 'Keras Adam Neural Network: F1 SD')
    print(round(f1_train_all.std(), 4), 'Keras Adam Neural Network: Training F1 SD')
    
    print(round(acc_all2.mean(), 4), 'Keras Adam Neural Network (L2 0.001): Accuracy Mean') 
    print(round(acc_train_all2.mean(), 4), 'Keras Adam Neural Network (L2 0.001): Training Accuracy Mean') 
    print(round(acc_all2.std(), 4), 'Keras Adam Neural Network (L2 0.001): Accuracy SD')
    print(round(acc_train_all2.std(), 4), 'Keras Adam Neural Network (L2 0.001): Training Accuracy SD')
    print(round(f1_all2.mean(), 4), 'Keras Adam Neural Network (L2 0.001): F1 Mean') 
    print(round(f1_train_all2.mean(), 4), 'Keras Adam Neural Network (L2 0.001): Training F1 Mean') 
    print(round(f1_all2.std(), 4), 'Keras Adam Neural Network (L2 0.001): F1 SD')
    print(round(f1_train_all2.std(), 4), 'Keras Adam Neural Network (L2 0.001): Training F1 SD')

    print(round(acc_all3.mean(), 4), 'Keras Adam Neural Network (L2 0.01): Accuracy Mean') 
    print(round(acc_train_all3.mean(), 4), 'Keras Adam Neural Network (L2 0.01): Training Accuracy Mean') 
    print(round(acc_all3.std(), 4), 'Keras Adam Neural Network (L2 0.01): Accuracy SD')
    print(round(acc_train_all3.std(), 4), 'Keras Adam Neural Network (L2 0.01): Training Accuracy SD')
    print(round(f1_all3.mean(), 4), 'Keras Adam Neural Network (L2 0.01): F1 Mean') 
    print(round(f1_train_all3.mean(), 4), 'Keras Adam Neural Network (L2 0.01): Training F1 Mean') 
    print(round(f1_all3.std(), 4), 'Keras Adam Neural Network (L2 0.01): F1 SD')
    print(round(f1_train_all3.std(), 4), 'Keras Adam Neural Network (L2 0.01): Training F1 SD')

    print(round(acc_all4.mean(), 4), 'Keras Adam Neural Network (L2 0.05): Accuracy Mean') 
    print(round(acc_train_all4.mean(), 4), 'Keras Adam Neural Network (L2 0.05): Training Accuracy Mean') 
    print(round(acc_all4.std(), 4), 'Keras Adam Neural Network (L2 0.05): Accuracy SD')
    print(round(acc_train_all4.std(), 4), 'Keras Adam Neural Network (L2 0.05): Training Accuracy SD')
    print(round(f1_all4.mean(), 4), 'Keras Adam Neural Network (L2 0.05): F1 Mean') 
    print(round(f1_train_all4.mean(), 4), 'Keras Adam Neural Network (L2 0.05): Training F1 Mean') 
    print(round(f1_all4.std(), 4), 'Keras Adam Neural Network (L2 0.05): F1 SD')
    print(round(f1_train_all4.std(), 4), 'Keras Adam Neural Network (L2 0.05): Training F1 SD')
    
if __name__ == '__main__':
    main()
