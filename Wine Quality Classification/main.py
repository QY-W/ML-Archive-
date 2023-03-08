import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelBinarizer
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
from sklearn.multiclass import _fit_binary
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC, SVR
from sklearn.manifold import TSNE
from sklearn import preprocessing
import os
import time
from sklearn.utils import resample

os.makedirs("./Graphs/", exist_ok=True)


class CustomOneVsRestClassifier(OneVsRestClassifier):
    """
    Not completed
    This part of the code to allow each one vs res classifier to have different parameters
    """

    def __init__(self, estimators, n_jobs=1):
        self.estimators = estimators
        self.n_jobs = n_jobs

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, y):
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_binary)(
            estimator, X, column, classes=[
                "not %s" % self.label_binarizer_.classes_[i],
                self.label_binarizer_.classes_[i]])
                                                        for i, (column, estimator) in
                                                        enumerate(zip(columns, self.estimators)))
        return self


# region plot
def plot_pca(x, y):
    x_pca = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x_pca)
    fig = plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x=principalComponents[:, 0], y=principalComponents[:, 1], hue=y.ravel(), marker="o", s=25, edgecolor="k",
        palette=["C0", "C1", "C2", "C3", "C4", "k"]).set_title("Using PCA to Visualize Classes")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("./Graphs/PCA_plot.png", dpi=400)
    plt.close("all")


def plot_tsne(x, y):
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(x)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=y.ravel(),
        palette=["C0", "C1", "C2", "C3", "C4", "k"]
    ).set_title("Using TSNE to Visualize Classes")
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")
    plt.savefig("./Graphs/TSNE_plot.png", dpi=400)
    plt.close("all")


def plot_lda(x, y):
    x_lda = StandardScaler().fit_transform(x)
    lda = LDA(n_components=2)
    lda_ds = lda.fit_transform(x_lda, y.ravel())
    fig = plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x=lda_ds[:, 0], y=lda_ds[:, 1], hue=y.ravel(), marker="o", s=25, edgecolor="k",
        palette=["C0", "C1", "C2", "C3", "C4", "k"]
    ).set_title("Using LDA to Visualize Classes")
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.savefig("./Graphs/LDA_plot.png", dpi=400)
    plt.close("all")


def plot_confusion_matrix(matrix, title, classes):
    cm_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
    cm_display.plot()
    plt.savefig("./Graphs/{}.png".format(title), dpi=400)
    plt.close("all")


def plot_feature_score(ds):
    scores = ds.iloc[:, -1:].values
    features = ds.iloc[:, 0:-1].values
    fig = plt.figure(figsize=(14, 14))
    plt.barh(features.ravel(), scores.ravel(), color=(0.1, 0.1, 0.1, 0.1), edgecolor='black')
    plt.xscale("log")
    plt.savefig("./Graphs/Features_scores.png", dpi=400)
    plt.close("all")


# endregion

# region classification function
@ignore_warnings(category=ConvergenceWarning)
def oneVsOne(model, xtrain, ytrain, xtest, ytest, model_name, hyper_para=None, label=False):
    if hyper_para is None:
        clfs = OneVsOneClassifier(model, n_jobs=-1)
        start_time_fitting = time.perf_counter()
        clfs.fit(xtrain, ytrain.ravel())
        end_time_fitting = time.perf_counter()
        start_time_test = time.perf_counter()
        yhat = clfs.predict(xtest)
        end_time_test = time.perf_counter()
        print("classification report one vs one classifiers ({}):".format(model_name))
        print(f"Execution Time For fitting :{end_time_fitting - start_time_fitting :0.6f} sec")
        print(f"Execution Time For Test : {end_time_test - start_time_test:0.6f} sec")
        title = "onevsone classifiers({})".format(model_name)
        if not label:
            print(classification_report(ytest, yhat, zero_division=0))
            matrix = confusion_matrix(ytest, yhat)
            plot_confusion_matrix(matrix, title, [3, 4, 5, 6, 7, 8])
        else:
            print(classification_report(ytest, yhat, zero_division=0, target_names=['high', 'low', 'mid']))
            matrix = confusion_matrix(ytest, yhat)

            plot_confusion_matrix(matrix, title, le.classes_)
        return clfs
    else:
        clfs = OneVsOneClassifier(model, n_jobs=-1)
        grids_clf = RandomizedSearchCV(clfs, hyper_para, n_iter=40, cv=5, n_jobs=-1, random_state=50)
        # grids_clf = GridSearchCV(clfs, hyper_para, cv=5, verbose=5, n_jobs=-1)
        start_time_fitting = time.perf_counter()
        clf_opt = grids_clf.fit(xtrain, ytrain.ravel())
        end_time_fitting = time.perf_counter()
        start_time_test = time.perf_counter()
        yhat = clf_opt.predict(xtest)
        end_time_test = time.perf_counter()
        print("classification report one vs one classifiers ({}) with GridSearch:".format(model_name))
        print(f"Execution Time For fitting :{end_time_fitting - start_time_fitting :0.6f} sec")
        print(f"Execution Time For Test : {end_time_test - start_time_test:0.6f} sec")
        title = "onevsone classifiers({})_GridSearch".format(model_name)
        if not label:
            print(classification_report(ytest, yhat, zero_division=0))
            matrix = confusion_matrix(ytest, yhat)
            plot_confusion_matrix(matrix, title, [3, 4, 5, 6, 7, 8])
        else:
            print(classification_report(ytest, yhat, zero_division=0, target_names=['high', 'low', 'mid']))
            matrix = confusion_matrix(ytest, yhat)
            plot_confusion_matrix(matrix, title, le.classes_)
        return clf_opt.best_estimator_


def ensemble_predictions(members, testX):
    """
       (there are some issues with this function)
       This method only used for  ensample_classifiers function
       :param members:
       :param n_members:
       :param testX:
       :param testy:
       :return:
    """
    # make predictions
    yhats = [model.predict_proba(testX) for model in members]
    yhats = np.array(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    result = np.argmax(summed, axis=1)
    result[result == 5] = 8
    result[result == 4] = 7
    result[result == 3] = 6
    result[result == 2] = 5
    result[result == 1] = 4
    result[result == 0] = 3
    return result


def evaluate_n_members(members, n_members, testX, testy):
    """
    This method only used for  ensample_classifiers function
    :param members:
    :param n_members:
    :param testX:
    :param testy:
    :return:
    """
    # select a subset of members
    subset = members[:n_members]
    # make prediction
    yhat = ensemble_predictions(subset, testX)
    # calculate accuracy
    return accuracy_score(testy, yhat)


@ignore_warnings(category=ConvergenceWarning)
def evaluate_model(model, xtrain, ytrain, x_validation, y_validation):
    """
     This method only used for  ensample_classifiers function
    :param model:
    :param xtrain:
    :param ytrain:
    :param x_validation:
    :param y_validation:
    :return:
    """
    model.fit(xtrain, ytrain.ravel())
    yhat = model.predict(x_validation)
    valid_acc = accuracy_score(y_validation, yhat)
    return model, valid_acc


def ensample_classifers(xtrain, ytrain, xtest, ytest, hyper_para, grid_status=False):
    """
    (not complete)
    :param model:
    :param xtrain:
    :param ytrain:
    :param xtest:
    :param ytest:
    :param hyper_para:
    :param grid_status:
    :return:
    """
    n_splits = 10
    scores, members = list(), list()
    for value in hyper_para:
        print("=" * 40)
        print(value)
        random_state = np.random.randint(100, 1000, 10)
        for _, s in zip(range(n_splits), random_state):
            print("random state:", s)
            model = MLPClassifier(max_iter=3000, random_state=55, hidden_layer_sizes=value)
            # split data
            trainX, validX, trainy, validy = train_test_split(xtrain, ytrain, test_size=0.10, random_state=s)
            # evaluate model
            model, test_acc = evaluate_model(model, trainX, trainy, validX, validy)
            print('>%.3f' % test_acc)
            scores.append(test_acc)
            members.append(model)

        indices = [ind for ind, score in enumerate(scores) if score < 0.70]
        for i in sorted(indices, reverse=True):
            del scores[i]
            del members[i]
        print(members)
        print("=" * 40)
    single_scores, ensemble_scores = list(), list()
    if members:
        for i in range(1, len(members) + 1):
            ensemble_score = evaluate_n_members(members, i, xtest, ytest)
            single_yhat = members[i - 1].predict(xtest)
            single_score = accuracy_score(ytest, single_yhat)
            print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
            ensemble_scores.append(ensemble_score)
            single_scores.append(single_score)


@ignore_warnings(category=ConvergenceWarning)
def stacking(group_clf, ds, xtest, ytest):
    # raise NotImplemented
    yhat = group_clf.predict(xtest)
    opt_yhat = []

    low_label_condition = ds.iloc[:, -1].isin([3, 4])
    mid_label_condition = ds.iloc[:, -1].isin([5, 6])
    high_label_condition = ds.iloc[:, -1].isin([7, 8])

    low_labels = pd.DataFrame(ds[low_label_condition])
    mid_labels = pd.DataFrame(ds[mid_label_condition])
    high_labels = pd.DataFrame(ds[high_label_condition])

    X_low = low_labels.iloc[:, :-1].values
    Y_low = low_labels.iloc[:, -1:].values
    X_mid = mid_labels.iloc[:, :-1].values
    Y_mid = mid_labels.iloc[:, -1:].values
    X_high = high_labels.iloc[:, :-1].values
    Y_high = high_labels.iloc[:, -1:].values
    Xtrain_high, _, Ytrain_high, _ = train_test_split(X_high, Y_high, test_size=0.2, random_state=100)
    Xtrain_mid, _, Ytrain_mid, _ = train_test_split(X_mid, Y_mid, test_size=0.2, random_state=100)
    Xtrain_low, _, Ytrain_low, _ = train_test_split(X_low, Y_low, test_size=0.2, random_state=100)

    clf_low = MLPClassifier(max_iter=3000, random_state=55)
    clf_mid = MLPClassifier(max_iter=3000, random_state=55)
    clf_high = MLPClassifier(max_iter=3000, random_state=55)

    clf_low.fit(Xtrain_low, Ytrain_low.ravel())
    clf_mid.fit(Xtrain_mid, Ytrain_mid.ravel())
    clf_high.fit(Xtrain_high, Ytrain_high.ravel())

    for ind, pred in enumerate(yhat):
        if pred == 2:  # mid
            pred2 = clf_mid.predict(xtest[ind, :].reshape(1, -1))
            opt_yhat.append(pred2)
        elif pred == 0:  # high
            pred2 = clf_high.predict(xtest[ind, :].reshape(1, -1))
            opt_yhat.append(pred2)
        elif pred == 1:  # low
            pred2 = clf_low.predict(xtest[ind, :].reshape(1, -1))
            opt_yhat.append(pred2)

    print("classification report Stack classification :")
    print(classification_report(ytest, opt_yhat, zero_division=0))


def rf_classification(xtrain, ytrain, xtest, ytest, data_type, hyper_para=None, label=False):
    if hyper_para is None:
        model = RandomForestClassifier(random_state=55, n_estimators=250, class_weight='balanced')
        start_time_fitting = time.perf_counter()
        model.fit(xtrain, ytrain.ravel())
        end_time_fitting = time.perf_counter()
        start_time_test = time.perf_counter()
        y_hat = model.predict(xtest)
        end_time_test = time.perf_counter()
        print("Classification report for RandomForest on {}:".format(data_type))
        print(f"Execution Time For fitting :{end_time_fitting - start_time_fitting :0.6f} sec")
        print(f"Execution Time For Test : {end_time_test - start_time_test:0.6f} sec")
        title = "RandomForest on {}:".format(data_type)
        if not label:
            print(classification_report(ytest, y_hat, zero_division=0))
            matrix = confusion_matrix(ytest, y_hat)
            plot_confusion_matrix(matrix, title, [3, 4, 5, 6, 7, 8])
        else:
            print(classification_report(ytest, y_hat, zero_division=0, target_names=['high', 'low', 'mid']))
            matrix = confusion_matrix(ytest, y_hat)
            plot_confusion_matrix(matrix, title, le.classes_)
        return model
    else:
        grids_clf = GridSearchCV(RandomForestClassifier(random_state=55, class_weight='balanced'), hyper_para, cv=5,
                                 n_jobs=-1)
        start_time_fitting = time.perf_counter()
        clf_opt = grids_clf.fit(xtrain, ytrain.ravel())
        end_time_fitting = time.perf_counter()
        start_time_test = time.perf_counter()
        yhat = clf_opt.predict(xtest)
        end_time_test = time.perf_counter()
        print("classification report RandomForest with GridSearch {}:".format(data_type))
        print(f"Execution Time For fitting :{end_time_fitting - start_time_fitting :0.6f} sec")
        print(f"Execution Time For Test : {end_time_test - start_time_test:0.6f} sec")
        title = "GridSearch RandomForest on {}:".format(data_type)
        if not label:
            print(classification_report(ytest, yhat, zero_division=0))
            matrix = confusion_matrix(ytest, yhat)
            plot_confusion_matrix(matrix, title, [3, 4, 5, 6, 7, 8])
        else:
            print(classification_report(ytest, yhat, zero_division=0, target_names=['high', 'low', 'mid']))
            matrix = confusion_matrix(ytest, yhat)
            plot_confusion_matrix(matrix, title, le.classes_)
        return clf_opt.best_estimator_


# endregion

# region data preprocessing

def feature_selection(x, y):
    # apply SelectKBest class to extract top best features
    bestFeatures = SelectKBest(score_func=chi2, k='all')
    bestFeaturesFit = bestFeatures.fit(x, y)
    dfscores = pd.DataFrame(bestFeaturesFit.scores_)  # Store predictor scores in a column
    dfcolumns = pd.DataFrame(x.columns)  # Store predictor variable names in a column

    # #concatenate scores with predictor names
    predScores = pd.concat([dfcolumns, dfscores], axis=1)
    predScores.columns = ['Predictor', 'Score']  # naming the dataframe columns
    print(predScores.nlargest(13, 'Score'))  # print top (by score) 10 features
    # df_selected = x.drop('sulphates', 1)
    # df_selected = df_selected.drop('residual sugar', 1)
    df_selected = x.drop(columns='chlorides')
    df_selected = df_selected.drop(columns='pH')
    df_selected = df_selected.drop(columns='density')
    plot_feature_score(predScores.nlargest(13, 'Score'))
    return df_selected.to_numpy()


def group_label(ds):
    low_label_condition = ds.iloc[:, -1].isin([3, 4])
    mid_label_condition = ds.iloc[:, -1].isin([5, 6])
    high_label_condition = ds.iloc[:, -1].isin([7, 8])

    low_labels = pd.DataFrame(ds[low_label_condition])
    mid_labels = pd.DataFrame(ds[mid_label_condition])
    high_labels = pd.DataFrame(ds[high_label_condition])

    # low_labels['quality'].values.astype(str)
    # mid_labels['quality'].values.astype(str)
    # high_labels['quality'].values.astype(str)
    l, m, h = 'l', 'm', 'h'
    low_labels['quality'].replace([3, 4], l, inplace=True)
    mid_labels['quality'].replace([5, 6], m, inplace=True)
    high_labels['quality'].replace([7, 8], h, inplace=True)
    df_grouped = pd.concat([low_labels, mid_labels, high_labels])
    return df_grouped.to_numpy()[:, :-1], df_grouped.to_numpy()[:, -1:]


def upsample_training(ds_unbalanced,n):
    class3_minority_condition = ds_unbalanced.iloc[:, -1].isin([3])
    class4_minority_condition = ds_unbalanced.iloc[:, -1].isin([4])
    class8_minority_condition = ds_unbalanced.iloc[:, -1].isin([8])
    class7_minority_condition = ds_unbalanced.iloc[:, -1].isin([7])
    majority_condition = ds_unbalanced.iloc[:, -1].isin([5, 6,7])
    df_class3_minority = pd.DataFrame(ds_unbalanced[class3_minority_condition])
    df_class4_minority = pd.DataFrame(ds_unbalanced[class4_minority_condition])
    df_class7_minority = pd.DataFrame(ds_unbalanced[class7_minority_condition])
    df_class8_minority = pd.DataFrame(ds_unbalanced[class8_minority_condition])
    df_train_majority = pd.DataFrame(ds_unbalanced[majority_condition])
    df_class3_upsampled = resample(df_class3_minority, replace=True, n_samples=n, random_state=123)
    df_class4_upsampled = resample(df_class4_minority, replace=True, n_samples=n, random_state=123)
    #df_class7_upsampled = resample(df_class7_minority, replace=True, n_samples=200, random_state=123)
    df_class8_upsampled = resample(df_class8_minority, replace=True, n_samples=n, random_state=123)
    df_upsampled = pd.concat([df_train_majority,
                              df_class3_upsampled,
                              df_class4_upsampled,
                              df_class8_upsampled])
    df_upsampled=df_upsampled.sample(frac=1,random_state=1)
    return df_upsampled.to_numpy()[:, :-1], df_upsampled.to_numpy()[:, -1:]


def outliers_detection(data):
    # raise NotImplemented
    data = np.array(data)
    percentile_25 = np.percentile(data, 25)
    percentile_50 = np.percentile(data, 50)
    percentile_75 = np.percentile(data, 75)
    lower_bound = percentile_25 - 1.5 * (percentile_75 - percentile_25)
    upper_bound = percentile_75 + 1.5 * (percentile_75 - percentile_25)
    outliers = []
    for point in list(data):
        if point < lower_bound or point > upper_bound:
            outliers.append(point)
        else:
            outliers.append('not a outlier')

    return outliers


def clean_outliers(ds):
    # raise NotImplemented
    d_outliers_focused = {}
    for name in list(ds):
        d_outliers_focused.setdefault(name, outliers_detection(ds[name]))
    df_outliers_focused = pd.DataFrame(data=d_outliers_focused)

    series_list = []
    for index, row in df_outliers_focused.iterrows():
        for name in list(df_outliers_focused):
            if type(row[name]) == np.float64:
                series_list.append(row)
                break

    df_outliers = pd.DataFrame(series_list, columns=list(df_outliers_focused))
    outliers_indices = df_outliers.index.tolist()
    cleaned_dataset = ds.drop(ds.index[outliers_indices])
    return cleaned_dataset


df = pd.read_csv("./winequality-red.csv", sep=';')
cleaned_df = clean_outliers(df)

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1:].values
X2 = df.iloc[:, :-1]
Y2 = df.iloc[:, -1:]
X_cleaned = cleaned_df.iloc[:, :-1].values
Y_cleaned = cleaned_df.iloc[:, -1:].values
X_grouped, Y_grouped = group_label(df)

le = preprocessing.LabelEncoder()
le.fit(Y_grouped.ravel())
Y_grouped = le.transform(Y_grouped.ravel())

# print(cleaned_df.iloc[:, -1].value_counts())

selected_X = feature_selection(X2, Y2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
X_train_red, X_test_red, Y_train_red, Y_test_red = train_test_split(selected_X, Y, test_size=0.2, random_state=100)
X_train_cleaned, X_test_cleaned, Y_train_cleaned, Y_test_cleaned = train_test_split(X_cleaned, Y_cleaned, test_size=0.2,
                                                                                    random_state=100)
X_train_grouped, X_test_grouped, Y_train_grouped, Y_test_grouped = train_test_split(X_grouped, Y_grouped, test_size=0.2,
                                                                                    random_state=100)

# ressample original training Dataset
# standardize the datasets
norm = StandardScaler().fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)


# train_unbalanced = X_train
# train_unbalanced = np.append(train_unbalanced, Y_train, axis=1)
# df_train_unbalanced = pd.DataFrame(train_unbalanced)
# X_train_upsample, Y_train_upsample = upsample_training(df_train_unbalanced)


train_unbalanced = X_train
train_unbalanced = np.append(train_unbalanced, Y_train, axis=1)
df_train_unbalanced = pd.DataFrame(train_unbalanced)
X_train_upsample, Y_train_upsample = upsample_training(df_train_unbalanced,464)



# standardize the dataset without outlier
norm = StandardScaler().fit(X_train_cleaned)
X_train_cleaned = norm.transform(X_train_cleaned)
X_test_cleaned = norm.transform(X_test_cleaned)


norm = StandardScaler().fit(X_train_red)
X_train_red = norm.transform(X_train_red)
X_test_red = norm.transform(X_test_red)
# endregion

# region hyperparameters
activation = ['identity', 'logistic', 'tanh', 'relu']
solver = ['lbfgs', 'adam']
hidden_layer_sizes = [i for i in range(10, 150, 10)]
# hidden_layer_sizes.extend([(i, math.floor(i / 2)) for i in range(10, 200, 10)])
hyperparameters_mlp = dict(estimator__activation=activation, estimator__solver=solver,
                           estimator__hidden_layer_sizes=hidden_layer_sizes)

n_estimators = [40, 60, 70, 90, 100, 150, 200, 250]
criterion = ['gini', 'entropy', 'log_loss']

hyperparameters_RF2 = dict(estimator__criterion=criterion, estimator__n_estimators=n_estimators)
hyperparameters_RF = dict(criterion=criterion, n_estimators=n_estimators)

kernel = ["rbf", "sigmoid"]
gamma = [2 ** 3, 2 ** 1, 2 ** 0]
gamma.extend([2 ** i for i in range(-15, 0)])
C = [i for i in range(1, 20)]
hyperparameters_SVC = dict(estimator__gamma=gamma, estimator__kernel=kernel, estimator__C=C)
hyperparameters_svc = dict(gamma=gamma, kernel=kernel, C=C)
# endregion

# region running scripts
# region Baggin

# bag = BaggingClassifier(MLPClassifier(max_iter=10000, random_state=55, activation='tanh', hidden_layer_sizes=60), n_estimators=300, random_state=0, bootstrap_features=True, n_jobs=-1).fit(X_train, Y_train.ravel())
# yyyy=bag.predict(X_test)
# print("Classification report for MLP using Bagging")
# print(classification_report(Y_test,yyyy,zero_division=0))
# file="MLP_accuracy.pkl"
# joblib.dump(bag,file)


file_name="MLP_accuracy_updatd.pkl"
if os.path.exists(file_name):
    mlp_best = joblib.load(file_name)
    yyyy=mlp_best.predict(X_test)
    print("Classification report for MLP using Bagging")
    print(classification_report(Y_test,yyyy,zero_division=0))
    matrix = confusion_matrix(Y_test, yyyy)
    plot_confusion_matrix(matrix, "MLP with Bagging", [3, 4, 5, 6, 7, 8])
else:
    print("The pickle files '{}' must be in the same directory to show the best result".format(file_name))
    print("if you want to train the classifer from the scratch uncomment line# 558-563,  \n"
          "and remove the 'MLP_accuracy_updatd.pkl' from the directory. It will take long time to train ")


#endregion





# region SVC
# SVC
oneVsOne(SVC(random_state=55, class_weight='balanced', decision_function_shape='ovo'), X_train, Y_train, X_test, Y_test,
         "SVC", hyperparameters_SVC)

# SVC Grouped label
oneVsOne(SVC(random_state=55, class_weight='balanced', decision_function_shape='ovo'), X_train_grouped,
         Y_train_grouped, X_test_grouped, Y_test_grouped, "SVC - Grouped Label", hyperparameters_SVC, label=True)

# SVC reduced Dataset

oneVsOne(SVC(random_state=55, class_weight='balanced', decision_function_shape='ovo'), X_train_red, Y_train_red,
         X_test_red,
         Y_test_red, "SVC - Reduced Dataset", hyperparameters_SVC)


oneVsOne(SVC(random_state=55, class_weight='balanced', decision_function_shape='ovo'), X_train_upsample, Y_train_upsample,
         X_test,
         Y_test, "SVC - upsample Dataset", hyperparameters_SVC)

# endregion

# region MLP
# MLP
oneVsOne(MLPClassifier(max_iter=3000, random_state=55), X_train,
         Y_train, X_test, Y_test, "MLP")  # best MLP


# oneVsOne(MLPClassifier(max_iter=3000, random_state=55), X_train,
#          Y_train, X_test, Y_test,"MLP" , hyperparameters_mlp)

oneVsOne(MLPClassifier(max_iter=10000, random_state=55), X_train_upsample,
         Y_train_upsample, X_test, Y_test,"MLP - Upsample DS")

# MLP with reduced feature

oneVsOne(MLPClassifier(max_iter=3000, random_state=55), X_train_red,
         Y_train_red, X_test_red, Y_test_red, "MLP - reduced DS")

# MLP with cleaned Dataset
# oneVsOne(MLPClassifier(max_iter=3000, random_state=55), X_train_cleaned,
#          Y_train_cleaned, X_test_cleaned, Y_test_cleaned, "MLP - Dataset without Outliers")

# MLP with Grouped labels
oneVsOne(MLPClassifier(max_iter=3000, random_state=55), X_train_grouped,
         Y_train_grouped, X_test_grouped, Y_test_grouped, "MLP - Grouped Labels", label=True)
# endregion

# region RandomForest
# Random Forest
oneVsOne(RandomForestClassifier(random_state=55), X_train,
         Y_train, X_test, Y_test, "RandomForest - Original DS")

oneVsOne(RandomForestClassifier(random_state=55), X_train,
         Y_train, X_test, Y_test, "Random Forest - Original DS", hyperparameters_RF2)

rf_classification(X_train, Y_train, X_test, Y_test, "original DS")
rf_classification(X_train, Y_train, X_test, Y_test, "Original DS", hyperparameters_RF)

# Random Forest with cleaned Dataset
# rf_classification(X_train_cleaned, Y_train_cleaned, X_test_cleaned, Y_test_cleaned,data_type="Cleaned DS")
# rf_classification(X_train_cleaned,Y_train_cleaned, X_test_cleaned, Y_test_cleaned, hyperparameters_RF, data_type="Cleaned DS")

# RandomForest reduced feature
oneVsOne(RandomForestClassifier(random_state=55), X_train_red,
         Y_train_red, X_test_red, Y_test_red, "RandomForest Reduced Dataset")
#
# oneVsOne(RandomForestClassifier(random_state=55), X_train_red,
#          Y_train_red, X_test_red, Y_test_red,"RandomForest - Reduced features DS" ,hyperparameters_RF2)
rf_classification(X_train_red, Y_train_red, X_test_red, Y_test_red, data_type="reduced features DS")
rf_classification(X_train_upsample, Y_train_upsample, X_test, Y_test, "upsample DS", hyperparameters_RF)
# rf_classification(X_train_red, Y_train_red, X_test_red, Y_test_red, hyperparameters_RF, data_type="reduced features DS")

# RandomForest  grouped label
clf_group = rf_classification(X_train_grouped, Y_train_grouped, X_test_grouped, Y_test_grouped, data_type="Grouped DS",
                              label=True)


n_estimators = [40,90]
criterion = ['gini', 'entropy']
hyperparameters_RF = dict(criterion=criterion, n_estimators=n_estimators)
rf_classification(X_train_upsample, Y_train_upsample, X_test, Y_test, "upsample DS", hyperparameters_RF)

# endregion


stacking(clf_group, df, X_test, Y_test)

plot_pca(X, Y)
plot_lda(X, Y)
plot_tsne(X, Y)
# endregion
