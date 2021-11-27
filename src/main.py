import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import r_regression
from skrebate import ReliefF
import matplotlib.pyplot as plt

# Dataset used: default of credit card clients Data Set
# from: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#

if __name__ == '__main__':
    # load the data
    headers = ["limit", "sex", "education", "marriage", "age",
               "pay0", "pay2", "pay3", "pay4", "pay5", "pay6",
               "bill1", "bill2", "bill3", "bill4", "bill5", "bill6",
               "paid1", "paid2", "paid3", "paid4", "paid5", "paid6",
               "default"]
    data_df = pd.read_excel('data/data.xls', skiprows=[0, 1], names=headers)
    # split the data into X and y
    X = data_df.drop(["default"], axis=1)
    y = data_df["default"]
    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    # train without any feature selection or extraction
    # first: knn
    knn_1 = KNeighborsClassifier(n_neighbors=5)
    knn_1.fit(X_train, y_train)
    print("KNN without feature selection completed.")
    # second: svm
    svm_1 = SVC(kernel='rbf', C=1.0)
    svm_1.fit(X_train, y_train)
    print("SVM without feature selection completed.")
    # third: random forest
    rf_1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=10)
    rf_1.fit(X_train, y_train)
    print("Random Forest without feature selection completed.")
    # now, train with feature extraction using PCA
    pca1 = PCA(n_components=13)
    X_train_pca = pca1.fit_transform(X_train)
    print("PCA completed.")
    # first: knn
    knn_2 = KNeighborsClassifier(n_neighbors=5)
    knn_2.fit(X_train_pca, y_train)
    print("KNN with PCA feature extraction completed.")
    # second: svm
    svm_2 = SVC(kernel='rbf', C=1.0)
    svm_2.fit(X_train_pca, y_train)
    print("SVM with PCA feature extraction completed.")
    # third: random forest
    rf_2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=10)
    rf_2.fit(X_train_pca, y_train)
    print("Random Forest with PCA feature extraction completed.")
    # now, train with feature selection using pearson correlation
    # first, let's calculate the pearson correlation matrix
    corr_matrix = pd.DataFrame(X_train).corr().abs()
    # plot the correlation matrix
    plt.figure(figsize=(24, 10))
    sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Reds)
    plt.show()
    # get most correlated features
    most_correlated_features = np.abs(r_regression(X_train, y_train)) > 0.1
    X_train_corr = X_train[:, most_correlated_features]
    # first: knn
    knn_3 = KNeighborsClassifier(n_neighbors=5)
    knn_3.fit(X_train_corr, y_train)
    print("KNN with correlation feature selection completed.")
    # second: svm
    svm_3 = SVC(kernel='rbf', C=1.0)
    svm_3.fit(X_train_corr, y_train)
    print("SVM with correlation feature selection completed.")
    # third: random forest
    rf_3 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=10)
    rf_3.fit(X_train_corr, y_train)
    print("Random Forest with correlation feature selection completed.")
    now, we train with feature selection using relief method
    relief_matrix = ReliefF(n_features_to_select=13, n_neighbors=100, n_jobs=-1)
    relief_matrix.fit(X_train, y_train)
    print("ReliefF feature selection completed.")
    print(relief_matrix)










