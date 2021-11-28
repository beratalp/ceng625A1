import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import r_regression, SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report,\
    RocCurveDisplay
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
    knn_1 = KNeighborsClassifier(n_neighbors=11)
    knn_1.fit(X_train, y_train)
    print("KNN without feature selection completed.")
    # second: svm
    svm_1 = SVC(kernel='rbf', C=1.0)
    svm_1.fit(X_train, y_train)
    print("SVM without feature selection completed.")
    # third: random forest
    rf_1 = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=10)
    rf_1.fit(X_train, y_train)
    print("Random Forest without feature selection completed.")
    # now, train with feature extraction using PCA
    pca1 = PCA(n_components=13)
    X_train_pca = pca1.fit_transform(X_train)
    X_test_pca = pca1.fit_transform(X_test)
    print("PCA completed.")
    # first: knn
    knn_2 = KNeighborsClassifier(n_neighbors=11)
    knn_2.fit(X_train_pca, y_train)
    print("KNN with PCA feature extraction completed.")
    # second: svm
    svm_2 = SVC(kernel='rbf', C=1.0)
    svm_2.fit(X_train_pca, y_train)
    print("SVM with PCA feature extraction completed.")
    # third: random forest
    rf_2 = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=10)
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
    X_test_corr = X_test[:, most_correlated_features]
    # first: knn
    knn_3 = KNeighborsClassifier(n_neighbors=11)
    knn_3.fit(X_train_corr, y_train)
    print("KNN with correlation feature selection completed.")
    # second: svm
    svm_3 = SVC(kernel='rbf', C=1.0)
    svm_3.fit(X_train_corr, y_train)
    print("SVM with correlation feature selection completed.")
    # third: random forest
    rf_3 = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=10)
    rf_3.fit(X_train_corr, y_train)
    # print("Random Forest with correlation feature selection completed.")
    # now, we train with feature selection using information gain method
    most_info_gainer_features = mutual_info_classif(X_train, y_train) > 0.01
    X_train_info_gain = X_train[:, most_info_gainer_features]
    X_test_info_gain = X_test[:, most_info_gainer_features]
    # first: knn
    knn_4 = KNeighborsClassifier(n_neighbors=11)
    knn_4.fit(X_train_info_gain, y_train)
    print("KNN with information gain feature selection completed.")
    # second: svm
    svm_4 = SVC(kernel='rbf', C=1.0)
    svm_4.fit(X_train_info_gain, y_train)
    print("SVM with information gain feature selection completed.")
    # third: random forest
    rf_4 = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=10)
    rf_4.fit(X_train_info_gain, y_train)
    print("Random Forest with information gain feature selection completed.")
    # now, we train with both correlation and information gain
    X_train_hybrid = SelectKBest(mutual_info_classif, k=5).fit_transform(X_train_corr, y_train)
    X_test_hybrid = SelectKBest(mutual_info_classif, k=5).fit_transform(X_test_corr, y_test)
    # first: knn
    knn_5 = KNeighborsClassifier(n_neighbors=11)
    knn_5.fit(X_train_hybrid, y_train)
    print("KNN with hybrid feature selection completed.")
    # second: svm
    svm_5 = SVC(kernel='rbf', C=1.0)
    svm_5.fit(X_train_hybrid, y_train)
    print("SVM with hybrid feature selection completed.")
    # third: random forest
    rf_5 = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=10)
    rf_5.fit(X_train_hybrid, y_train)
    print("Random Forest with hybrid feature selection completed.")
    # now, we test the models
    # first: models with no feature selection or feature extraction
    # KNN
    y_pred_knn_1 = knn_1.predict(X_test)
    knn_1_conf_matrix = confusion_matrix(y_test, y_pred_knn_1)
    ConfusionMatrixDisplay(knn_1_conf_matrix).plot()
    plt.title("KNN without feature selection")
    plt.show()
    print("KNN without feature selection accuracy: ", accuracy_score(y_test, y_pred_knn_1))
    print("KNN without feature selection classification report:\n", classification_report(y_test, y_pred_knn_1))
    RocCurveDisplay.from_predictions(y_test, y_pred_knn_1).plot()
    plt.title("KNN without feature selection")
    plt.show()
    # SVM
    y_pred_svm_1 = svm_1.predict(X_test)
    svm_1_conf_matrix = confusion_matrix(y_test, y_pred_svm_1)
    ConfusionMatrixDisplay(svm_1_conf_matrix).plot()
    plt.title("SVM without feature selection")
    plt.show()
    print("SVM without feature selection accuracy: ", accuracy_score(y_test, y_pred_svm_1))
    print("SVM without feature selection classification report:\n", classification_report(y_test, y_pred_svm_1))
    RocCurveDisplay.from_predictions(y_test, y_pred_svm_1).plot()
    plt.title("SVM without feature selection")
    plt.show()
    # Random Forest
    y_pred_rf_1 = rf_1.predict(X_test)
    rf_1_conf_matrix = confusion_matrix(y_test, y_pred_rf_1)
    ConfusionMatrixDisplay(rf_1_conf_matrix).plot()
    plt.title("Random Forest without feature selection")
    plt.show()
    print("Random Forest without feature selection accuracy: ", accuracy_score(y_test, y_pred_rf_1))
    print("Random Forest without feature selection classification report:\n", classification_report(y_test, y_pred_rf_1))
    RocCurveDisplay.from_predictions(y_test, y_pred_rf_1, ).plot()
    plt.title("Random Forest without feature selection")
    plt.show()
    plt.figure(figsize=(10, 10))
    plt.show()
    # second: models with PCA feature extraction
    # KNN
    y_pred_knn_2 = knn_2.predict(X_test_pca)
    knn_2_conf_matrix = confusion_matrix(y_test, y_pred_knn_2)
    ConfusionMatrixDisplay(knn_2_conf_matrix).plot()
    plt.title("KNN with PCA feature extraction")
    plt.show()
    print("KNN with PCA feature extraction accuracy: ", accuracy_score(y_test, y_pred_knn_2))
    print("KNN with PCA feature extraction classification report:\n", classification_report(y_test, y_pred_knn_2))
    RocCurveDisplay.from_predictions(y_test, y_pred_knn_2).plot()
    plt.title("KNN with PCA feature extraction")
    plt.show()
    # SVM
    y_pred_svm_2 = svm_2.predict(X_test_pca)
    svm_2_conf_matrix = confusion_matrix(y_test, y_pred_svm_2)
    ConfusionMatrixDisplay(svm_2_conf_matrix).plot()
    plt.title("SVM with PCA feature extraction")
    plt.show()
    print("SVM with PCA feature extraction accuracy: ", accuracy_score(y_test, y_pred_svm_2))
    print("SVM with PCA feature extraction classification report:\n", classification_report(y_test, y_pred_svm_2))
    RocCurveDisplay.from_predictions(y_test, y_pred_svm_2).plot()
    plt.title("SVM with PCA feature extraction")
    plt.show()
    # Random Forest
    y_pred_rf_2 = rf_2.predict(X_test_pca)
    rf_2_conf_matrix = confusion_matrix(y_test, y_pred_rf_2)
    ConfusionMatrixDisplay(rf_2_conf_matrix).plot()
    plt.title("Random Forest with PCA feature extraction")
    print("Random Forest with PCA feature extraction accuracy: ", accuracy_score(y_test, y_pred_rf_2))
    print("Random Forest with PCA feature extraction classification report:\n", classification_report(y_test, y_pred_rf_2))
    RocCurveDisplay.from_predictions(y_test, y_pred_rf_2).plot()
    plt.title("Random Forest with PCA feature extraction")
    plt.show()
    # third: models with correlation feature selection
    # KNN
    y_pred_knn_3 = knn_3.predict(X_test_corr)
    knn_3_conf_matrix = confusion_matrix(y_test, y_pred_knn_3)
    ConfusionMatrixDisplay(knn_3_conf_matrix).plot()
    plt.title("KNN with correlation feature selection")
    plt.show()
    print("KNN with correlation feature selection accuracy: ", accuracy_score(y_test, y_pred_knn_3))
    print("KNN with correlation feature selection classification report:\n", classification_report(y_test, y_pred_knn_3))
    RocCurveDisplay.from_predictions(y_test, y_pred_knn_3).plot()
    plt.title("KNN with correlation feature selection")
    plt.show()
    # SVM
    y_pred_svm_3 = svm_3.predict(X_test_corr)
    svm_3_conf_matrix = confusion_matrix(y_test, y_pred_svm_3)
    ConfusionMatrixDisplay(svm_3_conf_matrix).plot()
    plt.title("SVM with correlation feature selection")
    plt.show()
    print("SVM with correlation feature selection accuracy: ", accuracy_score(y_test, y_pred_svm_3))
    print("SVM with correlation feature selection classification report:\n", classification_report(y_test, y_pred_svm_3))
    RocCurveDisplay.from_predictions(y_test, y_pred_svm_3).plot()
    plt.title("SVM with correlation feature selection")
    plt.show()
    # Random Forest
    y_pred_rf_3 = rf_3.predict(X_test_corr)
    rf_3_conf_matrix = confusion_matrix(y_test, y_pred_rf_3)
    ConfusionMatrixDisplay(rf_3_conf_matrix).plot()
    plt.title("Random Forest with correlation feature selection")
    plt.show()
    print("Random Forest with correlation feature selection accuracy: ", accuracy_score(y_test, y_pred_rf_3))
    print("Random Forest with correlation feature selection classification report:\n",
          classification_report(y_test, y_pred_rf_3))
    RocCurveDisplay.from_predictions(y_test, y_pred_rf_3).plot()
    plt.title("Random Forest with correlation feature selection")
    plt.show()
    # fourth: models with information gain feature selection
    # KNN
    y_pred_knn_4 = knn_4.predict(X_test_info_gain)
    knn_4_conf_matrix = confusion_matrix(y_test, y_pred_knn_4)
    ConfusionMatrixDisplay(knn_4_conf_matrix).plot()
    plt.title("KNN with information gain feature selection")
    plt.show()
    print("KNN with information gain feature selection accuracy: ", accuracy_score(y_test, y_pred_knn_4))
    print("KNN with information gain feature selection classification report:\n",
          classification_report(y_test, y_pred_knn_4))
    RocCurveDisplay.from_predictions(y_test, y_pred_knn_4).plot()
    plt.title("KNN with information gain feature selection")
    plt.show()
    # SVM
    y_pred_svm_4 = svm_4.predict(X_test_info_gain)
    svm_4_conf_matrix = confusion_matrix(y_test, y_pred_svm_4)
    ConfusionMatrixDisplay(svm_4_conf_matrix).plot()
    plt.title("SVM with information gain feature selection")
    plt.show()
    print("SVM with information gain feature selection accuracy: ", accuracy_score(y_test, y_pred_svm_4))
    print("SVM with information gain feature selection classification report:\n",
          classification_report(y_test, y_pred_svm_4))
    RocCurveDisplay.from_predictions(y_test, y_pred_svm_4).plot()
    plt.title("SVM with information gain feature selection")
    plt.show()
    # Random Forest
    y_pred_rf_4 = rf_4.predict(X_test_info_gain)
    rf_4_conf_matrix = confusion_matrix(y_test, y_pred_rf_4)
    ConfusionMatrixDisplay(rf_4_conf_matrix).plot()
    plt.title("Random Forest with information gain feature selection")
    plt.show()
    print("Random Forest with information gain feature selection accuracy: ", accuracy_score(y_test, y_pred_rf_4))
    print("Random Forest with information gain feature selection classification report:\n",
          classification_report(y_test, y_pred_rf_4))
    RocCurveDisplay.from_predictions(y_test, y_pred_rf_4).plot()
    plt.title("Random Forest with information gain feature selection")
    plt.show()
    # fifth: models with hybrid feature selection
    # KNN
    y_pred_knn_5 = knn_5.predict(X_test_hybrid)
    knn_5_conf_matrix = confusion_matrix(y_test, y_pred_knn_5)
    ConfusionMatrixDisplay(knn_5_conf_matrix).plot()
    plt.title("KNN with hybrid feature selection")
    plt.show()
    print("KNN with hybrid feature selection accuracy: ", accuracy_score(y_test, y_pred_knn_5))
    print("KNN with hybrid feature selection classification report:\n",
          classification_report(y_test, y_pred_knn_5))
    RocCurveDisplay.from_predictions(y_test, y_pred_knn_5).plot()
    plt.title("KNN with hybrid feature selection")
    plt.show()
    # SVM
    y_pred_svm_5 = svm_5.predict(X_test_hybrid)
    svm_5_conf_matrix = confusion_matrix(y_test, y_pred_svm_5)
    ConfusionMatrixDisplay(svm_5_conf_matrix).plot()
    plt.title("SVM with hybrid feature selection")
    plt.show()
    print("SVM with hybrid feature selection accuracy: ", accuracy_score(y_test, y_pred_svm_5))
    print("SVM with hybrid feature selection classification report:\n",
          classification_report(y_test, y_pred_svm_5))
    RocCurveDisplay.from_predictions(y_test, y_pred_svm_5).plot()
    plt.title("SVM with hybrid feature selection")
    plt.show()
    # Random Forest
    y_pred_rf_5 = rf_5.predict(X_test_hybrid)
    rf_5_conf_matrix = confusion_matrix(y_test, y_pred_rf_5)
    ConfusionMatrixDisplay(rf_5_conf_matrix).plot()
    plt.title("Random Forest with hybrid feature selection")
    plt.show()
    print("Random Forest with hybrid feature selection accuracy: ", accuracy_score(y_test, y_pred_rf_5))
    print("Random Forest with hybrid feature selection classification report:\n",
          classification_report(y_test, y_pred_rf_5))
    RocCurveDisplay.from_predictions(y_test, y_pred_rf_5).plot()
    plt.title("Random Forest with hybrid feature selection")
    plt.show()



