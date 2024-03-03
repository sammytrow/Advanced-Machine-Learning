import numpy as np
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler#
from collections import Counter
from data_prep import *
from matplotlib.backends.backend_pdf import PdfPages
from plots import *
from support_vector import SupportVector
from nearest_neighbour import knearestneighbour


a_data_list= ['../Data/a_affirmative_datapoints.txt', '../Data/a_conditional_datapoints.txt', '../Data/a_doubt_question_datapoints.txt',
              '../Data/a_emphasis_datapoints.txt', '../Data/a_negative_datapoints.txt', '../Data/a_relative_datapoints.txt',
              '../Data/a_topics_datapoints.txt', '../Data/a_wh_question_datapoints.txt', '../Data/a_yn_question_datapoints.txt']
a_target_list= ['../Data/a_affirmative_targets.txt', '../Data/a_conditional_targets.txt', '../Data/a_doubt_question_targets.txt',
              '../Data/a_emphasis_targets.txt', '../Data/a_negative_targets.txt', '../Data/a_relative_targets.txt',
              '../Data/a_topics_targets.txt', '../Data/a_wh_question_targets.txt', '../Data/a_yn_question_targets.txt']
b_data_list= ['../Data/b_affirmative_datapoints.txt', '../Data/b_conditional_datapoints.txt', '../Data/b_doubt_question_datapoints.txt',
              '../Data/b_emphasis_datapoints.txt', '../Data/b_negative_datapoints.txt', '../Data/b_relative_datapoints.txt',
              '../Data/b_topics_datapoints.txt', '../Data/b_wh_question_datapoints.txt', '../Data/b_yn_question_datapoints.txt']
b_target_list= ['../Data/b_affirmative_targets.txt', '../Data/b_conditional_targets.txt', '../Data/b_doubt_question_targets.txt',
              '../Data/b_emphasis_targets.txt', '../Data/b_negative_targets.txt', '../Data/b_relative_targets.txt',
              '../Data/b_topics_targets.txt', '../Data/b_wh_question_targets.txt', '../Data/b_yn_question_targets.txt']
expressions_list = ['Affirmative', 'Conditional', 'Doubt question',
              'Emphasis', 'Negative', 'Relative',
              'Topics', 'wh question', 'Yes no question']

def load_data(counter):
    data = np.loadtxt(a_data_list[counter], skiprows=1, delimiter=' ', usecols=range(0, 301))
    targets = np.loadtxt(a_target_list[counter])

    test_data = np.loadtxt(b_data_list[counter], skiprows=1, delimiter=' ', usecols=range(0, 301))
    test_targets = np.loadtxt(b_target_list[counter])

    norm_data = normalize_data(data)

    test_norm = normalize_data(test_data)

    return norm_data, targets, test_norm, test_targets

def basic_knn(X, Y, x_test, y_test, N, setN, num_k):
    accuracy = []
    report = []
    for i in range(1,N):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X,Y)
        results = knn.predict(x_test)
        accuracy.append(accuracy_score(y_test, results))
        report.append(classification_report(y_test, results, labels=np.unique(results)))

    k = StratifiedKFold(n_splits=num_k)
    kfold = k.split(X, Y)

    pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=setN))
    scores = []

    for k, (train, test) in enumerate(kfold):
        pipeline.fit(X[train], Y[train])
        score = pipeline.score(X[test], Y[test])
        scores.append(score)
        #print('Fold: ', score)
    cross_val_acc = np.mean(scores)
    results = pipeline.predict(x_test)

    precision, recall, thresholds = precision_recall_curve(y_test, results)

    print(classification_report(y_test, results))
    print(f1_score(y_test, results, average='weighted', labels=np.unique(results)))

    stats = {"accuracy": accuracy_score(y_test, results), "precision": precision, "recall": recall,
             "f1": f1_score(y_test, results, average='weighted'),
             "roc": roc_curve(y_test, results)}  # , "f" : f}#, "auc" : auc(y_test, results)}

    return accuracy, report, stats, cross_val_acc

def do_knn(counter, X, Y, testx, testy, type, N, setN, k):
    title_a = "" + type + "A train B test KNN " + str(expressions_list[counter])
    print(title_a)
    acc, report, a_train_stats, a_cross = basic_knn(X, Y, testx, testy, N, setN, k)

    plot_knn(acc, title_a)
    title_b = "" + type + "B train A test KNN " + str(expressions_list[counter])
    print(title_b)
    acc, report, b_train_stats, b_cross = basic_knn(testx, testy, X, Y, N, setN, k)

    plot_knn(acc, title_b)
    plot_svm(a_train_stats, title_a)
    plot_svm(b_train_stats, title_b)
    title = "" + type + "KNN " + str(expressions_list[counter])
    bar_plot(a_train_stats['accuracy'], b_train_stats['accuracy'], title)
    return a_cross, b_cross

def scratch_knn(counter, X, Y, testx, testy, type, N, setN):
    accuracy = []
    report = []
    for i in range(1, N):
        knn = knearestneighbour(i)
        knn.fit(X, Y)
        results = knn.prediction(testx)
        accuracy.append(accuracy_score(testy, results))
        report.append(classification_report(testy, results, labels=np.unique(results)))

    final_knn = KNeighborsClassifier(n_neighbors=setN)
    final_knn.fit(X, Y)
    final_results = final_knn.predict(testx)

    precision, recall, thresholds = precision_recall_curve(testy, final_results)

    print(classification_report(testy, final_results))
    print(f1_score(testy, final_results, average='weighted', labels=np.unique(final_results)))
    stats = {"accuracy": accuracy_score(testy, final_results), "precision": precision, "recall": recall,
             "f1": f1_score(testy, final_results, average='weighted'),
             "roc": roc_curve(testy, final_results)}  # , "f" : f}#, "auc" : auc(y_test, results)}

    return accuracy, report, stats

def do_scratch_knn(counter, X, Y, testx, testy, type, N, setN):
    titlea = "" + type + "A train B test Scratch KNN " + str(expressions_list[counter])
    print(titlea)
    acc, report, a_train_stats = scratch_knn(counter, X, Y, testx, testy, type, N, setN)

    plot_knn(acc, titlea)
    titleb = "" + type + "B train A test Scratch KNN " + str(expressions_list[counter])
    print(titleb)
    acc, report, b_train_stats = scratch_knn(counter, testx, testy, X, Y, type, N, setN)

    plot_knn(acc, titleb)
    plot_svm(a_train_stats, titlea)
    plot_svm(b_train_stats, titleb)
    title = "" + type + "scratch KNN " + str(expressions_list[counter])
    bar_plot(a_train_stats['accuracy'], b_train_stats['accuracy'], title)

def knn_loop(counter, X, Y, testx, testy, k, ispca):
    train_data, targets = X, Y
    test_data, test_targets = testx, testy
    type = "" + ispca + ""
    #a_cross, b_crwessssq   w
    #do_knn(counter, train_data, targets, test_data, test_targets, type, 50, 18, k)
    do_scratch_knn(counter, train_data, targets, test_data, test_targets, type, 50, 18)

    train_data, targets = oversampling(X, Y)
    test_data, test_targets = oversampling(testx, testy)
    type = "Oversampling " + ispca + ""
    #do_knn(counter, train_data, targets, test_data, test_targets, type, 100, 18, k)
    do_scratch_knn(counter, train_data, targets, test_data, test_targets, type, 50, 18)

    train_data, targets = undersampling(X, Y)
    test_data, test_targets = undersampling(testx, testy)
    type = "Undersampling " + ispca + ""

    #do_knn(counter, train_data, targets, test_data, test_targets, type, 100, 18, k)
    #do_scratch_knn(counter, train_data, targets, test_data, test_targets, type, 50, 18)

def basic_svm(X, Y, x_test, y_test, num_k):
    k = StratifiedKFold(n_splits=num_k)
    kfold = k.split(X, Y)

    pipeline = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf'))
    scores = []

    for k, (train, test) in enumerate(kfold):
        pipeline.fit(X[train], Y[train])
        score = pipeline.score(X[test], Y[test])
        scores.append(score)
        #print('Fold: ', score)

    cross_val_acc = np.mean(scores)
    #print('\n\nCross-Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    results = pipeline.predict(x_test)

    precision, recall, thresholds = precision_recall_curve(y_test, results)
    print(classification_report(y_test, results))
    print(f1_score(y_test, results, average='weighted', labels=np.unique(results)))

    stats = {"accuracy" : accuracy_score(y_test, results), "precision" : precision, "recall" : recall, "f1" : f1_score(y_test, results, average='weighted'),
    "roc" : roc_curve(y_test, results)}#, "f" : f}#, "auc" : auc(y_test, results)}

    return stats, cross_val_acc

def do_svm(counter, X, Y, testx, testy, type, k):
    title = "" + type + "A train B test SVM " + str(expressions_list[counter])
    print(title)
    a_train_stats, a_cross_val = basic_svm(X, Y, testx, testy, k)
    plot_svm(a_train_stats, title)

    title = "" + type + "B train A test SVM " + str(expressions_list[counter])
    print(title)
    b_train_stats, b_cross_val = basic_svm(testx, testy, X, Y, k)
    plot_svm(b_train_stats, title)
    title = "" + type + " SVM " + str(expressions_list[counter])
    bar_plot(a_train_stats['accuracy'], b_train_stats['accuracy'], title)
    return a_cross_val, b_cross_val

def do_scratch_svm(counter, X, Y, testx, testy, type):
    titlea = "" + type + "A train B test scratch SVM " + str(expressions_list[counter])
    titleb = "" + type + "B train A test scratch SVM " + str(expressions_list[counter])

    print(titlea)
    supvm = SupportVector(10000, 0.000001, 0.01)
    supvm.train(X, Y)
    supvm.plot_results(X, titlea)
    result = supvm.prediction(testx)

    precision, recall, thresholds = precision_recall_curve(testy, result)
    print(classification_report(testy, result))
    print(f1_score(testy, result, average='weighted', labels=np.unique(result)))

    a_stats = {"accuracy": accuracy_score(testy, result), "precision": precision, "recall": recall,
               "f1": f1_score(testy, result, average='weighted'),
               "roc": roc_curve(testy, result)}  # , "f" : f}#, "auc" : auc(y_test, results)}
    print(titleb)
    supvm = SupportVector(10000, 0.1, 0.01)
    supvm.train(testx, testy)
    supvm.plot_results(testx, titleb)
    b_result = supvm.prediction(X)
    print(classification_report(Y, b_result))
    print(f1_score(Y, b_result, average='weighted', labels=np.unique(b_result)))

    precision, recall, thresholds = precision_recall_curve(Y, b_result)
    b_stats = {"accuracy": accuracy_score(Y, b_result), "precision": precision, "recall": recall,
               "f1": f1_score(Y, b_result, average='weighted'),
               "roc": roc_curve(Y, b_result)}  # , "f" : f}#, "auc" : auc(y_test, results)}

    plot_svm(a_stats, titlea)
    plot_svm(b_stats, titleb)
    title = "" + type + "scratch SVM " + str(expressions_list[counter])
    bar_plot(a_stats['accuracy'], b_stats['accuracy'], title)

def svm_loop(counter, X, Y, testx, testy, k, ispca):
    train_data, targets = X, Y
    test_data, test_targets = testx, testy
    type = "" + ispca + ""
    a_cross, b_cross = do_svm(counter, train_data, targets, test_data, test_targets, type, k)
    do_scratch_svm(counter, train_data, targets, test_data, test_targets, type)

    train_data, targets = oversampling(X, Y)
    test_data, test_targets = oversampling(testx, testy)
    type = "Oversampling " + ispca + ""
    a_cross,b_cross = do_svm(counter, train_data, targets, test_data, test_targets, type, k)
    do_scratch_svm(counter, train_data, targets, test_data, test_targets, type)

    train_data, targets = undersampling(X, Y)
    test_data, test_targets = undersampling(testx, testy)
    type = "Undersampling " + ispca + ""
    a_cross,b_cross = do_svm(counter, train_data, targets, test_data, test_targets, type, k)
    do_scratch_svm(counter, train_data, targets, test_data, test_targets, type)
    return a_cross,b_cross

def main(N):

    for i in range(1, N):
        X, Y, testX, testY = load_data(i)
        scatter_example(X[0], str(expressions_list[i]))
        #a_cross,b_cross = svm_loop(i, X, Y, testX, testY, 3, "")
        #print(a_cross,b_cross)
        #pcaX, pcatestX = apply_pca(X, testX)
        #a_cross,b_cross = svm_loop(i, pcaX, Y, pcatestX, testY, 3, "PCA ")
        #print(a_cross,b_cross)
        knn_loop(i, X, Y, testX, testY, 3, "")
        #knn_loop(i, pcaX, Y, pcatestX, testY, 3, "PCA ")

def number_k_svm(N):

    for i in range(1, N):
        X, Y, testX, testY = load_data(i)
        scatter_example(X[0])
        a_cross_val_score = []
        b_cross_val_score = []
        pcaX, pcatestX = apply_pca(X, testX)
        for k in range(2,20):
            a_cross,b_cross = knn_loop(i, pcaX, Y, pcatestX, testY, k)
            a_cross_val_score.append(a_cross)
            b_cross_val_score.append(b_cross)
        plt.figure()
        plt.title("PCA KNN various number of k-fold A train K = 18")
        plt.plot(a_cross_val_score)
        plt.xlabel('Number of K')
        plt.ylabel('Cross Validation Accuracy')
        plt.savefig("plots/PCA_KNN_various_number_of_k-fold_A_train.jpg")
        plt.show()
        plt.figure()
        plt.title("PCA KNN various number of k-fold B train K = 18")
        plt.plot(b_cross_val_score)
        plt.xlabel('Number of K')
        plt.ylabel('Cross Validation Accuracy')
        plt.savefig("plots/PCA_KNN_various_number_of_k-fold_B_train.jpg")
        plt.show()

main(2)
#test_k_svm(2)