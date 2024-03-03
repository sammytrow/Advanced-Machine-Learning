import numpy as np
from matplotlib import pyplot as plt


def plot_knn(accuracy, title):
    plt.figure()
    plt.title(title)
    plt.plot(accuracy)
    filepath = "plots/" + title.replace(" ", "_") + "_overK.jpg"
    plt.savefig(filepath)
    plt.show()

def plot_svm(stats, title):
    # not working
    plt.figure()
    plt.plot(stats['recall'], stats['precision'])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title + " Precision over Recall")
    filepath = "plots/" + title.replace(" ", "_") + "_Precision_over_Recall.jpg"
    plt.savefig(filepath)
    fig, ax = plt.subplots()
    ax.plot(stats['recall'], stats['precision'], color='purple')

    # add axis labels to plot
    ax.set_title(title + 'Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    filepath = "plots/" + title.replace(" ", "_") + "Precision-Recall_Curve.jpg"
    fig.savefig(filepath)
    # display plot
    plt.show()
    # shit results
    plt.figure()
    plt.plot(stats['roc'][0], stats['roc'][1])
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title + " ROC curve")
    plt.grid(color='black', linestyle='-', linewidth=0.5)
    filepath = "plots/" + title.replace(" ", "_") + ".jpg"
    plt.savefig(filepath)
    plt.show()

def bar_plot(a_acc, b_acc, title):
    plt.figure()
    plt.title(title)
    plt.barh('A train B test', a_acc)
    plt.barh('B train A test', b_acc)
    plt.xlim(0, 1)
    plt.subplots_adjust(left=0.2)
    filepath = "plots/" + title.replace(" ", "_") + ".jpg"
    plt.savefig(filepath)
    plt.show()

def scatter_example(row, title):
    row2 = np.delete(row, 0)
    row2  = row2.reshape(100, 3)

    i = 0
    counter = 1
    X = []
    Y = []
    Z = []
    while i < len(row):
        if i == 0:
            pass
        elif counter == 1:
            X.append(row[i])
            counter += 1
        elif counter == 2:
            Y.append(row[i])
            counter += 1
        elif counter == 3:
            Z.append(row[i])
            counter = 1
        i += 1

    plt.gca().invert_yaxis()
    plt.scatter(X, Y)
    filepath = "plots/" + title.replace(" ", "_") + ".jpg"
    plt.title(title)
    plt.savefig(filepath)
    plt.show()
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(-np.array(X), -np.array(Z), -np.array(Y))
    plt.title(title)
    filepath = "plots/" + title.replace(" ", "_") + "_3d.jpg"
    plt.savefig(filepath)
    plt.show()