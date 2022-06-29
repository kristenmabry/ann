import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

def getResults(evalData: pd.DataFrame, guesses: list, runName: str, txtFile: str, imgFile: str, save: bool = True):
    # create confusion matrix
    cm = confusion_matrix(evalData['class'], guesses)

    # calculate error I column
    error1 = []
    for i in range(0, len(cm)):
        error1.append([sum(cm[i]) - cm[i][i]])

    # calculate error II row
    error2_temp = np.sum(cm, axis=0)
    error2 = []
    for i in range(0, len(error2_temp)):
        error2.append(error2_temp[i] - cm[i][i])
    error2.append(np.sum(error2))

    # add errors to matrix
    cm = np.append(cm, error1, axis=1)
    cm = np.append(cm, error2).reshape(11, 11)

    # plot confusion matrix as heat map and save as image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[])
    disp.plot(cmap=plt.cm.Reds)
    disp.ax_.xaxis.tick_top()
    disp.ax_.xaxis.set_label_position('top')
    disp.ax_.set_xticklabels(['a', 'c', 'e', 'm', 'n', 'o', 'r', 's', 'x', 'z', 'Error I'])
    disp.ax_.set_yticklabels(['a', 'c', 'e', 'm', 'n', 'o', 'r', 's', 'x', 'z', 'Error II'])
    disp.ax_.set_xlabel('Decision')
    disp.ax_.set_ylabel('Input')
    disp.ax_.set_title('Confusion Matrix for ' + runName)
    if save:
        plt.savefig(imgFile, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    # print(accuracy_score(eval1['class'], guesses))
    # print(recall_score(eval1['class'], guesses, average=None))
    # print(precision_score(eval1['class'], guesses, average=None))

    # write results to file
    if save:
        percentError = round(cm[10][10] / len(evalData) * 100, 2)
        file = open(txtFile, 'a')
        file.write('**** ' + runName + ' ****\n')
        file.write('Classification Results: ' + str(100-percentError) + '% correct, ' + str(percentError) + '% error.\n\n')
        file.close()

def getPercentError(evalData: pd.DataFrame, guesses: list):
    # create confusion matrix
    cm = confusion_matrix(evalData['class'], guesses)

    # calculate error I column
    error1 = []
    for i in range(0, len(cm)):
        error1.append([sum(cm[i]) - cm[i][i]])
    tot_err = np.sum(error1)
    return round(tot_err / len(evalData) * 100, 2)

def plotPoints(data: list, label: str, title: str, imgFile: str, save: bool = True):
    plt.clf()
    plt.scatter(list(range(len(data))), data, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.title(title)
    if save:
        plt.savefig(imgFile, bbox_inches='tight')
    else:
        plt.show()
    plt.close()