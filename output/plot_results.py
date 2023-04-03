import numpy as np
import matplotlib.pyplot as plt
import pathlib
from scipy.stats import ranksums

# First extract the final result matrix from each cv_res .txt file
def read_table_res(path):
    file = open(path)
    lines = file.readlines()
    file.close()
    res = np.eye(len(lines), 4)
    i = 0

    for l in lines:
        l = l.split()
        res[i][0] = np.float32(l[0])
        res[i][1] = np.float32(l[1])
        res[i][2] = np.float32(l[2])
        res[i][3] = np.float32(l[3])
        i += 1

    return res


#def print_wilcoxon_results(tb, name, *kwargs):



def print_average_res():
    path = str(pathlib.Path(__file__).parent.resolve())
    xgb = read_table_res(path + "/xgb_table.txt")
    svm = read_table_res(path + "/svm_table.txt")
    nn = read_table_res(path + "/nn_table.txt")
    # nb = read_table_res(path + "/nb_table.txt")
    # cl = read_table_res(path + "/cl_table.txt")
    # hc = read_table_res(path + "/hc_table.txt")
    hcsp = read_table_res(path + "/hcsp_table.txt")

    print('XGB results:')
    print(np.mean(xgb, axis=0))
    print('SVM results:')
    print(np.mean(svm, axis=0))
    print('NN results:')
    print(np.mean(nn, axis=0))
    print('HCSP results:')
    print(np.mean(hcsp, axis=0))

    print('Wilcoxon rank-sum test results:')
    print_wilcoxon_results(xgb, "XGB", )


def plot_column(col, label, name):
    plt.figure()
    plt.plot(np.arange(0, 11*4, 4), col, label='Accuracy')  # 11 prediction horizon, 4 variables (acc, f1, gmean, time)
    plt.title(name)
    plt.ylabel(label)
    plt.xlabel('Time (h)')
    plt.show()


def create_plot_column(col, label, name):
    plt.plot(np.arange(0, 11*4, 4), col, label='Accuracy')
    plt.title(name)
    plt.ylabel(label)
    plt.xlabel('Time (h)')


def create_named_plot(col, label, name):
    plt.plot(np.arange(0, 11*4, 4), col, label=name)
    plt.ylabel(label, fontsize=30)
    plt.yticks(fontsize=20)
    plt.xlabel('Time (h)', fontsize=30)
    plt.xticks(fontsize=20)


def plot_table_res(path, name):
    table = read_table_res(path)

    plot_column(table[:, 0], 'Accuracy', name)
    #plot_column(table[:, 1], 'F1 Score', name)
    plot_column(table[:, 2], 'g-mean Score', name)
    plot_column(table[:, 3], 'Pred. time', name)


def plot_all_in_one():
    path = str(pathlib.Path(__file__).parent.resolve())
    xgb = read_table_res(path + "/xgb_table.txt")
    svm = read_table_res(path + "/svm_table.txt")
    nn = read_table_res(path + "/nn_table.txt")
    # nb = read_table_res(path + "/nb_table.txt")
    cl = read_table_res(path + "/cl_table.txt")
    hc = read_table_res(path + "/hc_table.txt")
    hcsp = read_table_res(path + "/hcsp_table.txt")

    plt.subplot(1, 2, 1)
    create_named_plot(xgb[:, 0], 'Accuracy', "XGBoost")
    create_named_plot(svm[:, 0], 'Accuracy', "SVM")
    create_named_plot(nn[:, 0], 'Accuracy', "Neural Network")
    # create_named_plot(nb[:, 0], 'Accuracy', "NB")
    create_named_plot(cl[:, 0], 'Accuracy', "CL")
    create_named_plot(hc[:, 0], 'Accuracy', "HC")
    create_named_plot(hcsp[:, 0], 'Accuracy', "HCSP")
    plt.legend(loc="upper right", fontsize=12)

    # plt.subplot(1, 3, 2)
    # create_named_plot(xgb[:, 1], 'F1 score', "XGBoost")
    # create_named_plot(svm[:, 1], 'F1 score', "SVM")
    # create_named_plot(nn[:, 1], 'F1 score', "Neural Network")
    # # create_named_plot(nb[:, 1], 'F1 score', "NB")
    # create_named_plot(cl[:, 1], 'F1 score', "CL")
    # create_named_plot(hc[:, 1], 'F1 score', "HC")
    # create_named_plot(hcsp[:, 1], 'F1 score', "HCSP")
    # plt.legend(loc="upper right", fontsize=12)

    plt.subplot(1, 2, 2)
    create_named_plot(xgb[:, 2], 'g-mean', "XGBoost")
    create_named_plot(svm[:, 2], 'g-mean', "SVM")
    create_named_plot(nn[:, 2], 'g-mean', "Neural Network")
    # create_named_plot(nb[:, 2], 'g-mean', "NB")
    create_named_plot(cl[:, 2], 'g-mean', "CL")
    create_named_plot(hc[:, 2], 'g-mean', "HC")
    create_named_plot(hcsp[:, 2], 'g-mean', "HCSP")
    plt.legend(loc="upper right", fontsize=12)

    plt.show()


def plot_one_model(file="/nn_table.txt", model="Neural Network"):
    path = str(pathlib.Path(__file__).parent.resolve())
    table = read_table_res(path + file)

    plt.subplot(1, 2, 1)
    create_named_plot(table[:, 0], 'Accuracy', model)
    #plt.legend(loc="upper right", fontsize=12)

    plt.subplot(1, 2, 2)
    create_named_plot(table[:, 2], 'g-mean', model)
    #plt.legend(loc="upper right", fontsize=12)

    plt.show()


def plot_all_in_sub():
    path = str(pathlib.Path(__file__).parent.resolve())
    xgb = read_table_res(path + "/xgb_table.txt")
    nb = read_table_res(path + "/nb_table.txt")
    cl = read_table_res(path + "/cl_table.txt")
    hc = read_table_res(path + "/hc_table.txt")
    hcsp = read_table_res(path + "/hcsp_table.txt")

    plt.subplot(5, 2, 1)
    create_plot_column(xgb[:, 0], 'Accuracy', "XGBoost")
    plt.subplot(5, 2, 2)
    create_plot_column(xgb[:, 1], 'F1-score', "XGBoost")
    plt.subplot(5, 2, 3)
    create_plot_column(nb[:, 0], 'Accuracy', "Naive Bayes")
    plt.subplot(5, 2, 4)
    create_plot_column(nb[:, 1], 'F1-score', "Naive Bayes")
    plt.subplot(5, 2, 5)
    create_plot_column(cl[:, 0], 'Accuracy', "Chow-Liu")
    plt.subplot(5, 2, 6)
    create_plot_column(cl[:, 1], 'F1-score', "Chow-Liu")
    plt.subplot(5, 2, 7)
    create_plot_column(hc[:, 0], 'Accuracy', "Hill Climbing")
    plt.subplot(5, 2, 8)
    create_plot_column(hc[:, 1], 'F1-score', "Hill Climbing")
    plt.subplot(5, 2, 9)
    create_plot_column(hcsp[:, 0], 'Accuracy', "Hill Climbing Super-parent")
    plt.subplot(5, 2, 10)
    create_plot_column(hcsp[:, 1], 'F1-score', "Hill Climbing Super-parent")
    plt.show()


print_average_res()
plot_one_model()
#plot_all_in_one()