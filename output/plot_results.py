import numpy as np
import matplotlib.pyplot as plt


def read_table_res(path):
    file = open(path)
    lines = file.readlines()
    file.close()
    res = np.eye(len(lines), 3)
    i = 0

    for l in lines:
        l = l.split(' ')
        res[i][0] = np.float32(l[0])
        res[i][1] = np.float32(l[1])
        res[i][2] = np.float32(l[2])
        i += 1

    return res


def plot_column(col, label, name):
    plt.figure()
    plt.plot(np.arange(0, 21*4, 4), col, label='Accuracy')
    plt.title(name)
    plt.ylabel(label)
    plt.xlabel('Time (h)')
    plt.show()

def create_plot_column(col, label, name):
    plt.plot(np.arange(0, 21*4, 4), col, label='Accuracy')
    plt.title(name)
    plt.ylabel(label)
    plt.xlabel('Time (h)')

def create_named_plot(col, label, name):
    plt.plot(np.arange(0, 21*4, 4), col, label=name)
    plt.ylabel(label, fontsize=20)
    plt.xlabel('Time (h)', fontsize=20)

def plot_table_res(path, name):
    table = read_table_res(path)

    plot_column(table[:, 0], 'Accuracy', name)
    plot_column(table[:, 1], 'F1 Score', name)
    plot_column(table[:, 2], 'Pred. time', name)

def plot_all_in_one():
    xgb = read_table_res("../../covid_bbva/xgboost-dbn/XGBoost-DBN/output/xgb_table.txt")
    nb = read_table_res("../../covid_bbva/xgboost-dbn/XGBoost-DBN/output/nb_table.txt")
    cl = read_table_res("../../covid_bbva/xgboost-dbn/XGBoost-DBN/output/cl_table.txt")
    hc = read_table_res("../../covid_bbva/xgboost-dbn/XGBoost-DBN/output/hc_table.txt")
    hcsp = read_table_res("../../covid_bbva/xgboost-dbn/XGBoost-DBN/output/hcsp_table.txt")

    plt.subplot(1, 2, 1)
    create_named_plot(xgb[:, 0], 'Accuracy', "XGBoost")
    create_named_plot(nb[:, 0], 'Accuracy', "NB")
    create_named_plot(cl[:, 0], 'Accuracy', "CL")
    create_named_plot(hc[:, 0], 'Accuracy', "HC")
    create_named_plot(hcsp[:, 0], 'Accuracy', "HCSP")
    plt.legend(loc="upper right", fontsize=12)

    plt.subplot(1, 2, 2)
    create_named_plot(xgb[:, 1], 'F1-score', "XGBoost")
    create_named_plot(nb[:, 1], 'F1-score', "NB")
    create_named_plot(cl[:, 1], 'F1-score', "CL")
    create_named_plot(hc[:, 1], 'F1-score', "HC")
    create_named_plot(hcsp[:, 1], 'F1-score', "HCSP")
    plt.legend(loc="upper right", fontsize=12)

    plt.show()

def plot_all_in_sub():
    xgb = read_table_res("../../covid_bbva/xgboost-dbn/XGBoost-DBN/output/xgb_table.txt")
    nb = read_table_res("../../covid_bbva/xgboost-dbn/XGBoost-DBN/output/nb_table.txt")
    cl = read_table_res("../../covid_bbva/xgboost-dbn/XGBoost-DBN/output/cl_table.txt")
    hc = read_table_res("../../covid_bbva/xgboost-dbn/XGBoost-DBN/output/hc_table.txt")
    hcsp = read_table_res("../../covid_bbva/xgboost-dbn/XGBoost-DBN/output/hcsp_table.txt")

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

