import matplotlib.pyplot as plt
import numpy as np
from save_load import load, save
import pandas as pd

def bar_plot(label, data1, data2, metric):

    # create data
    df = pd.DataFrame([data1, data2],
                      columns=label)
    df1 = pd.DataFrame()
    df1['Learning Rate (%)'] = [70, 80]
    df = pd.concat((df1, df), axis=1)
    # plot grouped bar chart
    df.plot(x='Learning Rate (%)',
            kind='bar',
            stacked=False)


    plt.ylabel(metric)
    plt.legend(loc='lower left')
    plt.savefig('./Results/'+metric+'.png', dpi=400)
    plt.show(block=False)

def plotres():

    # 80, 20 variation
    svm_80 = load('svm[17]_80')
    knn_80 = load('knn[20]_80')
    rf_80 = load('rf[16]_80')
    dtree_80 = load('dtree[25]_80')
    ann_80 = load('ann_80')
    proposed_80 = load('proposed_80')

    data = {
        'SVM[17]': svm_80,
        'KNN[20]': knn_80,
        'RF[16]': rf_80,
        'DTree[25]': dtree_80,
        'ANN' : ann_80,
        'ULE': proposed_80
    }

    ind = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']
    table = pd.DataFrame(data, index=ind)
    save('table1', table)
    tab = table.to_excel('./Results/table_80.xlsx')

    val1 = np.array(table)

    # learn rate 70, 30
    svm_70 = load('svm[17]_70')
    knn_70 = load('knn[20]_70')
    rf_70 = load('rf[16]_70')
    dtree_70 = load('dtree[25]_70')
    ann_70 = load('ann_70')
    proposed_70 = load('proposed_70')

    data1 = {
        'SVM[17]': svm_70,
        'KNN[20]': knn_70,
        'RF[16]': rf_70,
        'DTree[25]': dtree_70,
        'ANN': ann_70,
        'ULE': proposed_70
    }


    ind = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']
    table1 = pd.DataFrame(data1, index=ind)
    save('table2', table1)
    tab = table1.to_excel('./Results/table_70.xlsx')

    val2 = np.array(table1)

    method = ["SVM[17]", "KNN[20]", "RF[16]", "DTree[25]", "ANN", "ULE"]
    metrices_plot = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']
    metrices = [val2, val1]
    save('met', metrices)

    for i in range(len(metrices_plot)):
        bar_plot(method, metrices[0][i, :], metrices[1][i, :],
                 metrices_plot[i])

    for i in range(2):
        print('Metrices-Dataset--' + str(i + 1))
        tab = pd.DataFrame(metrices[i], index=metrices_plot, columns=method)
        print(tab)


