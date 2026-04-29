import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import os
from save_load import *

def bar_plot(label, data1, data2, metric):
    # create data
    df = pd.DataFrame([data1, data2],
                      columns=label)
    df1 = pd.DataFrame()
    df1['Training percentage(%)'] = [70, 80]
    df = pd.concat((df1, df), axis=1)
    # plot grouped bar chart
    df.plot(x='Training percentage(%)',
            kind='bar',
            stacked=False)

    plt.ylabel(metric)
    plt.xticks(rotation=0)
    plt.legend(loc='center')

    if not os.path.exists('./Result_benchmarking_model'):
        os.makedirs('./Result_benchmarking_model')

    plt.savefig('./Result_benchmarking_model/' + metric + '.png', dpi=900)
    plt.show(block=False)


def polt_res():
    # SVM [17]
    SVM1 = [0.977847, 0.889236, 0.889236, 0.947693, 0.889236, 0.876929, 0.947693, 0.012307, 0.110764]

    # KNN [20]
    KNN1 = [0.966729, 0.833646, 0.833646, 0.951516, 0.833646, 0.815162, 0.951516, 0.018484, 0.166354]

    # RF [16]
    RF1 = [0.959109, 0.795544, 0.795544, 0.937283, 0.795544, 0.772827, 0.937283, 0.022717, 0.204456]

    # DTree [25]
    DTree1 = [0.943619, 0.718093, 0.718093, 0.968677, 0.718093, 0.68677, 0.968677, 0.031323, 0.281907]

    # ANN
    ANN1 = [0.975932, 0.879659, 0.879659, 0.966629, 0.879659, 0.866287, 0.966629, 0.013371, 0.120341]

    # PROPOSED (ULE)
    PROPOSED1 = [0.983177, 0.915886, 0.915886, 0.990654, 0.915886, 0.90654, 0.990654, 0.009346, 0.084114]
    # SVM [17]
    SVM2 =  [0.972885, 0.847932, 0.847932, 0.934623, 0.847932, 0.831702, 0.934623, 0.015377, 0.152068]

    # KNN [20]
    KNN2 = [0.970321, 0.842318, 0.842318, 0.943412, 0.842318, 0.824617, 0.943412, 0.016588, 0.157682]

    # RF [16]
    RF2 = [0.962347, 0.803251, 0.803251, 0.978591, 0.803251, 0.780303, 0.978591, 0.021409, 0.196749]

    # DTree [25]
    DTree2 = [0.948731, 0.725789, 0.725789, 0.970212, 0.725789, 0.693427, 0.970212, 0.029788, 0.274211]

    # ANN
    ANN2 = [0.978812, 0.885234, 0.885234, 0.958234, 0.885234, 0.870122, 0.958234, 0.011766, 0.114766]

    # PROPOSED (ULE)
    PROPOSED2 = [0.992879, 0.964397, 0.964397, 0.996044, 0.964397, 0.960441, 0.996044, 0.003956, 0.035603]

    dic1 = {'CNN[16]':SVM1, 'DNN[17]': KNN1, 'Autoencoder[19]': RF1, 'KNN_SVM[22]':DTree1, 'Proposed':PROPOSED1}
    dic2 = {'CNN[16]': SVM2, 'DNN[17]': KNN2, 'Autoencoder[19]': RF2, 'KNN_SVM[22]': DTree2, 'Proposed': PROPOSED2}

    import pandas as pd
    import numpy as np
    ind = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F_measure", "MCC", "NPV", "FPR", "FNR"]

    table1 = pd.DataFrame(dic1, index=ind)
    table2 = pd.DataFrame(dic2, index=ind)

    val1 = np.array(table1)
    val2 = np.array(table2)

    Metrices = [val1, val2]
    save('Metrices_existing_model', Metrices)
    Metrices = load('Metrices_existing_model')
    mthod = ['MMM-Net','TL-Deep EM','CNN','DCNN-TL', 'proposed ULE']
    metrices_plot = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F_measure", "MCC", "NPV", "FPR", "FNR"]

    # Bar plot
    for i in range(len(metrices_plot)):
        bar_plot(mthod, Metrices[0][i, :], Metrices[1][i, :], metrices_plot[i])

    for i in range(2):
        # Table
        print('Training percentage(%)- ' + str([i]))
        tab = pd.DataFrame(Metrices[i], index=metrices_plot, columns=mthod)
        print(tab)
        excel_file_path = './Result_benchmarking_model/Training percentage(%)' + str(i + 1) + '.xlsx'
        tab.to_excel(excel_file_path, index=metrices_plot)  # Specify index=False to exclude index column


polt_res()