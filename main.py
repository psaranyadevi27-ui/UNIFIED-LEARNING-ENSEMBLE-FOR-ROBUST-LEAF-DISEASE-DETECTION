import os
os.makedirs('./Results', exist_ok=True)
os.makedirs('./Saved data', exist_ok=True)
os.makedirs('./Pictorial Results', exist_ok=True)
from datagen import datagen
from Objective_function import objective_func_70, objective_func_80
from save_load import load, save
import numpy as np
from HFWOA import hfwoa
from classification import UnifiedLearningEnsemble, SVM, KNN, RF, ANN, DTree
from plot_result import plotres
import matplotlib.pyplot as plt


def full_analysis():
    #datagen()
    # learning rate 70
    x_train_70 = load('x_train_70')
    x_test_70 = load('x_test_70')
    y_train_70 = load('y_train_70')
    y_test_70 = load('y_test_70')
    # learning rate 80
    x_train_80 = load('x_train_80')
    x_test_80 = load('x_test_80')
    y_train_80 = load('y_train_80')
    y_test_80 = load('y_test_80')

    learn_data = [(x_train_70, y_train_70, x_test_70, y_test_70, objective_func_70),
                  (x_train_80, y_train_80, x_test_80, y_test_80, objective_func_80)]
    j = 70
    for i in learn_data:
        lb = np.zeros(i[0].shape[1])
        ub = np.ones(i[0].shape[1])
        pop_size = 6
        prob_size = len(lb)
        epochs = 100
        best_solution = hfwoa(i[-1], lb, ub, pop_size, prob_size, epochs)
        save('best_solution_' + str(j), best_solution)

        soln = np.round(best_solution)
        selected_indices = np.where(soln == 1)[0]

        x_train = i[0][:, selected_indices]
        y_train = i[1]
        x_test = i[2][:, selected_indices]
        y_test = i[3]

        pred, met = UnifiedLearningEnsemble(x_train, y_train, x_test, y_test)
        save('proposed_' + str(j), met)

        pred, met = SVM(i[0], i[1], i[2], i[3])  # x_train, y_train, x_test, y_test
        save('svm[17]_' + str(j), met)

        pred, met = KNN(i[0], i[1], i[2], i[3])
        save('knn[20]_' + str(j), met)

        pred, met = RF(i[0], i[1], i[2], i[3])
        save('rf[16]_' + str(j), met)

        pred, met = ANN(i[0], i[1], i[2], i[3])
        save('ann_' + str(j), met)

        pred, met = DTree(i[0], i[1], i[2], i[3])
        save('dtree[25]_'+ str(j), met)

        j = 80


a = 0
if a == 0:
    full_analysis()

plotres()
plt.show()