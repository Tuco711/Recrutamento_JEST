# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree

load_data = pd.read_csv("C:\\Users\\arthu\\OneDrive - dei.uc.pt\\JEST\\ProjetoTI_part2-master\\Leukemya_data.csv",
                        header=None, index_col=None, delimiter=',')
labels = pd.read_csv("C:\\Users\\arthu\\OneDrive - dei.uc.pt\\JEST\\ProjetoTI_part2-master\\labels.csv", header=None,
                     index_col=None, delimiter=',')

lin, coluna = load_data.shape[0], load_data.shape[1]


# -------------------------------------------------- Outliers ----------------------------------------------------------
# Filtro
def outliers_filter(fator):
    soma = 0
    for col in range(0, coluna):
        mean = np.mean(load_data[:][col])
        desvio = np.std(load_data[:][col])

        limMax = mean + fator * desvio
        limMin = mean - fator * desvio

        outlierMax = np.where(load_data[:][col] >= limMax)[0]
        outlierMin = np.where(load_data[:][col] <= limMin)[0]

        load_data.loc[outlierMax, col] = limMax
        load_data.loc[outlierMin, col] = limMin

        soma = len(outlierMin) + len(outlierMax)

    print("Numero de outliers =", soma)


outliers_filter(2)

# --------------------------------------------- Matriz Correlação ------------------------------------------------------
plt.figure()
matrix = load_data.corr()
sns.heatmap(matrix)
plt.title("Matriz Correlação")
plt.show()

# -------------------------------------------- Feature Reduction -------------------------------------------------------
labels.to_numpy()
CCs = np.abs(load_data.corrwith(labels[:][0]))
CCs_idx = np.where(CCs > 0.6)[0]

CCs_idx.tolist()

data_treino = load_data.iloc[:128][CCs_idx]

# ----------------------------------------------- Modelização ----------------------------------------------------------
clf = tree.DecisionTreeClassifier()
clf.fit(data_treino, labels)
plt.figure()
tree.plot_tree(clf)
plt.title("Árvore de Decisão")
plt.show()
