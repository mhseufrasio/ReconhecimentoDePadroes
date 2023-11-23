import math
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
from sklearn import preprocessing


def matrix_confusao(Y_teste, Y_previsto):
    fp = 0
    fn = 0
    for i in range(len(Y_teste)):
        if (Y_teste[i] - Y_previsto[i]) == 1:
            fn += 1
        elif (Y_teste[i] - Y_previsto[i]) == -1:
            fp += 1

    vp = sum(Y_teste) - fn
    vn = len(Y_teste) - sum(Y_teste) - fp

    return vp, vn, fp, fn


# leitura dos dados
tabela = pd.read_csv("breast.csv", sep=",")

# iniciação da variaveis
ROWS = len(tabela)
COLS = tabela.shape[1]
W = np.ones(COLS)
t = 0
T = 100
custo = 0.0
funcao_custo = []
K = 3

# dividindo tabela em X e Y
divisao_tabela = np.hsplit(tabela, np.array([COLS - 1]))
X = divisao_tabela[0]
Y = divisao_tabela[1]

# dividindo X e Y em treino e teste
X_treino, X_teste, Y_treino, Y_teste = sk.train_test_split(X, Y, test_size=0.2)

ROWS_treino = len(X_treino)
ROWS_teste = len(X_teste)
Y_previsto = []

# normalizando os X
escalar_x = preprocessing.MinMaxScaler(feature_range=(0, 1))
escalar_x.fit(X_treino)
X_treino_normalizado = escalar_x.transform(X_treino)
X_teste_normalizado = escalar_x.transform(X_teste)

# normalizando os Y
escalar_y = preprocessing.MinMaxScaler(feature_range=(0, 1))
escalar_y.fit(Y_treino)
Y_treino_normalizado = escalar_y.transform(Y_treino)
Y_teste_normalizado = escalar_y.transform(Y_teste)
