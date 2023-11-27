import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from random import *

# leitura dos dados
tabela = pd.read_csv("penguins.csv", sep=",")
tabela = pd.DataFrame.to_numpy(tabela)
shuffle(tabela)

# inicialização de variáveis
COLS = tabela.shape[1]
ROWS = len(tabela)

# divisão da tabela em X e Y
divisao_tabela = np.hsplit(tabela, np.array([COLS - 1]))
X = divisao_tabela[0]
Y = divisao_tabela[1]

# normalizando X
escalar_x = preprocessing.MinMaxScaler(feature_range=(0, 1))
escalar_x.fit(X)
X_normalizado = escalar_x.transform(X)

# normalizando Y
escalar_y = preprocessing.MinMaxScaler(feature_range=(0, 1))
escalar_y.fit(Y)
Y_normalizado = escalar_y.transform(Y)

matriz_covariancia = np.cov(X_normalizado)
autovalores, autovetores = np.linalg.eig(matriz_covariancia)
print(autovalores)
print("------------------------------------------------------")
print(autovetores)
a = 0
