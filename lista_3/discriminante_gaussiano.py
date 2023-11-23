import numpy as np
import pandas as pd

from preparacao_tabela import *


def logaritmo(matriz):
    i = 0
    for values in matriz:
        j = 0
        for value in values:
            matriz[i][j] = math.log(value)
            j += 1
        i += 1
    return matriz


classes = [0, 1]
probabilidade_classes = []
media_classes = []
covariancia_classes = []

for classe in classes:
    qtd = pd.DataFrame(Y_treino)
    linhas = qtd['y']
    qtd_amostras_classe = qtd['y'].value_counts()[classe]
    probabilidade_classes.append(qtd_amostras_classe / ROWS_treino)
    somatorio_x_classe = 0

    for i in range(ROWS_treino):
        if int(Y_treino_normalizado[i]) == classe:
            somatorio_x_classe += sum(X_treino_normalizado[i])
    media_classes.append(somatorio_x_classe / qtd_amostras_classe)

    matriz_aux = np.ones((qtd_amostras_classe, COLS - 1))
    j = 0
    for i in range(ROWS_treino):
        if int(Y_treino_normalizado[i]) == classe:
            matriz_aux[j, :] = (X_treino_normalizado[i] - media_classes[classe])
            j += 1
    variancia = np.dot(matriz_aux, matriz_aux.T)
    variancia = variancia / (qtd_amostras_classe - 1)

    y_previsto_p1 = math.log(probabilidade_classes[classe])

    y_previsto_p2 = abs(variancia)
    y_previsto_p2 = logaritmo(y_previsto_p2)
    y_previsto_p2 = y_previsto_p2*(-0.5)

    y_previsto_p3 = (matriz_aux - media_classes[classe])
    y_previsto_p3 = np.dot(y_previsto_p3, y_previsto_p3.T)
    y_previsto_p3 = np.dot(y_previsto_p3, np.linalg.inv(variancia))

    y_previsto = y_previsto_p1 - 0.5 * y_previsto_p2 - 0.5 * y_previsto_p3
    print(np.argmax(y_previsto, axis=None), y_previsto.shape)
    print("---------------------------------------------------------------------")