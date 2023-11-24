import math
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
from sklearn import preprocessing
from random import *


# calcula as distâncias do cluster para um centroide
def distancias_euclidianas(centroide, matriz):
    distancias = []
    for linha in matriz:
        soma = 0
        soma = (linha[0] - centroide[0]) ** 2
        distancias.append((math.sqrt(soma)) ** 2)
    return distancias


# calcula as distancias dos clusters para os centroides e o erro de reconstrução
def calcular_distancias_e_erro_reconstrucao(lista_centroides, lista_clusters):
    distancias_clusters = []
    for cluster in lista_clusters:
        for centroide in lista_centroides:
            distancias_clusters.append(distancias_euclidianas(centroide, cluster))
    erro = somar_distancias(distancias_clusters)
    return distancias_clusters, erro


def somar_distancias(distancias):
    soma = 0
    for distancia in distancias:
        soma += sum(distancia)
    return soma


def criar_centroides(n_centroides):
    lista_centroides = []
    for indice in range(n_centroides):
        x_centroide = random()
        y_centroide = random()
        lista_centroides.append([x_centroide, y_centroide])
    return lista_centroides


def dividir_clusters(n_clusters, matriz):
    ROWS = len(matriz)
    clusters = []
    rows_clusters = (int)(ROWS / n_clusters)
    for indice in range(n_clusters - 1):
        divisao = np.vsplit(matriz, np.array([rows_clusters]))
        clusters.append(divisao[0])
        matriz = divisao[1]
    clusters.append(divisao[1])
    return clusters


def reagrupar_clusters(distancias, clusters):
    n_clusters = len(clusters)
    divisao = []
    n_distancias = len(distancias)

    # código da internet
    for indice in range(n_clusters):
        start = int(indice * n_distancias / n_clusters)
        end = int((indice + 1) * n_distancias / n_clusters)
        divisao.append(distancias[start:end])
    # ----------------------------------------
    # //div/h2[.='Consultas']/../button
    # //div[@id='Consultasbh-header']/..//table//th/button[2]

    qtd_clusters = len(clusters)
    novos_clusters = clusters
    for indice in range(qtd_clusters):
        distancias_centroide = tuple(zip(*divisao[indice]))
        indice_auxiliar = 0
        for comparacao in distancias_centroide:
            minimo = min(comparacao)
            indice_minimo = comparacao.index(minimo)
            if indice_minimo != indice:
                aux = clusters[indice][indice_auxiliar]
                np.delete(novos_clusters[indice], indice_auxiliar)
                np.append(novos_clusters[indice_minimo], aux)
            indice_auxiliar += 1
    centroides = novos_centroides(qtd_clusters, novos_clusters)
    return centroides, novos_clusters

def novos_centroides(qtd_clusters, clusters):
    centroides = []
    for indice in range(qtd_clusters):
        somatorio = sum(clusters[indice])
        centroides.append(somatorio/(len(clusters[indice])))
    return centroides





# leitura dos dados
tabela = pd.read_csv("quake.csv", sep=",")
tabela = pd.DataFrame.to_numpy(tabela)
shuffle(tabela)

# inicialização de variáveis
ROWS = len(tabela)
COLS = tabela.shape[1]

# normalizando os dados
escalar = preprocessing.MinMaxScaler(feature_range=(0, 1))
escalar.fit(tabela)
dados_normalizado_tabela = escalar.transform(tabela)

for i in range(4, 20):
    centroides = criar_centroides(i)
    clusters = dividir_clusters(i, dados_normalizado_tabela)
    for indice in range(20):
        distancias, erro = calcular_distancias_e_erro_reconstrucao(centroides, clusters)
        centroides, clusters = reagrupar_clusters(distancias, clusters)
        plt.plot(centroides[0][0], centroides[0][1], 'o')
        plt.show()
    print(erro)