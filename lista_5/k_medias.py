import copy
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from random import *


def indice_de(matriz, valor, indice_final):
    for indice in range(indice_final):
        aux_x = matriz[indice][0]
        aux_y = matriz[indice][1]
        valor_x = valor[0][0]
        valor_y = valor[0][1]
        if aux_x == valor_x and aux_y == valor_y:
            return indice
    return 0


def indice_davies_bouldin(centroides, clusters, distancias):
    max_indices_DB = []
    for indice_cluster_comparativo in range(len(clusters)-1):
        if len(clusters[indice_cluster_comparativo]) != 0:
            espalhamento_intra = sum(distancias[indice_cluster_comparativo]) / (len(clusters[indice_cluster_comparativo]))
        else:
            espalhamento_intra = 1000
        indices_DB = []
        for indice_cluster_comparado in range(indice_cluster_comparativo+1, len(clusters)):
            if len(clusters[indice_cluster_comparado]) != 0:
                espalhamento_intra_comparado = sum(distancias[indice_cluster_comparado]) / (
                    len(clusters[indice_cluster_comparado]))
            else:
                espalhamento_intra_comparado = 1000
            soma_x = (centroides[indice_cluster_comparativo][0] - centroides[indice_cluster_comparado][0]) ** 2
            soma_y = (centroides[indice_cluster_comparativo][1] - centroides[indice_cluster_comparado][1]) ** 2
            espalhamento_entre_grupos = (math.sqrt(soma_x + soma_y))
            indices_DB.append((espalhamento_intra+espalhamento_intra_comparado)/espalhamento_entre_grupos)
        max_indices_DB.append(max(indices_DB)/len(clusters))
    indice_DB = sum(max_indices_DB)/(len(clusters))
    return indice_DB




# calcula as distâncias do cluster para um centroide
def distancias_euclidianas(centroide, matriz):
    distancias = []
    for linha in matriz:
        soma_x = (linha[0] - centroide[0]) ** 2
        soma_y = (linha[1] - centroide[1]) ** 2
        distancias.append(math.sqrt(soma_x + soma_y))
    return distancias


# calcula as distancias dos clusters para os centroides e o erro de reconstrução
def calcular_distancias_e_erro_reconstrucao(lista_centroides, lista_clusters):
    distancias_clusters = []
    for cluster in lista_clusters:
        for centroide in lista_centroides:
            abc = distancias_euclidianas(centroide, cluster)
            distancias_clusters.append(abc)
            if len(cluster) == 0:
                a = 0
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


def reagrupar_clusters(distancias, clusters, centroides):
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
    novos_clusters = copy.deepcopy(clusters)
    for indice in range(qtd_clusters):
        distancias_centroide = tuple(zip(*divisao[indice]))
        indice_auxiliar = 0
        for comparacao in distancias_centroide:
            minimo = min(comparacao)
            indice_minimo = comparacao.index(minimo)
            if indice_minimo != indice:
                aux = np.array(clusters[indice][indice_auxiliar]).reshape(-1, 1)
                aux = np.array(aux).reshape(1, -1)
                novos_clusters[indice] = np.delete(novos_clusters[indice], indice_de(novos_clusters[indice], aux, indice_auxiliar), 0)
                novos_clusters[indice_minimo] = np.append(novos_clusters[indice_minimo], aux, axis=0)
            indice_auxiliar += 1
    centroides = novos_centroides(qtd_clusters, novos_clusters, centroides)
    return centroides, novos_clusters


def novos_centroides(qtd_clusters, clusters, centroides_antigos):
    centroides = []
    for indice in range(qtd_clusters):
        somatorio = sum(clusters[indice])
        if len(clusters[indice]) != 0:
            centroides.append(somatorio/(len(clusters[indice])))
        else:
            centroides.append(centroides_antigos[indice])
    return centroides


def plotar_clusters_e_centroides(clusters, centroides):
    for indice in range(len(clusters)):
        plt.plot(clusters[indice][:, 0], clusters[indice][:, 1], "o")
        plt.plot(centroides[indice][0], centroides[indice][1], '8', markeredgewidth=5, markersize=15)
    plt.show()


# leitura dos dados
tabela = pd.read_csv("quake.csv", sep=",")
tabela = pd.DataFrame.to_numpy(tabela)
shuffle(tabela)

# inicialização de variáveis
ROWS = len(tabela)
COLS = tabela.shape[1]
distancias = []
indices_DB = []
lista_de_clusters = []
lista_de_centroides = []

# dividindo tabela em X e Y
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

# unindo dados normalizados
dados_normalizado_tabela = np.concatenate((X_normalizado, Y_normalizado), axis=1)

for i in range(4, 20):

    centroides = criar_centroides(i)
    clusters = dividir_clusters(i, dados_normalizado_tabela)
    for indice in range(20):
        distancias, erro = calcular_distancias_e_erro_reconstrucao(centroides, clusters)
        centroides, clusters = reagrupar_clusters(distancias, clusters, centroides)
    indices_DB.append(indice_davies_bouldin(centroides, clusters, distancias))
    lista_de_clusters.append(clusters)
    lista_de_centroides.append(centroides)
minimo_idice_DB = min(indices_DB)
qtd_clusters_minimo = indices_DB.index(minimo_idice_DB) + 4
plotar_clusters_e_centroides(lista_de_clusters[qtd_clusters_minimo-4], lista_de_centroides[qtd_clusters_minimo-4])
print("Quantidade de clusters com melhor resolução: ", qtd_clusters_minimo)
print("Índice de Davies Bouldin: ", minimo_idice_DB)
