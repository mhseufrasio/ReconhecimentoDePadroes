import math
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
from sklearn import preprocessing

def distancia_euclidiana(xlinha, xmatriz):
    distancias = []
    for linha in range(ROWS_treino):
        soma = 0
        for coluna in range(COLS-1):
            soma += (xlinha[coluna] - xmatriz[linha][coluna])**2
        distancias.append(math.sqrt(soma))
    return distancias