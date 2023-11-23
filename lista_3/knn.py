from preparacao_tabela import *


def distancia_euclidiana(xlinha, xmatriz):
    distancias = []
    for linha in range(ROWS_treino):
        soma = 0
        for coluna in range(COLS-1):
            soma += (xlinha[coluna] - xmatriz[linha][coluna])**2
        distancias.append(math.sqrt(soma))
    return distancias


Y_teste_normalizado = np.array(Y_teste_normalizado).reshape(-1, 1)

for i in range(ROWS_teste):
    distancia = distancia_euclidiana(X_teste_normalizado[i], X_treino_normalizado)
    dist_organizada = sorted(distancia)

    id1 = distancia.index(dist_organizada[0])
    id2 = distancia.index(dist_organizada[1])
    id3 = distancia.index(dist_organizada[2])

    y1 = Y_treino_normalizado[id1]
    y2 = Y_treino_normalizado[id2]
    y3 = Y_treino_normalizado[id3]
    y = (y1 + y2 + y3) / K
    Y_previsto.append(round(int(y), 0))

vp, vn, fp, fn = matrix_confusao(Y_teste_normalizado, Y_previsto)

acuracia = (vp+vn)/(vp+vn+fp+fn)
precisao = vp/(vp+fp)
revocacao = vp/(vp+fn)
f1_score = 2*(precisao*revocacao)/(precisao+revocacao)

print("Verdadeiros positivos: ", vp)
print("Verdadeiros negativos: ", vn)
print("Falsos positivos: ", fp)
print("Falsos negativos: ", fn)
print("----------------------------------------------------------------")
print("Acurácia: ", acuracia)
print("Precisão: ", precisao)
print("Revocação: ", revocacao)
print("F1_Score: ", f1_score)

# plot reta alterada
espaco = np.linspace(0, 1)
plt.plot(Y_teste_normalizado, 'o')
plt.plot(Y_previsto, 'o')
plt.show()