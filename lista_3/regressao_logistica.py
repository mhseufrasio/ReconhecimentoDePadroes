from preparacao_tabela import *


def sigmoid(x):
    o = 1 / (1 + math.exp(-x))
    return o


# adicionando coluna de 1s em X
ones = np.ones(len(X_treino_normalizado), dtype=float)
ones = np.array(ones).reshape((-1, 1))
X_treino_normalizado = np.concatenate((ones, X_treino_normalizado), axis=1)

# calculando regressão
while t < T:
    for valor in range(len(X_treino_normalizado)):
        X = X_treino_normalizado[valor]
        y = Y_treino_normalizado[valor]
        y_previsto = sigmoid(np.dot(W.T, X))
        erro_quadratico = y - y_previsto
        custo += (-1 / ROWS_treino) * y * numpy.log(y_previsto) + (1 - y) * numpy.log(1 - y_previsto)
        W = W + 0.2 * erro_quadratico * X
    funcao_custo.append(custo)
    custo = 0.0
    t += 1

# adicionando coluna de 1s em X
ones = np.ones(len(X_teste_normalizado), dtype=float)
ones = np.array(ones).reshape((-1, 1))
X_teste_normalizado = np.concatenate((ones, X_teste_normalizado), axis=1)
W = np.array(W).reshape((-1, 1))
Y_previsto = []

for i in range(len(X_teste_normalizado)):
    y = sigmoid(np.dot(X_teste_normalizado[i], W))
    Y_previsto.append(round(y, 0))

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

# plot resultados e custo
espaco = np.linspace(0, 1)
plt.plot(funcao_custo)
plt.show()
plt.plot(Y_teste_normalizado, 'o')
plt.plot(Y_previsto, 'o')
plt.show()