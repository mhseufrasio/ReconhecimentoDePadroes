import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import *
from tratamento_tabela import *
from sklearn import svm


def povoar(n_vezes, inicio):
    lista = []
    for i in range(n_vezes):
        lista.append(2**inicio)
        inicio += 2
    return lista


# normalizando os X
escalar_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
escalar_x.fit(X_treino)
X_treino_normalizado = escalar_x.transform(X_treino)
X_teste_normalizado = escalar_x.transform(X_teste)

# normalizando os Y
escalar_y = preprocessing.MinMaxScaler(feature_range=(-1, 1))
escalar_y.fit(Y_treino)
Y_treino_normalizado = escalar_y.transform(Y_treino)
Y_teste_normalizado = escalar_y.transform(Y_teste)

C = povoar(11, -5)
Gama = povoar(11, -15)
print(C)
print(Gama)
parametro_C = []
parametro_Gama = []
accuracy = []
matrizes_confusao = []
Y_predito = []
K = 10
kfold = sk.KFold(n_splits=K)

for c in C:
    for gama in Gama:
        classificador = svm.SVC(C=c, gamma=gama, kernel='rbf')

        acuracia = []
        for treino, teste in kfold.split(X_treino_normalizado):
            y_treino_kfold = []
            y_teste_kfold = []
            x_treino_kfold = []
            x_teste_kfold = []

            for indice in treino:
                x_treino_kfold.append(X_treino_normalizado[indice])
                y_treino_kfold.append(Y_treino_normalizado[indice])

            for indice in teste:
                x_teste_kfold.append(X_treino_normalizado[indice])
                y_teste_kfold.append(Y_treino_normalizado[indice])

            classificador.fit(x_treino_kfold, y_treino_kfold)
            acuracia.append(classificador.score(x_teste_kfold, y_teste_kfold))
            y_predito_kfold = classificador.predict(x_teste_kfold)
            vp, vn, fp, fn = matrix_confusao(y_teste_kfold, y_predito_kfold)

        acuracia_media = sum(acuracia) / K
        classificador.fit(X_treino_normalizado, Y_treino_normalizado)

        Y_predito.append(classificador.predict(X_teste_normalizado))

        vp, vn, fp, fn = matrix_confusao(Y_teste_normalizado, Y_predito[-1])

        parametro_C.append(c)
        accuracy.append(acuracia_media)
        parametro_Gama.append(gama)
        matrizes_confusao.append([vp, vn, fp, fn])

# procurar a maior acuracia
maior_acuracia = max(accuracy)
indice = accuracy.index(maior_acuracia)

# matriz confusão
vp = matrizes_confusao[indice][0]
vn = matrizes_confusao[indice][1]
fp = matrizes_confusao[indice][2]
fn = matrizes_confusao[indice][3]

acuracia = (vp + vn) / (vp + vn + fp + fn)
precisao = vp / (vp + fp)
revocacao = vp / (vp + fn)
f1_score = 2 * (precisao * revocacao) / (precisao + revocacao)

print("Verdadeiros positivos: ", vp)
print("Verdadeiros negativos: ", vn)
print("Falsos positivos: ", fp)
print("Falsos negativos: ", fn)
print("----------------------------------------------------------------")
print("Acurácia: ", acuracia)
print("Precisão: ", precisao)
print("Revocação: ", revocacao)
print("F1_Score: ", f1_score)
print("----------------------------------------------------------------")
print("C: ", parametro_C[indice])
print("Gama: ", parametro_Gama[indice])

# curva PRC
y_proba = classificador.decision_function(X_teste_normalizado)
precision, recall, thresholds = precision_recall_curve(Y_teste_normalizado, y_proba)
plt.plot(recall, precision)
plt.xlabel('revocação')
plt.ylabel('precisão')
plt.show()

# curva ROC
svc_disp = RocCurveDisplay.from_estimator(classificador, X_teste_normalizado, Y_teste_normalizado)
plt.show()
