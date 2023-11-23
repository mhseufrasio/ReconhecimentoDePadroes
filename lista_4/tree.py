import matplotlib.pyplot as plt
from sklearn.metrics import *
from tratamento_tabela import *
from sklearn import tree

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

alturas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
min_nos_folha = [1, 2, 3, 4, 5]
parametro_altura = []
parametro_folha = []
parametro_erros = []
matrizes_confusao = []
arvores = []
Y_predito = []
K = 10
kfold = sk.KFold(n_splits=K)

for altura in alturas:
    for minimo in min_nos_folha:
        classificador = tree.DecisionTreeClassifier(max_depth=altura, min_samples_leaf=minimo)

        erros_kfold = []
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

            arvore = classificador.fit(x_treino_kfold, y_treino_kfold)
            y_predito_kfold = arvore.predict(x_teste_kfold)
            vp, vn, fp, fn = matrix_confusao(y_teste_kfold, y_predito_kfold)
            erros_kfold.append(fp + fn)

        erro_medio = (sum(erros_kfold)) / K
        arvore = classificador.fit(X_treino_normalizado, Y_treino)

        Y_predito.append(arvore.predict(X_teste_normalizado))

        vp, vn, fp, fn = matrix_confusao(Y_teste_normalizado, Y_predito[-1])

        parametro_altura.append(altura)
        parametro_folha.append(minimo)
        arvores.append(arvore)
        parametro_erros.append(erro_medio)
        matrizes_confusao.append([vp, vn, fp, fn])

# procurar o menor erro
menor_erro = min(parametro_erros)
indice = parametro_erros.index(menor_erro)

tree.plot_tree(arvores[indice])
plt.show()

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
print("Altura: ", parametro_altura[indice])
print("Nº mínimo de amostras por folhas: ", parametro_folha[indice])

y_pred_proba = arvores[indice].predict_proba(X_teste_normalizado)[:, 1]
# curva Precision-recall
## quão bem o algoritmo consegue classificar as amostras
precision, recall, thresholds = precision_recall_curve(Y_teste_normalizado, y_pred_proba)
plt.plot(recall, precision, label='Árvore de Decisão - Precision-Recall Curve')
plt.xlabel('revocação')
plt.ylabel('precisão')
plt.show()

# curva ROC
taxa_falso_positivo, taxa_negativo_verdadeiro, thresholds = roc_curve(Y_teste_normalizado, y_pred_proba)
plt.plot(taxa_falso_positivo, taxa_negativo_verdadeiro, label='Árvore de Decisão - ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Falsos Positivos')
plt.ylabel('Negativos Verdadeiros')
plt.show()
