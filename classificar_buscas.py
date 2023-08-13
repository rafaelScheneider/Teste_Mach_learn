from dados import carregar_buscas
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from collections import Counter
 
 # teste inicial: home, busca, logado => comprou
 # home, busca
 # home, logado
 # busca, logado
 # busca: 85,71% (7 testes)
 
df = pd.read_csv('buscas.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.8
porcentagem_de_teste = 0.1

tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_teste = int(porcentagem_de_teste * len(Y))
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

# 0 até 799
treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0  :tamanho_de_treino]

# 800 até 899
fim_de_teste = tamanho_de_treino + tamanho_de_teste
teste_dados = X[tamanho_de_treino:fim_de_teste]
teste_marcacoes = Y[tamanho_de_treino:fim_de_teste]

# 900 até 999
validacao_dados = X[fim_de_teste:]
validacao_marcacoes = Y[fim_de_teste:]

modelo = AdaBoostClassifier()

def fit_and_predict(nome, modelo, treino_dados, treino_marcadores, teste_dados, teste_marcadores):
    modelo.fit(treino_dados, treino_marcadores)

    resultado = modelo.predict(teste_dados)
    acertos = resultado == teste_marcadores

    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
    msg = "Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_de_acerto)

    print(msg)
    
    return taxa_de_acerto
    
    
modeloMultinomial = MultinomialNB()
resultado_multinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

modelo_adaBoost = AdaBoostClassifier()
resultado_adaBoost = fit_and_predict("AdaBoostClassifier", modelo_adaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if resultado_multinomial > resultado_adaBoost:
    escolido = modeloMultinomial
else:
    escolido = modelo_adaBoost

resultado = escolido.predict(validacao_dados)
acertos = (resultado == validacao_marcacoes)
total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto)
print(msg)

# Dummy para ver se o algoritimo principal é melhor que um modelo "Burro"
acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

total_de_elementos = len(validacao_dados)
print("Total de teste: %d" % total_de_elementos)