import numpy as np
import pandas as pd

testeCego = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade']
treino = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade', 'classe']

arqTeste = '01_treino_sinais_vitais_sem_label.txt'      # com coluna classe
arqTreino = '02_treino_sinais_vitais_sem_label.txt'     # sem coluna classe

features = ['qPA', 'pulso', 'resp']
faixas = [25.0, 50.0, 75.0]

def carregarTreino():
    arq = pd.read_csv(arqTreino, names=treino, header = None)
    X = arq[features].to_numpy(dtype=np.float)
    y_reg = arq['gravidade'].to_numpy(dtype=np.float)
    y_clf = arq['classe'].to_numpy(dtype=np.int)

    return X, y_reg, y_clf

def carregarCego():
    arq = pd.read_csv(arqTeste, names=testeCego, header = None)
    ids = arq['id'].to_numpy()
    X = arq[features].to_numpy(dtype=np.float)

    return X, ids

def gravidadeParaClasse(g): # Converte gravidade em classe 
    g = np.array(g, dtype = float)
    classe =  np.ones(g.shape, dtype= int)  # Cria vetor do mesmo tamanho de g preenchido com 1
    classe[g > faixas[0]] = 2               
    classe[g > faixas[1]] = 3
    classe[g > faixas[2]] = 4

    return classe

def dividirTreinoTeste(X, y, propTeste = 0.3, seed = 42, estratificar = None):      # Divisão de treino e teste
    rand = np.random.default_rng(seed) 
    n = len(y)

    if estratificar is None:
        idx = rand.permutation(n)       # Cria vetor com índices aleatórios
        div = int(n * (1 - propTeste))    # Divide quantas amostras são para treino e quantas são para testes
        tr, te = idx[:div], idx[div:]
    else:
        tr, te = [], []
        
        for cls in np.unique(estratificar):
            iclasse = np.where(estratificar == cls)[0]
            rand.shuffle(iclasse)
            div = int(len(iclasse) * (1 - propTeste))
            tr.extend(iclasse[:div]); te.extend(iclasse[div:])
        
        tr, te = np.array(tr), np.array(te)
        rand.shuffle(tr); rand.shuffle(te)
    
    return X[tr], y[tr], X[te], y[te]

class No:   # Nó da árvore

    def __init__(self, feature = None , limiar = None, esquerda = None, direita = None, predicao = None):
        self.feature = feature
        self.limiar = limiar
        self.esquerda = esquerda
        self.direita = direita
        self.predicao = predicao 

    def isFolha(self):
        return self.predicao is not None
    
def gini(y):
    if len(y) == 0:
        return 0
    classes, contagens = np.unique(y, return_counts=True)
    prob = contagens / len(y)
    return 1 - np.sum(prob ** 2)

def mse(y):
    if len(y) == 0:
        return 0
    
    media = np.mean(y)
    return np.mean((y - media) ** 2)

def melhorDivisao(X, y, fnImpureza, featuresCandidatas = None):
    impurezaPai = fnImpureza(y)
    melhorGanho = 0.0
    melhorF, melhorT = None, None
    n = len(y)

    cols = range(X.shape[1]) if featuresCandidatas is None else featuresCandidatas

    for f in cols:
        valores = np.unique(X[:, f])

        if len(valores) < 2:
            continue

        candidatas = (valores[:1] + valores[1:]) / 2.0

    for t in candidatas:
        mascaraEsq = X[:, f] <= t
        ye = y[mascaraEsq]
        yd = y[~mascaraEsq]
        
        if len(ye) == 0 or len(yd) == 0:
            continue

        impurezaFilhos = (len(ye)/n) * fnImpureza(ye) + (len(yd)/n) * fnImpureza(yd)
        ganho = impurezaPai - impurezaFilhos
        if ganho > melhorGanho:
            melhorGanho, melhorF, melhorT = ganho, f, t

    return melhorF, melhorT

def folhaClassificacao(y):
    valores, contagens = np.unique(y, return_counts=True)
    return valores[np.argmax(contagens)]

def folhaRegressao(y):
    return np.mean(y)

def construirArvore(X, y, fnImpureza, fnFolha, profMax = 10, minAmostras = 5, profundidade = 0, maxFeatures = None, rand = None):
    if (profundidade >= profMax
        or len(y) < minAmostras
        or fnImpureza(y) == 0):
        return No(predicao=fnFolha(y))

    if maxFeatures is None:
        candidatas = None
    else:
        k = min(maxFeatures, X.shape[1])
        candidatas = rand.choice(X.shape[1], size=k, replace=False)

    f, t, ganho = melhorDivisao(X, y, fnImpureza,
                                 featuresCandidatas=candidatas)
    if f is None or ganho <= 0:
        return No(predicao=fnFolha(y))
 
    masc = X[:, f] <= t
    esq = construirArvore(X[masc], y[masc], fnImpureza, fnFolha,
                           profMax, minAmostras, profundidade + 1,
                           maxFeatures, rand)
    dir = construirArvore(X[~masc], y[~masc], fnImpureza, fnFolha,
                           profMax, minAmostras, profundidade + 1,
                           maxFeatures, rand)
    return No(feature=f, limiar=t, esquerda=esq, direita=dir)