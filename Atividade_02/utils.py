import numpy as np
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm.notebook import tqdm


def selecionar_melhor_k_knn(ks, X_treino, X_val, y_treino, y_val):
    
    def treinar_knn(k, X_treino, X_val, y_treino, y_val):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_treino, y_treino)
        pred = knn.predict(X_val)
        return f1_score(y_val, pred)
        
    acuracias_val = Parallel(n_jobs=4)(delayed(treinar_knn)(k, X_treino, X_val, y_treino, y_val) for k in ks)       
        
    melhor_val = max(acuracias_val)
    melhor_k = ks[np.argmax(acuracias_val)]        
    knn = KNeighborsClassifier(n_neighbors=melhor_k)
    knn.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])
    
    return knn, melhor_k

def dupla_cv_knn(X, y, n_neighbors):
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    acuracias = []

    pgb = tqdm(total=10, desc='Folds avaliados')
    
    for idx_treino, idx_teste in skf.split(X, y):
        X_treino = X[idx_treino]
        y_treino = y[idx_treino]
        X_teste = X[idx_teste]
        y_teste = y[idx_teste]

        X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, stratify=y_treino,
                                                            test_size=0.2, random_state=1)
        
        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)
        X_val = ss.transform(X_val)
        
        ks = n_neighbors
        
        knn, melhor_k = selecionar_melhor_k_knn(ks, X_treino, X_val, y_treino, y_val)
        
        params = {'n_neighbors' : n_neighbors}
        clf = GridSearchCV(knn, params, scoring='f1', cv=StratifiedKFold(n_splits=5))
        clf.fit(X_treino, y_treino)

        acuracias.append(f1_score(y_teste, clf.predict(X_teste)))
        
        pgb.update(1)
        
    pgb.close()
        
    return acuracias, melhor_k


def selecionar_melhor_svm(Cs, gammas, X_treino : np.ndarray, X_val : np.ndarray, 
                          y_treino : np.ndarray, y_val : np.ndarray, n_jobs=4):
    
    def treinar_svm(C, gamma, X_treino, X_val, y_treino, y_val):
        svm = SVC(C=C, gamma=gamma)
        svm.fit(X_treino, y_treino)
        pred = svm.predict(X_val)
        return f1_score(y_val, pred)
    
    combinacoes_parametros = list(itertools.product(Cs, gammas))
    
    acuracias_val = Parallel(n_jobs=n_jobs)(delayed(treinar_svm)
                                            (c, g, X_treino, X_val, y_treino, y_val) for c, g in 
                                            combinacoes_parametros)       
    
    melhor_val = max(acuracias_val)
    melhor_comb = combinacoes_parametros[np.argmax(acuracias_val)]   
    melhor_c = melhor_comb[0]
    melhor_gamma = melhor_comb[1]
    
    svm = SVC(C=melhor_c, gamma=melhor_gamma)
    svm.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])

    return svm, melhor_c, melhor_gamma


def dupla_cv_svm(X, y, Cs, gammas):
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    acuracias = []

    pgb = tqdm(total=10, desc='Folds avaliados')
    
    for idx_treino, idx_teste in skf.split(X, y):
        X_treino = X[idx_treino]
        y_treino = y[idx_treino]
        X_teste = X[idx_teste]
        y_teste = y[idx_teste]

        X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, stratify=y_treino,
                                                            test_size=0.2, random_state=1)
        
        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)
        X_val = ss.transform(X_val)
        
        svm, melhor_c, melhor_gamma = selecionar_melhor_svm(Cs, gammas, X_treino, X_val, y_treino, y_val)
        
        params = [{'kernel': ['rbf'], 'gamma': gammas, 'C': Cs}]
        
        clf = GridSearchCV(svm, params, scoring='f1', cv=StratifiedKFold(n_splits=5))
        clf.fit(X_treino, y_treino)

        acuracias.append(f1_score(y_teste, clf.predict(X_teste)))
        
        pgb.update(1)
        
    pgb.close()
        
    return acuracias, melhor_c, melhor_gamma

def calcular_estatisticas(resultados):
    return np.mean(resultados), np.std(resultados), np.min(resultados), np.max(resultados)