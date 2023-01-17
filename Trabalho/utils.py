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
    
    return knn, melhor_k, melhor_val

def do_cv_knn(X, y, cv_splits, ks):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    acuracias = []
    fold = 1
    
    pgb = tqdm(total=cv_splits, desc='Folds avaliados')
    
    for treino_idx, teste_idx in skf.split(X, y):

        X_treino = X[treino_idx]
        y_treino = y[treino_idx]

        X_teste = X[teste_idx]
        y_teste = y[teste_idx]

        X_treino, X_val, y_treino, y_val = train_test_split(X_treino,y_treino,stratify=y_treino,
                                                            test_size=0.2, random_state=1)

        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)
        X_val = ss.transform(X_val)

        knn, _, _ = selecionar_melhor_k_knn(ks, X_treino, X_val, y_treino, y_val)
        pred = knn.predict(X_teste)

        acuracias.append(f1_score(y_teste, pred))
        
        fold+=1
        pgb.update(1)
        
    pgb.close()
    
    return acuracias


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

    return svm, melhor_comb, melhor_val

def do_cv_svm(X, y, cv_splits, Cs=[1], gammas=['scale']):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    acuracias = []
    
    pgb = tqdm(total=cv_splits, desc='Folds avaliados')
    
    for treino_idx, teste_idx in skf.split(X, y):

        X_treino = X[treino_idx]
        y_treino = y[treino_idx]

        X_teste = X[teste_idx]
        y_teste = y[teste_idx]

        X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, stratify=y_treino,
                                                            test_size=0.2, random_state=1)

        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)
        X_val = ss.transform(X_val)

        svm, _, _ = selecionar_melhor_svm(Cs, gammas, X_treino, X_val, y_treino, y_val)
        pred = svm.predict(X_teste)

        acuracias.append(f1_score(y_teste, pred))
        
        pgb.update(1)
        
    pgb.close()
    
    return acuracias

def selecionar_melhor_ad(X_treino, X_val, y_treino, y_val, n_jobs=4, cv_folds=None,
                         max_depths=[None], min_samples_leafs=[1], min_samples_splits=[2]):
    
    def treinar_ad(max_depth, min_samples_leaf, min_samples_split, X_treino, X_val, y_treino, y_val):
        ad = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=1)
        ad.fit(X_treino, y_treino)
        pred = ad.predict(X_val)
        if len(set(y_treino)) > 2:
            return f1_score(y_val, pred, average='weighted')
        else:
            return f1_score(y_val, pred)


    if cv_folds is not None:
        ad = DecisionTreeClassifier()

        pg = {
            'max_depth' : max_depths,
            'min_samples_leaf' : min_samples_leafs, 
            'min_samples_split' : min_samples_splits,
        }

        score_fn = 'f1' if len(set(y_treino)) < 3 else 'f1_weighted'

        ad = GridSearchCV(ad, pg, cv=cv_folds, n_jobs=n_jobs, scoring=score_fn)

        ad.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])

        melhor_comb = (ad.best_params_['max_depth'], ad.best_params_['min_samples_leaf'], ad.best_params_['min_samples_split'])
        melhor_val = ad.best_score_

    else:
        combinacoes_parametros = list(itertools.product(max_depths, min_samples_leafs, min_samples_splits))
        f1s_val = Parallel(n_jobs=n_jobs)(delayed(treinar_ad)
                                         (md, msl, mss, X_treino, X_val, y_treino, y_val) for md, msl, mss in combinacoes_parametros)

        melhor_val = max(f1s_val)
        melhor_comb = combinacoes_parametros[np.argmax(f1s_val)]
        melhor_md, melhor_msl, melhor_mss = melhor_comb

        ad = DecisionTreeClassifier(max_depth=melhor_md, min_samples_leaf=melhor_msl, min_samples_split=melhor_mss, random_state=1)
        ad.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])

    return ad, melhor_comb, melhor_val


def do_cv_ad(X, y, cv_splits, param_cv_folds=None, **params_kwargs):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    f1s = []

    pgb = tqdm(total=cv_splits, desc='Folds avaliados')

    for treino_idx, teste_idx in skf.split(X, y):

        X_treino = X[treino_idx]
        y_treino = y[treino_idx]

        X_teste = X[teste_idx]
        y_teste = y[teste_idx]

        X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, stratify=y_treino, test_size=0.2, random_state=1)

        ad, melhor_comb, _ = selecionar_melhor_ad(X_treino, X_val, y_treino, y_val, cv_folds=param_cv_folds, **params_kwargs)
        pred = ad.predict(X_teste)

        if len(set(y_treino)) > 2:
            f1 = f1_score(y_teste, pred, average='weighted')
        else:
            f1 = f1_score(y_teste, pred)
        f1s.append(f1)

        pgb.update(1)

    pgb.close()

    return f1s

def dupla_cv(X, y, k1, k2, n_neighbors):
    #k1 = controla o número de vias da validação cruzada para estimar o desempenho do modelo
    #k2 = controla o número de vida da validação cruzada para otimização de hiperparametros
    
    skf = StratifiedKFold(n_splits=k1, shuffle=True, random_state=1)

    acuracias = []

    pgb = tqdm(total=k1, desc='Folds avaliados')
    
    for idx_treino, idx_teste in skf.split(X, y):
        X_treino = X[idx_treino]
        y_treino = y[idx_treino]
        X_teste = X[idx_teste]
        y_teste = y[idx_teste]

        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)

        params = {'n_neighbors' : n_neighbors}
        knn_cd = KNeighborsClassifier()
        clf = GridSearchCV(knn_cd, params, cv=StratifiedKFold(n_splits=k2))
        clf.fit(X_treino, y_treino)

        acuracias.append(f1_score(y_teste, clf.predict(X_teste)))
        
        pgb.update(1)
        
    pgb.close()
        
    return acuracias


