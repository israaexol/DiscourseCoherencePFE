## Évaluation de la cohérence du discours en utilisant les techniques d'apprentissage profond

## Dépendances

Les dépendances de ce code, écrit en Python, sont:
* Environement d'exécution [Anaconda] avec [Python 3.9.7]
* [scikit-learn] (http://scikit-learn.org/stable/)
* [NLTK] >= 3 (https://www.nltk.org/install.html)
* [progressbar2](https://pypi.org/project/progressbar2/)

## Dataset utilisé

Le corpus de GCDC proposé par (Lai et Tetreault, 2018) et disponible sur demande via (https://github.com/aylai/GCDC-corpus)
Pour la configuration de validation croisée, les 4 domaines (Yahoo, Yelp, Clinton et Enron, le training et le test à la fois) ont été jumlés dans un seul corpus global

## Évaluation des modèles

Les modèles sont évalués sur la tâche de classification à trois niveaux (low, medium, high coherence), et dont le traitement de la cohérence peut être réparti en deux catégories :

## Niveau sémantique

1) Au niveau des phrases (SENT_AVG)
* Sans la validation croisée :
```
python main.py --model_name sentavg_model --train_corpus GCDC --model_type sent_avg --task class
```
* Avec la validation croisée :
```
python main.py --model_name sentavg_model_cv --train_corpus GCDC --model_type sent_avg --task class --cross_val 1
```

2) Au niveau des paragraphes (PAR_SEQ)
* Sans la validation croisée :
```
python main.py --model_name parseq_model --train_corpus GCDC --model_type par_seq --task class
```
* Avec la validation croisée :
```
python main.py --model_name parseq_model_cv --train_corpus GCDC --model_type par_seq --task class --cross_val 1
```
3) Combinaison des deux niveaux (SEM_REL)
* Sans la validation croisée :
```
python main.py --model_name semrel_model --train_corpus GCDC --model_type sem_rel --task class
```

* Avec la validation croisée :
```
python main.py --model_name semrel_model_cv --train_corpus GCDC --model_type sem_rel --task class --cross_val 1
```

## Niveau syntaxique (CNN_POS_TAG)
* Sans la validation croisée :
```
python main.py --model_name cnn_postag_model --train_corpus GCDC --model_type cnn_pos_tag --task class --pos_tag 1
```

* Avec la validation croisée :
```
python main.py --model_name cnn_postag_model_cv --train_corpus GCDC --model_type cnn_pos_tag --task class --pos_tag 1 --cross_val 1
```
