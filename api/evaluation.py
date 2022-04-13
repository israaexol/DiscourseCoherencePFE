import numpy as np
from torch.autograd import Variable
import torch
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
from scipy.stats import spearmanr
import csv
from sklearn import metrics
from sklearn import linear_model
import torch.nn.functional as F
from sklearn.utils import class_weight
from numpy import mean
from numpy import std
from numpy import argmax
from numpy import tensordot
from numpy.linalg import norm
from itertools import product
from sklearn.metrics import accuracy_score
import sys

# normalize a vector to have unit norm
def normalize(weights):
	# calculate l1 vector norm
	result = norm(weights, 1)
	# check for a vector of all zeros
	if result == 0.0:
		return weights
	# return normalized vector (unit norm)
	return weights / result

def eval_docs(model, loss_fn, eval_data, labels, data_obj, params):
    steps = int(len(eval_data) / params['batch_size'])
    if len(eval_data) % params['batch_size'] != 0:
        steps += 1
    eval_indices = list(range(len(eval_data)))
    eval_pred = []
    eval_labels = []
    global_avg_deg_test = []
    best_weights = None
    best_score = 0
    w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    loss = 0
    model.eval()
    for step in range(steps):
        end_idx = (step + 1) * params['batch_size']
        if end_idx > len(eval_data):
            end_idx = len(eval_data)
        batch_ind = eval_indices[(step * params['batch_size']):end_idx]
        sentences, orig_batch_labels = data_obj.get_batch(eval_data, labels, batch_ind, params['model_type'])
        batch_padded, batch_lengths, original_index = data_obj.pad_to_batch(
            sentences, data_obj.word_to_idx, params['model_type'])
        if params['model_type']== 'sem_rel':
            y_pred = []
            test_pred_sent, test_pred_par = model(batch_padded, batch_lengths, original_index)
            # Regrouper les prédictions du niveau phrases et les prédictions du niveau paragraphes dans un seul tableau
            y_pred.append(test_pred_sent)
            y_pred.append(test_pred_par)
            y_pred = np.array(y_pred)
            for weights in product(w, repeat=2):
                # Si les poids sont égaux
                if len(set(weights)) == 1:
                    continue
                # hack, normalize weight vector
                weights = normalize(weights)
                # calculer les poids à associer pour chaque ensemble de prédictions
                # somme pondérée à travers les deux modèles
                summed = tensordot(y_pred, weights, axes=((0),(0)))
                # argmax à travers les classes
                result = argmax(summed, axis=1)
                # Calculer l'exactitude
                score = accuracy_score(orig_batch_labels, result)
                # Récuperer le meilleur score d'exactitude à travers la combinaison des poids associée
                if score > best_score:
                    best_score = score
                    best_weights = weights
                    best_weights = list(best_weights)
            final_pred = model(batch_padded, batch_lengths, original_index, weights=best_weights)
            eval_labels.extend(orig_batch_labels)
            loss += loss_fn(final_pred, Variable(LongTensor(orig_batch_labels))).cpu().data.numpy()
            eval_pred.extend(list(np.argmax(final_pred.cpu().data.numpy(), axis=1)))
        elif params['model_type']=='sem_rel_prod':
            batch_pred = model(batch_padded, batch_lengths, original_index)
            eval_labels.extend(orig_batch_labels)
            loss += loss_fn(batch_pred, Variable(LongTensor(orig_batch_labels))).cpu().data.numpy()
            eval_pred.extend(list(np.argmax(batch_pred.cpu().data.numpy(), axis=1)))      
        else:
            batch_pred, avg_deg_test = model(batch_padded, batch_lengths, original_index)
            # print("=========== BATCH PRED ===========")
            # print(batch_pred)
            global_avg_deg_test += avg_deg_test
            eval_labels.extend(orig_batch_labels)
            # print("=========== EVAL LABELS ===========")
            # print(eval_labels)
            if params['task'] == 'score_pred':
                loss += loss_fn(batch_pred, Variable(FloatTensor(orig_batch_labels))).cpu().data.numpy()
                eval_pred.extend(list(batch_pred.cpu().data.numpy())) 
            else:
                loss += loss_fn(batch_pred, Variable(LongTensor(orig_batch_labels))).cpu().data.numpy()
                eval_pred.extend(list(np.argmax(batch_pred.cpu().data.numpy(), axis=1)))
                # print("===============Eval pred size================")
                # print(len(eval_pred))
                # print("=========== EVAL PRED ===========")
                # print(eval_pred)
             
    if params['task'] == 'score_pred':
        mse = np.square(np.subtract(np.array(eval_pred), np.expand_dims(np.array(eval_labels), 1))).mean()
        corr = spearmanr(np.array(eval_pred), np.expand_dims(np.array(eval_labels), 1))[0]
        accuracy = corr
    else:
        accuracy, num_correct, num_total = evaluate(eval_pred, eval_labels, "accuracy")
        print("Accuracy :")
        print(accuracy)
        matrix = metrics.confusion_matrix(eval_labels, eval_pred, labels=[0, 1, 2])
        print("Confusion matrix :")
        print(matrix)
        sum = np.sum(matrix)
        acc_low = (matrix[0][0] + matrix[1][1] + matrix[1][2] + matrix[2][1] + matrix [2][2])/(sum) #exactitude relative à la classe low
        acc_medium = (matrix[0][0] + matrix[1][1] + matrix[0][2] + matrix[2][0] + matrix [2][2])/(sum) #exactitude relative à la classe medium
        acc_high = (matrix[0][0] + matrix[1][1] + matrix[0][1] + matrix[1][0] + matrix [2][2])/(sum) #exactitude relative à la classe high
        print("Accuracy low :")
        print(acc_low)  
        print("Accuracy medium :")
        print(acc_medium) 
        print("Accuracy high :")
        print(acc_high) 
        print("Average accuracy :")
        print((acc_low + acc_medium + acc_high)/3) 
        print("Classification report :")
        print(metrics.classification_report(eval_labels, eval_pred, labels=[0, 1, 2], zero_division=1))
    if params["model_type"] == 'sem_rel' or params['model_type']=='sem_rel_prod':
        return accuracy, loss
    else:
        return accuracy, loss, eval_pred, global_avg_deg_test

def eval_docs_rank(model, eval_docs, data_obj, params):
    num_correct = 0
    num_total = 0
    loss = 0
    model.eval()
    eval_pred = []
    eval_ids_perm = []
    for doc in eval_docs:
        orig_doc, perm_docs = data_obj.retrieve_doc_sents_by_label(doc)
        batch_padded_orig, batch_lengths_orig, original_index_orig = data_obj.pad_to_batch(orig_doc, data_obj.word_to_idx, params['model_type'])
        orig_pred = model(batch_padded_orig, batch_lengths_orig, original_index_orig)
        orig_coh_score = orig_pred.cpu().data.numpy()[0][1] # probability that doc is coherent
        for idx, perm_doc in enumerate(perm_docs):
            perm_doc = [perm_doc]
            batch_padded_perm, batch_lengths_perm, original_index_perm = data_obj.pad_to_batch(perm_doc, data_obj.word_to_idx, params['model_type'])
            perm_pred = model(batch_padded_perm, batch_lengths_perm, original_index_perm)
            pred_coh_score = perm_pred.cpu().data.numpy()[0][1]  # probability that doc is coherent
            if orig_coh_score > pred_coh_score:
                num_correct += 1
                eval_pred.append(1)
            else:
                eval_pred.append(0)
            eval_ids_perm.append(doc.id + "#" + str(idx+1))
            num_total += 1
    accuracy = num_correct / num_total
    return accuracy, loss

def evaluate(pred_labels, labels, type):
    num_correct = 0
    num_total = 0
    tp = 0
    fp = 0
    fn = 0
    for index, pred_val in enumerate(pred_labels):
        gold_val = labels[index]
        if type == "accuracy":
            if pred_val == gold_val:
                num_correct += 1
        elif type == "f05":
            if pred_val == gold_val:
                if gold_val == 1:
                    tp += 1
            else:
                if pred_val == 1:
                    fp += 1
                else:
                    fn += 1
        num_total += 1
    if type == "f05":
        precision = 0
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        recall = 0
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        f05 = 0
        if (precision + recall) > 0:
            f05 = (1.25 * precision * recall) / (1.25 * precision + recall)
        return f05, precision, recall
    return np.sum(np.array(pred_labels) == np.array(labels)) / float(
        len(pred_labels)), num_correct, num_total


def eval_docs_fusion(model_cnn, model_sem, loss_fn, eval_data_cnn, eval_data_sem, labels, data_obj, params):
    steps = int(len(eval_data_cnn) / params['batch_size'])
    if len(eval_data_cnn) % params['batch_size'] != 0:
        steps += 1
    eval_indices = list(range(len(eval_data_cnn)))
    eval_pred = []
    eval_labels = []
    global_avg_deg_test = []
    global_eval_pred = []
    global_eval_labels=[]
    best_weights = None
    best_score = 0
    w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    loss = 0
    model_cnn.eval()
    model_sem.eval()
    for step in range(steps):
        end_idx = (step + 1) * params['batch_size']
        if end_idx > len(eval_data_cnn):
            end_idx = len(eval_data_cnn)
        batch_ind = eval_indices[(step * params['batch_size']):end_idx]
        sentences_cnn, orig_batch_labels = data_obj.get_batch(eval_data_cnn, labels, batch_ind, params['model_type'])
        sentences_sem, orig_batch_labels = data_obj.get_batch(eval_data_sem, labels, batch_ind, params['model_type'])
        
        batch_padded_cnn, batch_lengths_cnn, original_index = data_obj.pad_to_batch(
            sentences_cnn, data_obj.word_to_idx, params['model_type'])
        batch_padded_sem, batch_lengths_sem, original_index = data_obj.pad_to_batch(
            sentences_sem, data_obj.word_to_idx, params['model_type'])
        if params['model_type']== 'fusion_sem_syn':
            y_pred = []
            test_pred_cnn = model_cnn(batch_padded_cnn, batch_lengths_cnn, original_index)
            coherence_pred_cnn_Tensor = test_pred_cnn
            test_pred_cnn = test_pred_cnn.tolist()
            
            test_pred_sent, test_pred_par = model_sem(batch_padded_sem, batch_lengths_sem, original_index)
            coherence_pred_sem_sentTensor = test_pred_sent
            test_pred_sent = test_pred_sent.tolist()
            
            coherence_pred_sem_parTensor = test_pred_par
            test_pred_par = test_pred_par.tolist()
            #gather coherence predictions into one array
            y_pred.append(test_pred_cnn)
            y_pred.append(test_pred_sent)
            y_pred.append(test_pred_par)
            y_pred = np.array(y_pred)
            
            # define weights to consider
            # iterate all possible combinations (cartesian product)
            for weights in product(w, repeat=3):
                # skip if all weights are equal
                if len(set(weights)) == 1:
                    continue
                # hack, normalize weight vector
                weights = normalize(weights)
                # evaluate weights
                # weighted sum across ensemble members
                summed = tensordot(y_pred, weights, axes=((0),(0)))
                # argmax across classes
                result = argmax(summed, axis=1)
                # calculate accuracy
                score = accuracy_score(orig_batch_labels, result)
                #score = evaluate_ensemble(members, weights, testX, testy)
                if score > best_score:
                    best_score = score
                    best_weights = weights
                    best_weights = list(best_weights)
            
            coherence_pred_cnn = torch.mul(coherence_pred_cnn_Tensor, best_weights[0])
            coherence_pred_sent = torch.mul(coherence_pred_sem_sentTensor, best_weights[1])
            coherence_pred_par = torch.mul(coherence_pred_sem_parTensor, best_weights[2])
            final_prediction1 = coherence_pred_sent.add(coherence_pred_par)
            final_pred = final_prediction1.add(coherence_pred_cnn)
            eval_labels.extend(orig_batch_labels)
            
            loss += loss_fn(final_pred, Variable(LongTensor(orig_batch_labels))).cpu().data.numpy()
            eval_pred.extend(list(np.argmax(final_pred.cpu().data.numpy(), axis=1)))
             
    if params['task'] == 'score_pred':
        mse = np.square(np.subtract(np.array(eval_pred), np.expand_dims(np.array(eval_labels), 1))).mean()
        corr = spearmanr(np.array(eval_pred), np.expand_dims(np.array(eval_labels), 1))[0]
        accuracy = corr
      
    elif params['task'] == 'minority':
        f05, precision, recall = evaluate(eval_pred, eval_labels, "f05")
    else:
        accuracy, num_correct, num_total = evaluate(eval_pred, eval_labels, "accuracy")
        print("=========== eval labels ===========")
        print(eval_labels)
        print("=========== EVAL PRED ===========")
        print(eval_pred)
        print("============Accuracy============")
        print(accuracy)
        matrix = metrics.confusion_matrix(eval_labels, eval_pred, labels=[0, 1, 2])
        print("====================Confusion matrix==================================")
        print(matrix)
        ###Accuracy for low class
        sum = np.sum(matrix)
        acc_low = (matrix[0][0] + matrix[1][1] + matrix[1][2] + matrix[2][1] + matrix [2][2])/(sum)
        acc_medium = (matrix[0][0] + matrix[1][1] + matrix[0][2] + matrix[2][0] + matrix [2][2])/(sum)
        acc_high = (matrix[0][0] + matrix[1][1] + matrix[0][1] + matrix[1][0] + matrix [2][2])/(sum)
        print("==================== Accuracy low ===================")
        print(acc_low)  
        print("==================== Accuracy medium ==================")
        print(acc_medium) 
        print("==================== Accuracy high ====================")
        print(acc_high) 
        print("==================== Average accuracy =================")
        print((acc_low + acc_medium + acc_high)/3) 
        print("====================Classification report============")
        print(metrics.classification_report(eval_labels, eval_pred, labels=[0, 1, 2], zero_division=1))
    if params['task'] == 'minority':
        return f05, precision, recall, loss
    elif params['task']=='class':
        return accuracy, loss
    else:
        return accuracy, loss, eval_pred, global_avg_deg_test
