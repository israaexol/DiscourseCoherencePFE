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
    global_eval_pred = []
    global_eval_labels=[]
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
            #gather coherence predictions into one array
            y_pred.append(test_pred_sent)
            y_pred.append(test_pred_par)
            y_pred = np.array(y_pred)
            print("=============== Y_PRED =====================")
            print(y_pred)
            # define weights to consider
            # iterate all possible combinations (cartesian product)
            for weights in product(w, repeat=2):
                # skip if all weights are equal
                if len(set(weights)) == 1:
                    continue
                # hack, normalize weight vector
                weights = normalize(weights)
                # evaluate weights
                # weighted sum across ensemble members
                summed = tensordot(y_pred, weights, axes=((0),(0)))
                print("================summed===================")
                print(summed)
                # argmax across classes
                result = argmax(summed, axis=1)
                print("================result===================")
                print(result)
                # calculate accuracy
                score = accuracy_score(orig_batch_labels, result)
                print("====================score==================")
                print(score)
                #score = evaluate_ensemble(members, weights, testX, testy)
                if score > best_score:
                    best_score = score
                    best_weights = weights
                    print("best_weights")
                    best_weights = list(best_weights)
                    print(best_weights)
            final_pred = model(batch_padded, batch_lengths, original_index, weights=best_weights)
            # class_weights = class_weight.compute_class_weight(class_weight='balanced',classes= np.unique(Y),y= Y)
            # print("===================== Class weights ==========================")
            # print(class_weights)
            # classWeight = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}
            # print("===================== Class weights dictionary ==========================")
            # print(classWeight)
            
            
            #loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
            #loss = loss_fn(final_pred, Variable(LongTensor(orig_batch_labels)))
            eval_labels.extend(orig_batch_labels)
            
            class_weights = class_weight.compute_class_weight(class_weight='balanced',classes= np.unique(orig_batch_labels),y= orig_batch_labels)
            print("===================== Class weights ==========================")
            print(class_weights)
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
            loss += loss_fn(final_pred, Variable(LongTensor(orig_batch_labels))).cpu().data.numpy()
            eval_pred.extend(list(np.argmax(final_pred.cpu().data.numpy(), axis=1)))
            
        else:
            batch_pred, avg_deg_test = model(batch_padded, batch_lengths, original_index)
            print("=========== BATCH PRED ===========")
            print(batch_pred)
            global_avg_deg_test += avg_deg_test
            eval_labels.extend(orig_batch_labels)
            print("=========== EVAL LABELS ===========")
            print(eval_labels)
            if params['task'] == 'score_pred':
                loss += loss_fn(batch_pred, Variable(FloatTensor(orig_batch_labels))).cpu().data.numpy()
                eval_pred.extend(list(batch_pred.cpu().data.numpy())) 
            
            else:
                loss += loss_fn(batch_pred, Variable(LongTensor(orig_batch_labels))).cpu().data.numpy()
                eval_pred.extend(list(np.argmax(batch_pred.cpu().data.numpy(), axis=1)))
                print("===============Eval pred size================")
                print(len(eval_pred))
                print("=========== EVAL PRED ===========")
                print(eval_pred)
             
    if params['task'] == 'score_pred':
        mse = np.square(np.subtract(np.array(eval_pred), np.expand_dims(np.array(eval_labels), 1))).mean()
        corr = spearmanr(np.array(eval_pred), np.expand_dims(np.array(eval_labels), 1))[0]
        accuracy = corr
      
    elif params['task'] == 'minority':
        f05, precision, recall = evaluate(eval_pred, eval_labels, "f05")
    else:
        print("============Eval pred============")
        print(eval_pred)
        print("============Eval labels============")
        print(eval_labels)
        accuracy, num_correct, num_total = evaluate(eval_pred, eval_labels, "accuracy")
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
        print("====================Accuracy low==================================")
        print(acc_low)  
        print("====================Accuracy medium==================================")
        print(acc_medium) 
        print("====================Accuracy high==================================")
        print(acc_high) 
        print("====================Average accuracy==================================")
        print((acc_low + acc_medium + acc_high)/3) 
        print("====================Classification report==================================")
        print(metrics.classification_report(eval_labels, eval_pred, labels=[0, 1, 2], zero_division=1))
    if params['task'] == 'minority':
        return f05, precision, recall, loss
    elif params["model_type"] == 'sem_rel':
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


def eval_cliques(model, loss_fn, clique_data, clique_labels, batch_size, clique_size, data_obj, model_type, task):
    steps = int(len(clique_data) / batch_size)
    if len(clique_data) % batch_size != 0:
        steps += 1
    dev_indices = list(range(len(clique_data)))
    eval_pred = []
    eval_labels = []
    loss = 0
    model.eval()
    for step in range(steps):
        end_idx = (step + 1) * batch_size
        if end_idx > len(clique_data):
            end_idx = len(clique_data)
        batch_ind = dev_indices[(step * batch_size):end_idx]
        sentences, orig_batch_labels = data_obj.get_batch(clique_data, clique_labels, batch_ind, model_type, clique_size)
        batch_padded, batch_lengths, original_index = data_obj.pad_to_batch(sentences, data_obj.word_to_idx, model_type, clique_size)
        batch_pred = model(batch_padded, batch_lengths, original_index)
        if task == 'score_pred':
            loss += loss_fn(batch_pred, Variable(FloatTensor(orig_batch_labels))).cpu().data.numpy()
            eval_pred.extend(list(batch_pred.cpu().data.numpy()))
        else:
            loss += loss_fn(batch_pred, Variable(LongTensor(orig_batch_labels))).cpu().data.numpy()
            eval_pred.extend(list(np.argmax(batch_pred.cpu().data.numpy(), axis=1)))
        eval_labels.extend(orig_batch_labels)
    if task == 'score_pred':
        mse = np.square(np.subtract(np.array(eval_pred), np.expand_dims(np.array(eval_labels), 1))).mean()
        corr = spearmanr(np.array(eval_pred), np.expand_dims(np.array(eval_labels), 1))[0]
        accuracy = corr
    else:
        accuracy, num_correct, num_total = evaluate(eval_pred, eval_labels, "accuracy")
    return accuracy, loss


def eval_doc_cliques(model, docs, data_obj, params):
    num_correct = 0
    num_total = 0
    tp = 0
    fp = 0
    fn = 0
    model.eval()
    eval_ids = []
    eval_pred = []
    eval_labels = []
    for doc in docs:
        if params['task'] == 'perm':
            orig_doc_cliques, perm_doc_cliques = data_obj.retrieve_doc_cliques_by_label(doc, params['task'])
            orig_doc_score = score_doc(model, orig_doc_cliques, params['batch_size'], params['clique_size'], data_obj, params['model_type'])
            for perm_count, cliques in enumerate(perm_doc_cliques):
                perm_doc_score = score_doc(model, cliques, params['batch_size'], params['clique_size'], data_obj, params['model_type'])
                eval_ids.append(doc.id + "#" + str(perm_count))
                if orig_doc_score > perm_doc_score:
                    num_correct += 1
                    eval_pred.append(1)
                else:
                    eval_pred.append(0)
                num_total += 1
        elif params['task'] == 'class':
            orig_doc_cliques, _ = data_obj.retrieve_doc_cliques_by_label(doc, params['task'])
            pred_label = label_doc(model, orig_doc_cliques, params['batch_size'], params['clique_size'], data_obj, params['model_type'])
            eval_pred.append(pred_label)
            if pred_label == doc.label:
                num_correct += 1
            num_total += 1
        elif params['task'] == 'minority':
            orig_doc_cliques, _ = data_obj.retrieve_doc_cliques_by_label(doc, params['task'])
            pred_label = label_doc(model, orig_doc_cliques, params['batch_size'], params['clique_size'], data_obj,
                                   params['model_type'])
            eval_pred.append(pred_label)
            if pred_label == doc.label:
                num_correct += 1
            if pred_label == doc.label:
                if doc.label == 1:
                    tp = 1
            else:
                if pred_label == 1:
                    fp += 1
                else:
                    fn += 1
            num_total += 1
        elif params['task'] == 'score_pred':
            orig_doc_cliques, _ = data_obj.retrieve_doc_cliques_by_label(doc, params['task'])
            pred_score = score_doc_regression(model, orig_doc_cliques, params['batch_size'], params['clique_size'], data_obj, params['model_type'])
            eval_pred.append(pred_score)
            eval_labels.append(doc.label)
    precision = 0
    recall = 0
    f05 = 0
    if params['task'] == 'score_pred':
        mse = np.square(np.subtract(eval_pred, eval_labels)).mean()
        corr = spearmanr(eval_pred, eval_labels)[0]
        accuracy = corr
    else:
        accuracy = num_correct / num_total
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        if (precision + recall) > 0:
            f05 = (1.25 * precision * recall) / (1.25 * precision + recall)
    return accuracy, precision, recall, f05


# average scores of all cliques for a single document (3-class task)
def label_doc(model, doc_cliques, batch_size, clique_size, data_obj, model_type):
    steps = int(len(doc_cliques) / batch_size)
    labels = [-1 for clique in doc_cliques]
    if len(doc_cliques) % batch_size != 0:
        steps += 1
    clique_indices = list(range(len(doc_cliques)))
    pred_distributions = None
    model.eval()
    for step in range(steps):
        end_idx = (step + 1) * batch_size
        if end_idx > len(doc_cliques):
            end_idx = len(doc_cliques)
        batch_ind = clique_indices[(step * batch_size):end_idx]
        sentences, orig_batch_labels = data_obj.get_batch(doc_cliques, labels, batch_ind, model_type, clique_size)
        batch_padded, batch_lengths, original_index = data_obj.pad_to_batch(sentences, data_obj.word_to_idx, model_type, clique_size)
        batch_pred = model(batch_padded, batch_lengths, original_index)
        batch_data = batch_pred.cpu().data.numpy()
        if pred_distributions is None:
            pred_distributions = batch_data
        else:
            pred_distributions = np.concatenate([pred_distributions, batch_data])
    pred_label = np.argmax(np.mean(pred_distributions, axis=0))
    return pred_label


# average scores of all cliques for a single document (binary task)
def score_doc(model, doc_cliques, batch_size, clique_size, data_obj, model_type):
    steps = int(len(doc_cliques) / batch_size)
    labels = [-1 for clique in doc_cliques]
    if len(doc_cliques) % batch_size != 0:
        steps += 1
    clique_indices = list(range(len(doc_cliques)))
    prob_list = []
    model.eval()
    for step in range(steps):
        end_idx = (step + 1) * batch_size
        if end_idx > len(doc_cliques):
            end_idx = len(doc_cliques)
        batch_ind = clique_indices[(step * batch_size):end_idx]
        sentences, orig_batch_labels = data_obj.get_batch(doc_cliques, labels, batch_ind, model_type, clique_size)
        batch_padded, batch_lengths, original_index = data_obj.pad_to_batch(sentences, data_obj.word_to_idx, model_type, clique_size)
        batch_pred = model(batch_padded, batch_lengths, original_index)
        batch_data = batch_pred.cpu().data.numpy()
        for row in batch_data:
            prob_list.append(row[1]) # probability that the clique is coherent
    score = np.mean(prob_list)
    return score


# average scores of all cliques for a single document (score prediction task)
def score_doc_regression(model, doc_cliques, batch_size, clique_size, data_obj, model_type):
    steps = int(len(doc_cliques) / batch_size)
    labels = [-1 for clique in doc_cliques]
    if len(doc_cliques) % batch_size != 0:
        steps += 1
    clique_indices = list(range(len(doc_cliques)))
    prob_list = []
    model.eval()
    for step in range(steps):
        end_idx = (step + 1) * batch_size
        if end_idx > len(doc_cliques):
            end_idx = len(doc_cliques)
        batch_ind = clique_indices[(step * batch_size):end_idx]
        sentences, orig_batch_labels = data_obj.get_batch(doc_cliques, labels, batch_ind, model_type, clique_size)
        batch_padded, batch_lengths, original_index = data_obj.pad_to_batch(sentences, data_obj.word_to_idx, model_type, clique_size)
        batch_pred = model(batch_padded, batch_lengths, original_index)
        batch_data = batch_pred.cpu().data.numpy()
        for row in batch_data:
            prob_list.append(row[0]) # regression score
    score = np.mean(prob_list)
    return score