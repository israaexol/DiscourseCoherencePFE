import torch
import torch.optim as optim
import time
import random
from torch.autograd import Variable
from evaluation import *
import progressbar
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.utils import class_weight
from numpy import *
import math
from numpy import mean
from numpy import std
from numpy import argmax
from numpy import tensordot
from numpy.linalg import norm
import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
    
# normalize a vector to have unit norm
def normalize(weights):
	# calculate l1 vector norm
	result = norm(weights, 1)
	# check for a vector of all zeros
	if result == 0.0:
		return weights
	# return normalized vector (unit norm)
	return weights / result

def train_test(params, training_docs, test_docs, data, model): #To test the embeddings 
    if params['model_type'] == 'par_seq':
        training_data, training_labels, train_ids = data.create_doc_sents(training_docs, 'paragraph', params['task'],
                                                                          params['train_data_limit'])
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'paragraph', params['task'], params['train_data_limit'])
   
    if USE_CUDA:
        model.cuda()
    if params['train_data_limit'] != -1:
        training_docs = training_docs[:10]
        test_docs = test_docs[:10]
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, weight_decay=params['l2_reg'])
    scheduler = None
    if params['lr_decay'] == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif params['lr_decay'] == 'lambda':
        lambda1 = lambda epoch: 0.95 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])
    if params['task'] == 'class' or params['task'] == 'perm' or params['task'] == 'minority':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif params['task'] == 'score_pred':
        loss_fn = torch.nn.MSELoss()
    timestamp = time.time()
    best_test_acc = 0

    for epoch in range(params['num_epochs']):
        if params['lr_decay'] == 'lambda' or params['lr_decay'] == 'step':
            scheduler.step()
            print(optimizer.param_groups[0]['lr'])
        print("EPOCH "+str(epoch))
        total_loss = 0
        steps = int(len(training_data) / params['batch_size'])
        indices = list(range(len(training_data)))
        random.shuffle(indices)
        bar = progressbar.ProgressBar()
        model.train()
        for step in bar(range(steps)):

            batch_ind = indices[(step * params["batch_size"]):((step + 1) * params["batch_size"])]
            sentences, orig_batch_labels = data.get_batch(training_data, training_labels, batch_ind, params['model_type'], params['clique_size'])
            batch_padded, batch_lengths, original_index = data.pad_to_batch(sentences, data.word_to_idx, params['model_type'], params['clique_size'])
            model.zero_grad()
            model(batch_padded, batch_lengths, original_index)
            print('======== batch padded size =========')
            print(len(batch_padded[0]))

    return best_test_acc

def train(params, training_docs, test_docs, data, model):
    if params['model_type'] == 'clique':
        training_data, training_labels = data.create_cliques(training_docs, params['task'], params['train_data_limit'])
        test_data, test_labels = data.create_cliques(test_docs, params['task'], params['train_data_limit'])
    elif params['model_type'] == 'sent_avg':
        training_data, training_labels, train_ids = data.create_doc_sents(training_docs, 'sentence', params['task'], params['train_data_limit'])
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'sentence', params['task'], params['train_data_limit'])
    elif params['model_type'] == 'par_seq':
        training_data, training_labels, train_ids = data.create_doc_sents(training_docs, 'paragraph', params['task'],
                                                                          params['train_data_limit'])
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'paragraph', params['task'], params['train_data_limit'])
    elif params['model_type']=='sem_rel':
        training_data, training_labels, train_ids = data.create_doc_sents(training_docs, 'paragraph', params['task'],
                                                                          params['train_data_limit'])
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'paragraph', params['task'], params['train_data_limit'])
    elif params['model_type']=='cnn_pos_tag':
        training_data, training_labels, train_ids = data.create_doc_sents(training_docs, 'sentence', params['task'],
                                                                          params['train_data_limit'])
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'sentence', params['task'], params['train_data_limit'])

    if USE_CUDA:
        model.cuda()
    if params['train_data_limit'] != -1:
        training_docs = training_docs[:10]
        test_docs = test_docs[:10]
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, weight_decay=params['l2_reg'])
    scheduler = None
    if params['lr_decay'] == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif params['lr_decay'] == 'lambda':
        lambda1 = lambda epoch: 0.95 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])
    if params['task'] == 'class' or params['task'] == 'perm' or params['task'] == 'minority':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif params['task'] == 'score_pred':
        loss_fn = torch.nn.MSELoss()
    timestamp = time.time()
    best_test_acc = 0
    best_weights = None
    best_score = 0
    w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for epoch in range(params['num_epochs']):
        if params['lr_decay'] == 'lambda' or params['lr_decay'] == 'step':
            scheduler.step()
            print(optimizer.param_groups[0]['lr'])
        print("EPOCH "+str(epoch))
        total_loss = 0
        steps = int(len(training_data) / params['batch_size'])
        indices = list(range(len(training_data)))
        random.shuffle(indices)
        bar = progressbar.ProgressBar()
        model.train()
        for step in bar(range(steps)):

            batch_ind = indices[(step * params["batch_size"]):((step + 1) * params["batch_size"])]
            sentences, orig_batch_labels = data.get_batch(training_data, training_labels, batch_ind, params['model_type'], params['clique_size'])
            batch_padded, batch_lengths, original_index = data.pad_to_batch(sentences, data.word_to_idx, params['model_type'], params['clique_size'])
            model.zero_grad()
            if params['model_type']== 'sem_rel':
                y_pred = []
                coherence_pred_sent, coherence_pred_par = model(batch_padded, batch_lengths, original_index)
                #gather coherence predictions into one array
                y_pred.append(coherence_pred_sent)
                y_pred.append(coherence_pred_par)
                y_pred = np.array(y_pred)
                print("=============== Y_PRED =====================")
                print(y_pred)
                for weights in product(w, repeat=2):
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
                        best_weights = list(best_weights)
                        print("best_weights")
                        print(best_weights)
                final_pred = model(batch_padded, batch_lengths, original_index, weights=best_weights)
                class_weights = class_weight.compute_class_weight(class_weight='balanced',classes= np.unique(orig_batch_labels),y= orig_batch_labels)
                print("===================== Class weights ==========================")
                print(class_weights)
                class_weights = torch.tensor(class_weights, dtype=torch.float)
               
                loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
                loss = loss_fn(final_pred, Variable(LongTensor(orig_batch_labels)))
            elif params['model_type']== 'cnn_pos_tag': 
                pred_coherence = model(batch_padded, batch_lengths, original_index)
                loss = loss_fn(pred_coherence, Variable(LongTensor(orig_batch_labels))) 
            else: 
                pred_coherence, avg_deg_train= model(batch_padded, batch_lengths, original_index)
                if params['task'] == 'score_pred':
                    loss = loss_fn(pred_coherence, Variable(FloatTensor(orig_batch_labels)))
                else:
                    loss = loss_fn(pred_coherence, Variable(LongTensor(orig_batch_labels)))
            mean_loss = loss / params["batch_size"]
            mean_loss.backward()
            total_loss += loss.cpu().data.numpy()
            optimizer.step()
        current_time = time.time()
        print("Time %-5.2f min" % ((current_time - timestamp) / 60.0))
        print("Train loss: " + str(total_loss))
        output_name = params['model_name'] + '_epoch' + str(epoch)
        if params['model_type'] == 'sent_avg' or params['model_type'] == 'par_seq' or params['model_type']=='sem_rel' or params['model_type']=='cnn_pos_tag':
            if params['task'] == 'minority':
                test_f05, test_precision, test_recall, test_loss = eval_docs(model, loss_fn, test_data, test_labels,
                                                                        data, params)
            elif params['model_type']== 'sem_rel' or params['model_type']== 'cnn_pos_tag':
                test_accuracy, test_loss = eval_docs(model, loss_fn, test_data, test_labels, data, params)                                
            elif params['task'] == 'class' or params['task'] == 'score_pred':
                test_accuracy, test_loss, global_eval_pred, global_avg_deg_test = eval_docs(model, loss_fn, test_data, test_labels, data, params) 

            elif params['task'] == 'perm':
                test_accuracy, test_loss = eval_docs_rank(model, test_docs, data, params)
            print("Test loss: %0.3f" % test_loss)
            if params['task'] == 'score_pred':
                print("Test correlation: %0.5f" % (test_accuracy))
            elif params['task'] == 'minority':
                print("Test F0.5: %0.2f  Precision: %0.2f  Recall: %0.2f" % (test_f05, test_precision, test_recall))
            else:
                print("Test accuracy: %0.2f%%" % (test_accuracy * 100))
        elif params['model_type'] == 'clique':
            train_accuracy, train_loss = eval_cliques(model, loss_fn, training_data,
                                                                                            training_labels,
                                                                                            params['batch_size'],
                                                                                            params['clique_size'], data,
                                                                                            params['model_type'], params['task'])
            if params['task'] == 'score_pred':
                print("Train clique corr: %0.5f" % (train_accuracy))
            else:
                print("Train clique accuracy: %0.2f%%" % (train_accuracy * 100))
            test_clique_accuracy, test_loss = eval_cliques(model, loss_fn, test_data,
                                                                                            test_labels,
                                                                                            params['batch_size'],
                                                                                            params['clique_size'], data, params['model_type'], params['task'])
            print("Test loss: %0.3f" % test_loss)
            if params['task'] == 'score_pred':
                print("Test clique corr: %0.5f" % ((test_clique_accuracy)))
            else:
                print("Test clique accuracy: %0.2f%%" % ((test_clique_accuracy * 100)))
            doc_accuracy, test_precision, test_recall, test_f05 = eval_doc_cliques(model, test_docs, data, params)
            if params['task'] == 'score_pred':
                print("Test document corr: %0.5f" % (doc_accuracy))
            elif params['task'] == 'minority':
                print("Test F0.5: %0.2f  Precision: %0.2f  Recall: %0.2f" % (test_f05, test_precision, test_recall))
            else:
                print("Test document ranking accuracy: %0.2f%%" % (doc_accuracy * 100))
            test_accuracy = doc_accuracy
        if params['task'] == 'minority':
            if test_f05 > best_test_acc:
                best_test_acc = test_f05
                # save best model
                torch.save(model.state_dict(), params['model_dir'] + '/' + params['model_name'] + '_best')
                print('saved model ' + params['model_dir'] + '/' + params['model_name'] + '_best')
        else:
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                # save best model
                torch.save(model.state_dict(), params['model_dir'] + '/' + params['model_name'] + '_best')
                print('saved model ' + params['model_dir'] + '/' + params['model_name'] + '_best')
        print()
        print("==================== BEST TEST ACCURACY =================================")
        print(best_test_acc)
    return best_test_acc

def train_cv(params, data_docs, data, model):
    
    if USE_CUDA:
        model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, weight_decay=params['l2_reg'])
    scheduler = None
    if params['lr_decay'] == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif params['lr_decay'] == 'lambda':
        lambda1 = lambda epoch: 0.95 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])
    if params['task'] == 'class':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif params['task'] == 'score_pred':
        loss_fn = torch.nn.MSELoss()
    timestamp = time.time()
    best_test_acc = 0
    best_weights = None
    best_score = 0
    w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    kfold = StratifiedKFold(n_splits = 5, shuffle = False) 
    for epoch in range(params['num_epochs']):
        if params['lr_decay'] == 'lambda' or params['lr_decay'] == 'step':
            scheduler.step()
            print(optimizer.param_groups[0]['lr'])
        print("EPOCH " + str(epoch))
        total_loss = 0
        model.train()
        labels = []
        for i in range(len(data_docs)):
            labels.append(data_docs[i].label)
        for train, test in kfold.split(np.zeros(4800), labels):
            training_data = np.array(data_docs)[train]
            test_data = np.array(data_docs)[test]
            training_data, training_labels, train_ids = data.create_doc_sents(training_data, 'paragraph', params['task'],
                                                                          params['train_data_limit'])
            test_data, test_labels, test_ids = data.create_doc_sents(test_data, 'paragraph', params['task'], params['train_data_limit'])

            steps = int(len(training_data) / params['batch_size'])
            indices = list(range(len(training_data)))
            random.shuffle(indices)
            bar = progressbar.ProgressBar()
            for step in bar(range(steps)):

                batch_ind = indices[(step * params["batch_size"]):((step + 1) * params["batch_size"])]
                sentences, orig_batch_labels = data.get_batch(training_data, training_labels, batch_ind, params['model_type'], params['clique_size'])
                batch_padded, batch_lengths, original_index = data.pad_to_batch(sentences, data.word_to_idx, params['model_type'], params['clique_size'])
                model.zero_grad()
                if params['model_type']== 'sem_rel':
                    y_pred = []
                    coherence_pred_sent, coherence_pred_par = model(batch_padded, batch_lengths, original_index)
                    #gather coherence predictions into one array
                    y_pred.append(coherence_pred_sent)
                    y_pred.append(coherence_pred_par)
                    y_pred = np.array(y_pred)
                    for weights in product(w, repeat=2):
                        if len(set(weights)) == 1:
                            continue
                        weights = normalize(weights)
                        summed = tensordot(y_pred, weights, axes=((0),(0)))
                        # argmax across classes
                        result = argmax(summed, axis=1)
                        # calculate accuracy
                        score = accuracy_score(orig_batch_labels, result)
                        if score > best_score:
                            best_score = score
                            best_weights = weights
                            best_weights = list(best_weights)
                    final_pred = model(batch_padded, batch_lengths, original_index, weights=best_weights)
                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(final_pred, Variable(LongTensor(orig_batch_labels)))
                else: 
                    pred_coherence = model(batch_padded, batch_lengths, original_index)
                    if params['task'] == 'score_pred':
                        loss = loss_fn(pred_coherence, Variable(FloatTensor(orig_batch_labels)))
                    else:
                        loss = loss_fn(pred_coherence, Variable(LongTensor(orig_batch_labels)))
                mean_loss = loss / params["batch_size"]
                mean_loss.backward()
                total_loss += loss.cpu().data.numpy()
                optimizer.step()
            fold = 0
            current_time = time.time()
            print("Time %-5.2f min" % ((current_time - timestamp) / 60.0))
            print("Fold" + str(fold) + " - Train loss: " +str(total_loss))
            output_name = params['model_name'] + '_epoch' + str(epoch)
            if params['model_type'] == 'sent_avg' or params['model_type'] == 'par_seq' or params['model_type']=='sem_rel' or params['model_type']=='cnn_pos_tag':
                
                if params['model_type']== 'sem_rel':
                    test_accuracy, test_loss = eval_docs(model, loss_fn, test_data, test_labels, data, params)                                
                if params['model_type'] == 'cnn_pos_tag':
                    test_accuracy, test_loss = eval_docs(model, loss_fn, test_data, test_labels, data, params) 
                print("Fold" + str(fold) +" - Test loss: %0.3f" % test_loss)
                if params['task'] == 'score_pred':
                    print("Test correlation: %0.5f" % (test_accuracy))
                else:
                    print("Fold" + str(fold) +" - Test accuracy: %0.2f%%" % (test_accuracy * 100))
            
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                # save best model
                torch.save(model.state_dict(), params['model_dir'] + '/' + params['model_name'] + '_best')
                print('saved model ' + params['model_dir'] + '/' + params['model_name'] + '_best')
            print()
            fold += 1
    print("==================== BEST TEST ACCURACY =================================")
    print(best_test_acc)
    return best_test_acc

def train_prod(params, data_docs, data, model):
    
    if USE_CUDA:
        model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, weight_decay=params['l2_reg'])
    scheduler = None
    if params['lr_decay'] == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif params['lr_decay'] == 'lambda':
        lambda1 = lambda epoch: 0.95 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])
    if params['task'] == 'class':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif params['task'] == 'score_pred':
        loss_fn = torch.nn.MSELoss()
    timestamp = time.time()
    best_test_acc = 0
    kfold = StratifiedKFold(n_splits = 10, shuffle = False) 
    for epoch in range(params['num_epochs']):
        fold = 0
        if params['lr_decay'] == 'lambda' or params['lr_decay'] == 'step':
            scheduler.step()
            print(optimizer.param_groups[0]['lr'])
        print("EPOCH " + str(epoch))
        total_loss = 0
        model.train()
        labels = []
        for i in range(len(data_docs)):
            labels.append(data_docs[i].label)
        for train, test in kfold.split(np.zeros(4800), labels):
            training_data = np.array(data_docs)[train]
            test_data = np.array(data_docs)[test]
            training_data, training_labels, train_ids = data.create_doc_sents(training_data, 'paragraph', params['task'],
                                                                          params['train_data_limit'])
            test_data, test_labels, test_ids = data.create_doc_sents(test_data, 'paragraph', params['task'], params['train_data_limit'])

            steps = int(len(training_data) / params['batch_size'])
            indices = list(range(len(training_data)))
            random.shuffle(indices)
            bar = progressbar.ProgressBar()
            for step in bar(range(steps)):

                batch_ind = indices[(step * params["batch_size"]):((step + 1) * params["batch_size"])]
                sentences, orig_batch_labels = data.get_batch(training_data, training_labels, batch_ind, params['model_type'], params['clique_size'])
                batch_padded, batch_lengths, original_index = data.pad_to_batch(sentences, data.word_to_idx, params['model_type'], params['clique_size'])
                # print("============batch padded==============")
                # print(batch_padded)
                model.zero_grad()
                if params['model_type']== 'sem_rel_prod':
                    coherence_pred = model(batch_padded, batch_lengths, original_index)
                    loss = loss_fn(coherence_pred, Variable(LongTensor(orig_batch_labels)))
                mean_loss = loss / params["batch_size"]
                mean_loss.backward()
                total_loss += loss.cpu().data.numpy()
                optimizer.step()
            current_time = time.time()
            print("Time %-5.2f min" % ((current_time - timestamp) / 60.0))
            print("Fold" + str(fold) + " - Train loss: " +str(total_loss))
            output_name = params['model_name'] + '_epoch' + str(epoch)
            if params['model_type'] == 'sent_avg' or params['model_type'] == 'par_seq' or params['model_type']=='sem_rel_prod':
                
                if params['model_type']== 'sem_rel_prod':
                    test_accuracy, test_loss = eval_docs(model, loss_fn, test_data, test_labels, data, params)                                

                print("Fold" + str(fold) +" - Test loss: %0.3f" % test_loss)
                if params['task'] == 'score_pred':
                    print("Test correlation: %0.5f" % (test_accuracy))
                else:
                    print("Fold" + str(fold) +" - Test accuracy: %0.2f%%" % (test_accuracy * 100))
            
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                # save best model
                torch.save(model.state_dict(), params['model_dir'] + '/' + params['model_name'] + '_best')
                print('saved model ' + params['model_dir'] + '/' + params['model_name'] + '_best')
            print()
            fold +=1
    print("==================== BEST TEST ACCURACY =================================")
    print(best_test_acc)
    return best_test_acc


def test(params, test_docs, data, model):
    if params['model_type'] == 'clique':
        test_data, test_labels = data.create_cliques(test_docs, params['task'])
    elif params['model_type'] == 'sent_avg':
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'sentence', params['task'], params['train_data_limit'])
    elif params['model_type'] == 'par_seq':
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'paragraph', params['task'], params['train_data_limit'])

    if USE_CUDA:
        model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    # output_name = params['model_name'] + '_test'
    if params['model_type'] == 'par_seq' or params['model_type'] == 'sent_avg':
        test_accuracy, test_loss = eval_docs(model, loss_fn, test_data, test_labels, data, params)
        print("Test accuracy: %0.2f%%" % (test_accuracy * 100))
    elif params['model_type'] == 'clique':
        doc_accuracy = eval_doc_cliques(model, test_docs, data, params)
        print("Test document ranking accuracy: %0.2f%%" % (doc_accuracy * 100))
