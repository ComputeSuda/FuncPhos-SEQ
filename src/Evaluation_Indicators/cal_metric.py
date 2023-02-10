import numpy as np
from sklearn.metrics import *


def cal_two_class_metric(true_label, pred_prob):  
    """
    Calculate the value of each evaluation index
    """

    pred_label = pred_prob.copy()     # prediction probability
    pred_label[pred_label>=0.5] = 1    # prediction label
    pred_label[pred_label<0.5] = 0    # prediction label


    m = {}


    c_matrix = confusion_matrix(true_label, pred_label,labels=[1, 0])
    # print(c_matrix)
    TP, FN, FP, TN = c_matrix[0, 0], c_matrix[0, 1], c_matrix[1, 0], c_matrix[1, 1]

    # print(classification_report(true_label,pred_label))
    
    # accuracy = accuracy_score(true_label, pred_label, normalize=True, sample_weight=None)
    acc = (TP + TN) / (TP + FN + FP + TN)
    # print('accuracy:', accuracy, acc)
    m['accuracy'] = round(acc, 3)

    # precision = precision_score(true_label, pred_label, labels=None, pos_label=1, average='binary', sample_weight=None)
    pre = TP / (TP + FP)
    # print('precision:', precision, pre)
    m['precision'] = round(pre, 3)

    # sensitivity = recall_score(true_label, pred_label, labels=None, pos_label=1, average='binary', sample_weight=None)
    sen = TP / (TP + FN)
    # print('sensitivity:', sensitivity, sen)
    m['sensitivity'] = round(sen, 3)  # recall

    specificity = TN / (TN + FP)
    # print('specificity:', specificity)
    m['specificity'] = round(specificity, 3) 

    true_positive_rate = TP / (TP + FN)
    m['true_positive_rate'] = round(true_positive_rate, 3)

    true_negative_rate = TN / (FP+ TN) 
    m['true_negative_rate'] = round(true_negative_rate, 3)

    false_positive_rate = FP / (FP + TN)
    # print('false positive rate', false_positive_rate)
    m['false_positive_rate'] = round(false_positive_rate, 3)

    false_negative_rate = FN / (TP + FN)
    # print('false negative rate:', false_negative_rate)
    m['false_negative_rate'] = round(false_negative_rate, 3)

    # f1_score = f1_score(true_label, pred_label, labels=None, pos_label=1, average='binary', sample_weight=None)
    f1 = 2 * TP / (2 * TP + FP + FN)
    # print('f1 score:', f1_score, f1)
    m['f1_score'] = round(f1, 3)
    
    auc_ = roc_auc_score(true_label, pred_prob, average='macro', sample_weight=None)
    # print('auc:', auc)
    m['auc'] = round(auc_, 3)
    # fpr, tpr, threshold = roc_curve(true_label, pred_prob)
    # roc_auc = auc(fpr, tpr)
    # print('auc:', roc_auc)


    m_c = matthews_corrcoef(true_label, pred_label)
    # print('matthews corrcoef:', m_c)
    m['matthews_correlation_coefficient'] = round(m_c, 3)

    return c_matrix, m  
