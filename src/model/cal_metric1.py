import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn import metrics


def cal_two_class_metric(true_label, pred_prob):  # pred_prob:(samples, 1)
	pred_label = pred_prob.copy()     # 概率
	pred_label[pred_label>=0.5] = 1    # 类别
	pred_label[pred_label<0.5] = 0

	print(pred_prob.shape, pred_label.shape)  # (samples, 1)  (samples, 1)
	# print(true_label.shape)
	# print(pred_prob[:20])
	# print(pred_label[:20])
	
	m = {}

	# pred_label = np.array([1, 1, 0, 0, 0.99])
	# pred_label[pred_label>=0.5] = 1
	# true_label = np.array([0, 1, 0, 1, 1])
	# TP:2  FN:1
	# FP:1  TN:1

	c_matrix = confusion_matrix(true_label, pred_label,labels=[1, 0])
	# print(c_matrix)
	TP, FN, FP, TN = c_matrix[0, 0], c_matrix[0, 1], c_matrix[1, 0], c_matrix[1, 1]

	# print(classification_report(true_label,pred_label))
	
	# accuracy = accuracy_score(true_label, pred_label, normalize=True, sample_weight=None)
	acc = (TP + TN) / (TP + FN + FP + TN)
	# print('accuracy:', accuracy, acc)
	m['accuracy'] = round(acc, 5)

	# precision = precision_score(true_label, pred_label, labels=None, pos_label=1, average='binary', sample_weight=None)
	pre = TP / (TP + FP)
	# print('precision:', precision, pre)
	m['precision'] = round(pre, 5)

	# sensitivity = recall_score(true_label, pred_label, labels=None, pos_label=1, average='binary', sample_weight=None)
	sen = TP / (TP + FN)
	# print('sensitivity:', sensitivity, sen)
	m['sensitivity'] = round(sen, 5)  # 就是recall

	specificity = TN / (TN + FP)
	# print('specificity:', specificity)
	m['specificity'] = round(specificity, 5)

	true_positive_rate = TP / (TP + FN)
	m['true_positive_rate'] = round(true_positive_rate, 5)

	true_negative_rate = TN / (FP+ TN) 
	m['true_negative_rate'] = round(true_negative_rate, 5)

	false_positive_rate = FP / (FP + TN)
	# print('false positive rate', false_positive_rate)
	m['false_positive_rate'] = round(false_positive_rate, 5)

	false_negative_rate = FN / (TP + FN)
	# print('false negative rate:', false_negative_rate)
	m['false_negative_rate'] = round(false_negative_rate, 5)

	# f1_score = f1_score(true_label, pred_label, labels=None, pos_label=1, average='binary', sample_weight=None)
	f1 = 2 * TP / (2 * TP + FP + FN)
	# print('f1 score:', f1_score, f1)
	m['f1_score'] = round(f1, 5)
	
	auc_ = roc_auc_score(true_label, pred_prob, average='macro', sample_weight=None)
	# print('auc:', auc)
	m['auc'] = round(auc_, 5)
	# fpr, tpr, threshold = roc_curve(true_label, pred_prob)
	# roc_auc = auc(fpr, tpr)
	# print('auc:', roc_auc)


	m_c = matthews_corrcoef(true_label, pred_label)
	# print('matthews corrcoef:', m_c)
	m['matthews_correlation_coefficient'] = round(m_c, 5)
	m['prec'] = metrics.precision_score(true_label, pred_label)
	m['recal'] = metrics.recall_score(true_label, pred_label)
	m['f1_s'] = metrics.f1_score(true_label, pred_label)
	m['auc1'] = metrics.roc_auc_score(true_label, pred_label)
	return c_matrix, m	

def softmax(X):
	# 二维数组才计算
	assert(len(X.shape) == 2)
	# softmax(x) == softmax(x - max)
	row_max = np.max(X, axis=1).reshape(-1, 1)
	X -= row_max
	X_exp = np.exp(X)
	res = X_exp / np.sum(X_exp, axis=1, keepdims=True)

	return res


def cal_two_class_metric_capsule(true_label, pred_prob):  
	"""
		pred_prob: (sample, 2)
		首先要把每一行进行softmax化
	"""
	softmax_pred_prob = softmax(pred_prob)
	# 再取出第一列 即预测为类别1的概率
	softmax_pred_prob = softmax_pred_prob[: , 1].reshape(-1, 1)

	return cal_two_class_metric(true_label, softmax_pred_prob)


def cal_many_class_metric(true_label, pred_label):
	max_index = np.argmax(pred_label, axis=1)
	c_matrix = confusion_matrix(true_label, max_index,labels=[0, 1, 2])
	print(c_matrix)
	# print('acc:', np.sum(true_label == max_index) / true_label.shape[0])
	report = classification_report(true_label, max_index)
	save_report = classification_report(true_label, max_index, output_dict=True)
	# print(report)

	return str(c_matrix), report


if __name__ == '__main__':
	file=pd.read_excel('test.xlsx')
	true_label=file['label']
	pred_prob=file['pred']
	matrix, metric=cal_two_class_metric(true_label, pred_prob)
	print(matrix)
	print(metric)
	# cal_two_class_metric()
	# cal_many_class_metric()