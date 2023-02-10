from sgt import SGT
import csv
import numpy as np
# import keras.utils.np_utils as kutils
import pandas
#input format   label,proteinName, postion,sites, shortsequence,
#input must be a .csv file
#positive_position_file_name is an csv file
import pandas as pd
def onehot_coding(short_seqs,window_size):
    ONE_HOT_SIZE = 21
    # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
    letterDict = {}
    letterDict["A"] = 0
    letterDict["C"] = 1
    letterDict["D"] = 2
    letterDict["E"] = 3
    letterDict["F"] = 4
    letterDict["G"] = 5
    letterDict["H"] = 6
    letterDict["I"] = 7
    letterDict["K"] = 8
    letterDict["L"] = 9
    letterDict["M"] = 10
    letterDict["N"] = 11
    letterDict["P"] = 12
    letterDict["Q"] = 13
    letterDict["R"] = 14
    letterDict["S"] = 15
    letterDict["T"] = 16
    letterDict["V"] = 17
    letterDict["W"] = 18
    letterDict["Y"] = 19
    letterDict["#"] = 20

    Matr = np.zeros((len(short_seqs), window_size, ONE_HOT_SIZE))
    samplenumber = 0
    for seq in short_seqs:
        AANo = 0
        for AA in seq:
            index = letterDict[AA]
            Matr[samplenumber][AANo][index] = 1
            AANo = AANo+1
        samplenumber = samplenumber + 1
    return Matr
def blosum62_coding(short_seqs):
    d = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0],
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, 0],
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, 3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 0],
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 0],
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],
        'Q': [-1, 1, 0, 0, 3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0],
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, 2, 0],
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, 0],
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0],
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, 0],
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, 0],
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0],
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, 0],
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, 0],
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, 0],
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0],
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, 0],
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, 0],
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, 0],
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, 0],
        '#': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }


    Matr=[]
    for seq in short_seqs:
        coding = []
        for c in seq:
            coding.append(np.array(d[c]))
        coding = np.array(coding)
        Matr.append(coding)
    Matr=np.array(Matr)

    return Matr

def phy_che_coding(short_seqs):
    d = {
        'A':[0.008, 0.134, -0.475, -0.039, 0.181],
        'R':[0.171, -0.361, 0.107, -0.258, -0.364],
        'N':[0.255, 0.038, 0.117, 0.118, -0.055],
        'D':[0.303, -0.057, -0.014, 0.225, 0.156],
        'C':[-0.132, 0.174, 0.070, 0.565, -0.374],
        'Q':[0.149, -0.184, -0.030, 0.035, -0.112],
        'E':[0.221, -0.280, -0.315, 0.157, 0.303],
        'G':[0.218, 0.562, -0.024, 0.018, 0.106],
        'H':[0.023, -0.177, 0.041, 0.280, -0.021],
        'I':[-0.353, 0.071, -0.088, -0.195, -0.107],
        'L':[-0.267, 0.018, -0.265, -0.274, 0.206],
        'K':[0.243, -0.339, -0.044, -0.325, -0.027],
        'M':[-0.239, -0.141, -0.155, 0.321, 0.077],
        'F':[-0.329, -0.023, 0.072, -0.002, 0.208],
        'P':[0.173, 0.286, 0.407, -0.215, 0.384],
        'S':[0.199, 0.238, -0.015, -0.068, -0.196],
        'T':[0.068, 0.147, -0.015, -0.132, -0.274],
        'W':[-0.296, -0.186, 0.389, 0.083, 0.297],
        'Y':[-0.141, -0.057, 0.425, -0.096, -0.091],
        'V':[-0.274, 0.136, -0.187, -0.196, -0.299],
        '#':[0,0,0,0,0]
    }
    coding = []
    Matr = []
    for seq in short_seqs:
        coding = []
        for c in seq:
            coding.append(np.array(d[c]))
            # coding = np.array(coding)
        coding = np.array(coding)
        Matr.append(coding)
    Matr = np.array(Matr)
    return Matr

def sgt_coding(short_seqs):

    res_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '#']
    coding=[]

    for seq in short_seqs:
        sgt = SGT(flatten=False)
        # print(seq)
        sequence = np.array([i for i in seq])
        # print(sequence,sequence.shape)
        matrix = sgt.fit(sequence)
        Matr = np.zeros((21, 21))
    # print(Matr)
        for i in range(21):
            for j in range(21):
                try:
                    Matr[i, j] = matrix.loc[res_list[i], res_list[j]]
                except:
                    continue
        coding.append(Matr)
        # print(Matr)
    coding=np.array(coding)
    return coding

def getMatrixLabel(positive_position_file_name,sites, window_size, code,empty_aa = '*'):
    # input format   label, proteinName, postion,shortsequence,
    prot = []  # list of protein name
    pos = []  # list of position with protein name
    rawseq = []
    all_label = []

    short_seqs = []
    start = int(50-(window_size-1)/2)
    end=int(51+(window_size-1)/2)
    file=pd.read_excel(positive_position_file_name)
    for index,row in file.iterrows():

        sseq1 = row['SEQ'].split(',')
        sseq=sseq1[start:end]
        # rawseq.append(row['seq'])
        # center = sseq[position - 1]
        short_seqs.append(''.join(sseq))
        if row['regular']==0:
            all_label.append(0)
        else:
            all_label.append(1)


    targetY =np.array(all_label)
    # Matr = np.zeros((len(short_seqs), window_size))
    if code=='one_hot':
        Matr=onehot_coding(short_seqs,window_size)
        return Matr, targetY, prot
    elif code=='blosum62':
        Matr=blosum62_coding(short_seqs)
        return Matr, targetY, prot
    elif code=='SGT':
        Matr=sgt_coding(short_seqs)
        return Matr, targetY, prot
    else:
        Matr=phy_che_coding(short_seqs)
        return Matr, targetY, prot

def get_PTM_feature(feature_file_name,row_start,row_end):
    file=pd.read_excel(feature_file_name)
    feature=[]
    for index,row in file.iterrows():
        feature.append(np.array(row[row_start:row_end]))
    data = np.array(feature)
    print(data.shape)
    return data

