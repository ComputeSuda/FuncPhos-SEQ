import os
import pandas  as pd
import numpy as np
def pssm_feature(file_name,out_name):
    file_infor=pd.read_excel(file_name)
    pssm_feature=[]
    for index,row in file_infor.iterrows():
        print(index)
        acc_name = row['ACC_ID']
        site=int(row['RES'][1:])
        feature_name='PSSM_handle/'+acc_name+ '_pssm.txt'

        if os.path.exists(feature_name):
            feature = pd.read_table(feature_name,header=None)
            feature_infor=list(feature.iloc[site-1,:])
            pssm_feature.append(feature_infor)
        else:
             continue
        pssm_feature_npy=np.array(pssm_feature)
        np.save(out_name,pssm_feature_npy)


def pssm_code(file_name,win,out_file):
    file_infor = pd.read_excel(file_name)
    pssm_feature = []
    for index, row in file_infor.iterrows():
        print(index)
        acc_name = row['ACC_ID']
        site = int(row['RES'][1:])
        feature_name = 'PSSM_handle/' + acc_name + '_pssm.txt'
        one_feature=np.zeros((2*win+1,21))
        if os.path.exists(feature_name):
            feature = pd.read_table(feature_name, header=None)
            feature_infor = list(feature.iloc[site - 1, :])
            pssm_feature.append(feature_infor)
        else:
            continue
        pssm_feature_npy = np.array(pssm_feature)
        np.save(out_name, pssm_feature_npy)

if __name__ == '__main__':
    file_name=r'01dataset\phos_set.xlsx'
    out_name='pssm_feature.txt'
    pssm_feature(file_name,out_name)
