from keras import backend as K
from keras import models
from keras import layers, Input, regularizers
from keras import optimizers
from keras import activations
from keras.models import load_model
from cal_metric1 import cal_two_class_metric, cal_two_class_metric_capsule
import numpy as np
# from keras.engine.topology import Layer
from keras.layers import Layer
import random
import pandas as pd
from keras.layers import Dropout
from dataprocess import *
import os
from sklearn.model_selection import train_test_split
def sample_data(data, sample_num):
    """
    Sampling data. The number of sampling is sample_num.
    """
    random_index = random.sample(range(0, data.shape[0]), sample_num)
    sample_data = data[random_index]
    data = np.delete(data, random_index, 0)  # 删除已经采样的行
    return data, sample_data
def create_1_1_data(data_file,res_type,out_file):
    """
    Create training and test sets
    """
    data = pd.read_excel(data_file)
    data_0 = []#
    data_1 = []
    for indexs,row  in data.iterrows():
        if row['res_in_seq'] in res_type:
            # print(indexs)
            uniprot = row['ACC_ID']
            position = int(row['RES'][1:])
            uniprot_position = uniprot + '_' + str(position)
            functionality = row['regular']
            if functionality == 0:
                data_0.append(uniprot_position)
            else :
                data_1.append(uniprot_position)
    sample_num=len(data_1)
    data_0, sample_0 = sample_data(np.array(data_0), sample_num)
    sample= sample_0.tolist() + data_1
    final_data = []
    column_name = list(data.columns)[:]
    for index, row in data.iterrows():
        # print(index)
        infor = row['ACC_ID'] + '_' + str(row['RES'][1:])
        if infor in sample:
            final_data.append(list(row[:]))
    dataframe = pd.DataFrame(final_data, columns=column_name)
    dataframe.to_excel(out_file, index=False)
def create_specific_function_1_1(data_file,specific_function,res_type,plddt_score,out_file):

    """
    Create training and test sets
    """
    data = pd.read_excel(data_file)
    data_0 = []#
    data_1 = []
    for indexs,row in data.iterrows():
        if row['res_in_seq'] in res_type:
            # print(indexs)
            uniprot = row['ACC_ID']
            position = int(row['RES'][1:])
            uniprot_position = uniprot + '_' + str(position)
            functionality = row['ON_FUNCTION']
            plddt = row['plddt']
            regulatory=row['regular']
            if plddt < plddt_score:
                if regulatory == 0:
                    data_0.append(uniprot_position)
                    continue
                if specific_function in functionality:
                    data_1.append(uniprot_position)
    sample_num=len(data_1)
    data_0, sample_0 = sample_data(np.array(data_0), sample_num)
    sample= sample_0.tolist() + data_1
    final_data = []
    column_name = list(data.columns)[:]
    for index, row in data.iterrows():
        print(index)
        infor = row['ACC_ID'] + '_' + str(row['RES'][1:])
        if infor in sample:
            final_data.append(list(row[:]))
    dataframe = pd.DataFrame(final_data, columns=column_name)
    dataframe.to_excel(out_file, index=False)

    def split_dataset(file_name, train_file_name, test_file_name):
        traffic_feature = []
        traffic_target = []
        csv_file = pd.read_excel(file_name)  # csv格式为read_csv('file_name')
        for index, row in csv_file.iterrows():
            # data.append([float(i) for i in list(row)])
            traffic_feature.append(list(row)[:])  # 特征数据选择，[1:]表示从第二列开始后面所有，若要选择m到n列，改为[m-1,n-1]
            if row['regular'] == 1:
                traffic_target.append(1)  # label标签为0或1
            else:
                traffic_target.append(0)
        feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target,
                                                                                  test_size=0.3, random_state=0)
        train_data = pd.DataFrame(feature_train, columns=list(csv_file.columns)[:])
        train_data.to_excel(train_file_name, index=False)
        test_data = pd.DataFrame(feature_test, columns=list(csv_file.columns)[:])
        test_data.to_excel(test_file_name, index=False)
def split_dataset(file_name,train_file_name,test_file_name):
    traffic_feature = []
    traffic_target = []
    csv_file = pd.read_excel(file_name)  # csv格式为read_csv('file_name')
    for index, row in csv_file.iterrows():
        # data.append([float(i) for i in list(row)])
        traffic_feature.append(list(row)[:])  # 特征数据选择，[1:]表示从第二列开始后面所有，若要选择m到n列，改为[m-1,n-1]
        if row['regular'] == 1:
            traffic_target.append(1)  # label标签为0或1
        else:
            traffic_target.append(0)
    feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target,
                                                                              test_size=0.3, random_state=0)
    train_data = pd.DataFrame(feature_train, columns=list(csv_file.columns)[:])
    train_data.to_excel(train_file_name, index=False)
    test_data = pd.DataFrame(feature_test, columns=list(csv_file.columns)[:])
    test_data.to_excel(test_file_name, index=False)

def get_dataset(win,sites,code,res_conding,train_file_name,test_file_name,train_feature,test_feature,train_pssm2d,test_pssm2d):
    seq_encoding = ['seq', 'pssm', 'all']
    train_feature = np.load(train_feature,allow_pickle=True).astype(float)
    test_feature = np.load(test_feature,allow_pickle=True).astype(float)

    if res_conding in seq_encoding:
        if res_conding=='seq':
            train_data, train_label, ids = getMatrixLabel(train_file_name, sites, win,code)
            test_data, test_label, ids = getMatrixLabel(test_file_name, sites, win,code)
        elif res_conding=='pssm':
            train_data = np.load(train_pssm2d).astype(float)
            test_data = np.load(test_pssm2d).astype(float)
            train_label = np.load('regular_train_st_11_label.npy')
            test_label = np.load('regular_test_st_11_label.npy')
        else:
            train_data1, train_label, ids = getMatrixLabel(train_file_name, sites, win, code)
            test_data1, test_label, ids = getMatrixLabel(test_file_name, sites, win, code)
            train_data2 = np.load(train_pssm2d).astype(float)
            test_data2 = np.load(test_pssm2d).astype(float)
            train_data = np.concatenate([train_data1, train_data2], axis=1)
            test_data = np.concatenate([test_data1, test_data2], axis=1)
        print(train_data.shape, train_label.shape)
        print(test_data.shape, test_label.shape)
        return train_data, train_feature, train_label, test_data, test_feature, test_label
    else:
        train_data1, train_label, ids = getMatrixLabel(train_file_name, sites, win, code)
        test_data1, test_label, ids = getMatrixLabel(test_file_name, sites, win, code)
        train_data2 = np.load(train_pssm2d).astype(float)
        test_data2 = np.load(test_pssm2d).astype(float)
        return train_data1, train_data2,train_feature, train_label, test_data1, test_data2,test_feature, test_label
def get_dataset1(win,sites,code,res_conding,train_file_name,test_file_name,train_feature1,test_feature1,train_feature2,test_feature2,train_pssm2d,test_pssm2d):
    seq_encoding = ['seq', 'pssm', 'all']
    train_feature = np.load(train_feature1,allow_pickle=True).astype(float)
    test_feature = np.load(test_feature1,allow_pickle=True).astype(float)
    train_feature1 = np.load(train_feature2, allow_pickle=True).astype(float)
    test_feature1 = np.load(test_feature2, allow_pickle=True).astype(float)
    train_data1, train_label, ids = getMatrixLabel(train_file_name, sites, win, code)
    test_data1, test_label, ids = getMatrixLabel(test_file_name, sites, win, code)
    train_data2 = np.load(train_pssm2d).astype(float)
    test_data2 = np.load(test_pssm2d).astype(float)
    return train_data1, train_data2,train_feature, train_feature1,train_label, test_data1, test_data2,test_feature,test_feature1, test_label

class Position_Embedding(Layer):

    def __init__(self, size=None, mode='concat', **kwargs):

        self.size = size

        self.mode = mode

        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):

        if (self.size == None) or (self.mode == 'concat'):
            self.size = int(x.shape[-1])

        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)

        position_j = K.expand_dims(position_j, 0)

        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1

        position_i = K.expand_dims(position_i, 2)

        position_ij = K.dot(position_i, position_j)

        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)

        if self.mode == 'sum':

            return position_ij + x

        elif self.mode == 'concat':

            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):

        if self.mode == 'sum':

            return input_shape

        elif self.mode == 'concat':

            return (input_shape[0], input_shape[1], input_shape[2] + self.size)
class Self_Attention(Layer):
    def __init__(self, output_dim=128, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (self.output_dim ** 0.5)
        QK = K.softmax(QK)
        V = K.batch_dot(QK, WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)
from keras import initializers, regularizers, constraints
class Attention(Layer):
    def __init__(self, step_dim=5,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim
class Extract_outputs(Layer):
    def __init__(self,outputdim=0, **kwargs):
        #self.input_spec = [InputSpec(ndim='3+')]
        self.outputdim=outputdim
        super(Extract_outputs, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return tuple([None,input_shape[1], self.outputdim])
    
    def call(self, x, mask=None):
        x=x[:,:,:self.outputdim]
        #return K.batch_flatten(x)
        return x
    
    def get_config(self):
        config = {
        'outputdim': self.outputdim
        
        }
        base_config = super(Extract_outputs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class Extract_weight_c(Layer):
    def __init__(self,outputdim, **kwargs):
        #self.input_spec = [InputSpec(ndim='3+')]
        self.outputdim=outputdim
        super(Extract_weight_c, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return tuple([None,input_shape[1], input_shape[-1]-self.outputdim])
    
    def call(self, x, mask=None):
        x=x[:,:,self.outputdim:]
        #return K.batch_flatten(x)
        return x
    
    def get_config(self):
        config = {
        'outputdim': self.outputdim
        
        }
        base_config = super(Extract_weight_c, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def build_seq_pssm_feature2_model(dim1, dim2, dim3,dim4,dim5,dim6):
    input_x = Input(shape=(dim1, dim2), name='seq')  # , dim3))

    x = layers.Conv1D(filters=200, kernel_size=1, padding='valid', activation='relu')(input_x)
    # x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.75)(x)
    x = layers.Conv1D(filters=200, kernel_size=9, padding='valid', activation='relu')(x)
    # x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.75)(x)
    #  x = layers.Conv1D(filters=100,kernel_size=9, padding='valid', activation='relu')(x)
    # x = layers.MaxPooling2D(pool_size=(2,2))(x)
    #   x = Dropout(0.68)(x)
    x = layers.SeparableConv1D(filters=200, kernel_size=5, padding='valid', activation='relu')(x)
    x = Position_Embedding()(x)
    # x = Self_Attention(128)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)

    #
    input_y = Input(shape=(dim3, dim4), name='pssm')
    y = layers.Conv1D(filters=200, kernel_size=1, padding='valid', activation='relu')(input_y)
    # x = layers.MaxPooling2D(pool_size=(2,2))(x)
    y = Dropout(0.75)(y)
    y = layers.Conv1D(filters=200, kernel_size=9, padding='valid', activation='relu')(y)
    # x = layers.MaxPooling2D(pool_size=(2,2))(x)
    y = Dropout(0.75)(y)
    #  x = layers.Conv1D(filters=100,kernel_size=9, padding='valid', activation='relu')(x)
    # x = layers.MaxPooling2D(pool_size=(2,2))(x)
    #   x = Dropout(0.68)(x)
    y = layers.Conv1D(filters=200, kernel_size=5, padding='valid', activation='relu')(y)
    # x = Self_Attention(128)(x)
    y = Position_Embedding()(y)
    y = Self_Attention(128)(y)
    y = Dropout(0.5)(y)
    y = layers.Flatten()(y)
    y = layers.Dense(64, activation='relu')(y)

    # seq_out1 = layers.concatenate([x, y])
    # seq_out = layers.Dot(axes=1)([q,seq_out1])
    # seq_out=layers.Flatten()(seq_out)
    input_z = Input(shape=(dim5,), name='ppi')
    z= layers.Dense(64, activation='relu')(input_z)
    # z = layers.Reshape((1, 64))(z)

    input_q = Input(shape=(dim6,), name='sf')

    out = layers.Concatenate()([x,y,z,input_q])
    # out = layers.Flatten()(out)

    # seq=layers.concatenate([x,y,input_q])
    # seq = layers.Reshape((1, 131))(seq)
    # layers.Multiply
    # y=layers.dot([x,y],1)
    # out = layers.Dot(axes=1)([seq,z])
    # out = layers.Flatten()(out)
    out = layers.Dense(64, activation='relu')(out)
    out= layers.Dense(30, activation='relu')(out)
    # x = Dropout(0.75)(x)
    out_x = layers.Dense(1, activation='sigmoid')(out)
    model = models.Model([input_x,input_y,input_z,input_q], out_x)
    return model

def train_seq_pssm_feature2_model(train_data1, train_data2,feature_train,feature_train1,train_label, model_name,model_path):
    epochs = 1
    batch_size =256#一次训练所抓取的数据样本数量

    dim1 = train_data1.shape[1]
    dim2 = train_data1.shape[2]
    dim3 = train_data2.shape[1]
    dim4 = train_data2.shape[2]
    dim5 = feature_train.shape[1]
    dim6=feature_train1.shape[1]
    # print(dim1, dim2, dim3)
    # sys.exit()
    for model_num in range(5):
        print(model_num)
    if model_name == 'cnn':
        model = build_seq_pssm_feature2_model(dim1, dim2, dim3,dim4,dim5,dim6)
        model.compile(optimizer=optimizers.RMSprop(lr=0.0001, epsilon=1e-08), loss='binary_crossentropy', metrics=['acc'])
        save_file = model_path+ str(model_num) + '.h5'
        print(model.summary())
        # print(data.shape, label.shape)
        model.fit([train_data1,train_data2,feature_train,feature_train1], train_label, epochs=200, batch_size=64, verbose=1)
        model.save(save_file)
def test_seq_pssm_feature2_model(test_data1, test_data2,feature_test,feature_test1,test_label, model_name,model_path,result_path):
    pred_prob = None
    # np.savetxt('label.txt', test_label)
    for model_num in range(1):
        for i in range(5):

            model = load_model(model_path+ str(i) + '.h5', compile=False,custom_objects={'Self_Attention': Self_Attention,'Position_Embedding':Position_Embedding,'Attention':Attention})
            file = result_path+ str(i) + '.txt'
            pred_prob = model.predict([test_data1,test_data2,feature_test,feature_test1])
            prob=np.array(pred_prob)
            pro_file=result_path+ str(i) + '.npy'
            np.save(pro_file,prob)
                # np.savetxt('label.txt',test_label)
            matrix, metric = cal_two_class_metric(test_label, pred_prob)

            print(matrix)
            print(metric)
            score = []
            for key in metric.keys():
                score.append(metric[key])

            result = pd.read_excel('result.xlsx')
            # print(result1)
            column = result_path.split('/')[-1] + str(i)
            result[column] = score
            # print(result)
            result.to_excel('result.xlsx', index=False)



if __name__ == "__main__":
    RES_TYPE = ['Y']


    # sample  and split regular data
    print('start sample data')
    for i in [2]:
        predict_file_name='activity_st2_'+str(i)
        split_train_file = predict_file_name+'_train.xlsx'
        split_test_file = predict_file_name+'_test.xlsx'
        # specific_function = 'activity'
        #function
        # create_1_1_data(in_file_name, RES_TYPE, out_file_name)
        #specific_function
        # create_specific_function_1_1(in_file_name, specific_function, RES_TYPE, 500, out_file_name)
        # split_dataset(out_file_name, split_train_file, split_test_file)
        # print('start get feature')
        # get_all_feature(split_train_file)
        # get_all_feature(split_test_file)
        # pssm_path = 'PSSM_structure_handle'
        # get_pssm_2d_feature(split_train_file,pssm_path)
        # get_pssm_2d_feature(split_test_file,pssm_path)

        # print('start merge feature')
        # get_merge_feature(split_train_file)
        # get_merge_feature(split_test_file)

        test_outfile1 = split_test_file[:-5] + '_pssm_evol3.npy'
        train_outfile1 = split_train_file[:-5] + '_pssm_evol3.npy'



        print('traing')
        win_list = [31]
        for win in win_list:
            model_name = "cnn"
            train_pssm2d = split_train_file[:-5] + '_pssm2d'+str(win)+'.npy'
            test_pssm2d = split_test_file[:-5] + '_pssm2d'+str(win)+'.npy'

            code = ['SGT']
            sites = 'st'
            res_coding = 'seq_pssm2'
            seq_encoding=['seq', 'pssm', 'all']
            model_path = 'model/activity_st_'+str(win)+'_'+str(i)+'_'
            result_path = 'model/activity_st_'+str(win)+'_'+str(i)+'_'



            train_ppi = split_train_file[:-5] + '_ppi500.npy'
            test_ppi = split_test_file[:-5] + '_ppi500.npy'
            train_data1, train_data2, train_feature, train_feature1, train_label, test_data1, test_data2, test_feature, test_feature2, test_label = get_dataset1(
               31, sites, 'one_hot', res_coding, split_train_file, split_test_file, train_ppi, test_ppi, train_outfile1,test_outfile1, train_pssm2d, test_pssm2d)
            train_seq_pssm_feature2_model(train_data1, train_data2, train_feature,train_feature1, train_label, model_name, model_path)
            test_seq_pssm_feature2_model(test_data1, test_data2, test_feature, test_feature2, test_label,model_name, model_path,result_path)


