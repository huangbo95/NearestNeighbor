import numpy as np

def unpickle(file): #从文件中获取数据，并以numpy数组存放
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
fo=unpickle('data_batch_1')
data=np.array(fo[b'data'])
labels=np.array(fo[b'labels'])
labels=labels.reshape(labels.shape[0],1)

class NearestNeighbor(object):
    def __init__(self):
        pass
    def train(self,Xtr,Ytr):  #训练数据函数
        self.Xtr=Xtr
        self.Ytr=Ytr
    def predict(self,Xte):  #测试数据函数
        num_test=Xte.shape[0]
        Y_pred=np.zeros(1000,dtype = self.Ytr.dtype)
        Y_pred=Y_pred.reshape(Y_pred.shape[0],1)
        for i in range(num_test):
            distinct=np.sum(np.abs(self.Xtr-Xte[i,:]),axis=1)
            min_index=np.argmin(distinct)
            Y_pred[i]=self.Ytr[min_index]
        return Y_pred

if __name__=='__main__':
    Xtr=data[:9000,]     #对数据进行分割，将前9000行数据作为训练集，后1000行作为测试集
    Xte=data[9000:,]
    Ytr=labels[:9000,]
    Yte=np.zeros(1000,dtype = Ytr.dtype)
    Yte=Yte.reshape(Yte.shape[0],1)

    nn=NearestNeighbor()
    nn.train(Xtr,Ytr)
    Yte=nn.predict(Xte)
    print((np.mean(Yte==labels[9000:,]))) #预测的准确率