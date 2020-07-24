import torch
import model
import torch.optim as optim
import layer
import numpy as np


def main():
    #データのloading
    X = np.array([[0.1,0.9,1.], [0.1,0.9,1.], [0.9,0.1,1.], [0.9,0.1,1.],[0.9,0.1,1.]])
    Y = np.array([[0.,1.,1.], [0.,1.,1.], [1.,0.,1.], [1.,0.,1.],[1.,0.,1.]])
    X, Y = torch.FloatTensor(X).cuda(), torch.FloatTensor(Y).cuda()

    # モデルの作成
    torch.cuda.manual_seed(1000)
    m = model.DummyModel(din = X.shape[1], dout=Y.shape[1])

    # GPUに転送
    m.cuda()
    loss_f = layer.FrobeniusNorm()
    optimizer = optim.Adam(m.parameters(), lr=0.01, 
                                        weight_decay=5e-4) #lrが学習係数

    #訓練
    for i in range(100):
        m.train()
        optimizer.zero_grad()
        X_ = m(X)
        loss = loss_f(X_, Y)
        loss.backward()
        optimizer.step()
        if(i%10 == 0):
            print(X_)

    # 保存
    #torch.save(m, 'model')
    torch.save(m.state_dict(), 'model')


if __name__ == '__main__':
    main()