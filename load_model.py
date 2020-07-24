import numpy as np
import torch
import model
import torch.optim as optim
import torch.nn.functional as F


def main():
    X = np.array([[0.1,0.9,1.], [0.1,0.9,1.], [0.9,0.1,1.], [0.9,0.1,1.],[0.9,0.1,1.]])
    Y = np.array([[0.,1.,1.], [0.,1.,1.], [1.,0.,1.], [1.,0.,1.],[1.,0.,1.]])
    label = np.array([0,0,1,1,1])
    X, Y = torch.FloatTensor(X).cuda(), torch.FloatTensor(Y).cuda()
    label = torch.LongTensor(label).cuda()
    m = model.DummyModel(X.shape[1], Y.shape[1]).cuda()
    m.load_state_dict(torch.load('model'))
    optimizer = optim.Adam(m.parameters(), lr=0.01, 
                                        weight_decay=5e-4)

    for param in m.parameters():
        # 型を調べるとCPUかGPUかわかる。
        # CPU: torch.FloatTensor
        # GPU: torch.cuda.FloatTensor
        print(type(param.data))
    print(m.aff)
    m.eval()
    X_ = m(X)
    print(X_, end='\n---------next model2------------\n\n')

    #事前学習したモデル
    n_of_class = torch.max(label).to('cpu').item()+1
    m2 = model.DummyModel2(base=m, nclass=n_of_class)
    m2.cuda()
    print('m2(gc) ', end='')
    print(m2.aff)
    m2.eval()
    Label, X_ = m2(X)
    print('m2(X_) ', end='')
    print(X_)
    print('m2(label) ', end='')
    print(Label)
    print('-----------------train start-----------------')

    #訓練
    for i in range(100):
        m2.train()
        optimizer.zero_grad()
        pred, X_ = m2(X)
        loss = F.nll_loss(pred, label)
        loss.backward()
        optimizer.step()
        if(i%10 == 0):
            print(X_)
            print(pred)


if __name__ == '__main__':
    main()