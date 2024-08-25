import torch
import numpy as np
from torch import nn
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
l = 24
hidden_size1 = 48
hidden_size2 = 24
hidden_size3 = 12
hidden_size4 = 6


def argen(ar, psi, delta,gamma, length) :

    e = np.random.normal(loc=0, scale = 1,size = length)
    sigma = math.sqrt(1 / (1 - pow(ar, 2)))
    x = np.array(np.repeat(0, length), dtype= np.float64)
    x[0] = e[0]
    z = np.array(np.repeat(0, length), dtype=np.float64)

    for i in range(1, psi):
        x[i] = ar * x[i-1] + e[i]
        z[i] = x[i]
    for i in range(psi,len(x)):
        x[i] = ar * x[i - 1] + gamma*e[i]
        z[i] = x[i]
    for i in range(psi,len(z)):
        z[i] = z[i] + delta * sigma


    return z


class NeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size,num_layers,device):
        super(NeuralNetwork, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size= input_size,
                          hidden_size = hidden_size1,

                          num_layers = num_layers,
                          nonlinearity= "relu",
                          batch_first= True)
        self.fc = nn.Sequential(nn.Linear(hidden_size1,hidden_size2),
                                nn.GELU(),
                                nn.Linear(hidden_size2,hidden_size3),
                                nn.GELU(),
                                nn.Linear(hidden_size3,hidden_size4),
                                nn.GELU(),
                                nn.Linear(hidden_size4,1),
                                nn.Sigmoid()
                                )


    def forward(self, x):
        h0 = torch.zeros(x.size()[0], self.hidden_size).to(device)  # 초기 hidden state 설정하기.
        out, _ = self.rnn(x, h0)  # out: RNN의 마지막 레이어로부터 나온 output feature 를 반환한다. hn: hidden state를 반환한다.
        out = out.reshape(out.shape[0], -1)  # many to many 전략
        out = self.fc(out)
        return out



model = NeuralNetwork(input_size = l, hidden_size = hidden_size1, num_layers = 1, device = device).to(device)

model.load_state_dict(torch.load(f'../model_b_ar1_l24_v5.pt'))



def arl(ar,delta,gamma, run, length,cl) :
    rl = np.array([], dtype=np.float64)

    for i in tqdm(range(run)) :
        y = argen(ar=ar, psi=l-1, delta=delta, gamma = gamma,length=length)
        a = np.array([length-l])
        x = np.empty(shape=(length-l, l))
        for j in range(length-l):
            x[j] = y[j: j + l]
        x = torch.FloatTensor(x).to(device)

        model.eval()
        with torch.no_grad():
            for j in range(0,len(x)):
                input = x[[j]]

                output = model(input)

                if output[0] > cl :

                    a = np.array([j + 1])
                    #print(output)
                    break
                elif j == len(x):
                    a = len(x)

            #print(a)
            rl = np.append(rl,a)

    arl = np.mean(rl)
    #print(f'cl = {cl}')
    return arl



def arl1(ar,run,length,cl):
    a5 = arl(ar, 0.5, 1, run, length, cl)
    a1 = arl(ar, 1, 1, run, length, cl)
    a2 = arl(ar, 2, 1, run, length, cl)
    a3 = arl(ar, 3, 1, run, length, cl)
    b5 = arl(ar, 0.5, 1.5,run, length, cl)
    b1 = arl(ar, 1, 1.5, run, length, cl)
    b2 = arl(ar, 2, 1.5, run, length, cl)
    b3 = arl(ar, 3, 1, run, length, cl)
    c1 = arl(ar, 0, 1.5, run, length, cl)
    c2 = arl(ar, 0, 2, run, length, cl)
    c3 = arl(ar, 0, 3, run, length, cl)
    print(f'0.5: {a5}, 1:{a1},2:{a2},3:{a3}')
    print(f'0.5:{b5},1:{b1},2:{b2},3:{b3}')
    print(f'1.5:{c1},2:{c2},3:{c3}')

import pandas as pd

a = pd.read_csv('C:/Users/JPJ/PycharmProjects/sqc/heartrate.csv')
a = np.array(a)
a = np.concatenate(a)
def real(data,cl) :
    outputs = torch.zeros(size=((len(data),1))).to(device)
    length = len(data)

    x = np.empty(shape=(length - l, l))
    for j in range(length - l):
        x[j] = data[j: j + l]
        x[j] = (x[j] - np.mean(x[j]))
    x = torch.FloatTensor(x).to(device)

    model.eval()
    with torch.no_grad():
        for j in range(0, len(x)):
            input = x[[j]]
            output = model(input)
            outputs[j+24] = output
            if output >= cl:
                b = np.array([j -1+ l])
                print(b)
            elif j == len(x)-1:
                b = len(a)

        outputs = outputs.to('cpu')
        outputs = outputs.numpy()
    return b, outputs, x

y = a[:150]

sigma = 0.5 * math.sqrt(np.var(y))
b,c,d = real(y,0.9999)

plt.plot(c)
plt.title('procedure')
plt.ylim(0.6)
plt.hlines(0.9999, xmin = 0, xmax= len(c),colors='red')
plt.show()
