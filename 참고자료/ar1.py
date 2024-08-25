import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pytorchtools import EarlyStopping
import math


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)
early_stopping = EarlyStopping(patience = 5, verbose = True)

def saveModel():
    torch.save(model.state_dict(), f'../model_b_ar1_l24_v5.pt')
length = 24
hidden_size1 = 48
hidden_size2 = 24
hidden_size3 = 12
hidden_size4 = 6
learning_rate = 1e-6
epoch = 400
trainrun = 50
testrun = 25
validrun =25
phi1 = np.array([np.repeat(0,8),
                 np.repeat(0.25,8),
                 np.repeat(0.5,8),
                 np.repeat(0.75,8),
                 np.repeat(0.95,8)])
phi1 = np.concatenate(phi1)

psi1 = np.array([0, 10, 15, 20, 23, 14, 17, 23,
                 0, 10, 15, 20, 23, 14, 17, 23,
                 0, 10, 15, 20, 23, 14, 17, 23,
                 0, 10, 15, 20, 23, 14, 17, 23,
                 0, 10, 15, 20, 23, 14, 17, 23,])

de1 = np.array([0, 0.5, 1, 2, 3, 0, 0, 0,
                0, 0.5, 1, 2, 3, 0, 0, 0,
                0, 0.5, 1, 2, 3, 0, 0, 0,
                0, 0.5, 1, 2, 3, 0, 0, 0,
                0, 0.5, 1, 2, 3, 0, 0, 0,])


ga = np.array([1, 1, 1, 1, 1, 1.5, 2, 3,
               1, 1, 1, 1, 1, 1.5, 2, 3,
               1, 1, 1, 1, 1, 1.5, 2, 3,
               1, 1, 1, 1, 1, 1.5, 2, 3,
               1, 1, 1, 1, 1, 1.5, 2, 3,])


# 시계열 데이터 생성

def ar(ar1, delta,gamma, psi,length,run):
    y = np.empty(shape=(run, length))
    sigma = math.sqrt(1 / (1 - pow(ar1, 2)))
    for j in range(0, run):
        e = np.random.normal(loc=0, scale=1, size=length)
        x = np.array(np.repeat(0, length), dtype=np.float64)

        x[0] = e[0]

        for i in range(1, psi):
            x[i] = ar1 * x[i - 1] + e[i]
        for i in range(psi,len(x)):
            e[i] = gamma * e[i]
            x[i] = ar1 * x[i-1] + e[i]
        for i in range(psi,len(x)):
            x[i] = x[i] + delta*sigma
        y[j] = x

    return y
def totaldat(run,length):
    y = np.empty(shape=(len(phi1), run, length))
    for i in range(len(phi1)):
        y[i]= ar(phi1[i],de1[i],ga[i],psi1[i],length,run)

    return y
train_x = totaldat(trainrun,length)
train_x = train_x.reshape(trainrun*len(phi1),length)
train_y =  [np.repeat(0,trainrun),np.repeat(1,trainrun*7),
            np.repeat(0,trainrun),np.repeat(1,trainrun*7),
            np.repeat(0,trainrun),np.repeat(1,trainrun*7),
            np.repeat(0,trainrun),np.repeat(1,trainrun*7),
            np.repeat(0,trainrun),np.repeat(1,trainrun*7),]
train_y =  np.concatenate(train_y)
train_y = train_y.reshape(2000,1)

train_x = torch.FloatTensor(train_x).to(device)
train_y = torch.FloatTensor(train_y).to(device)



test_x = totaldat(run = testrun, length = length)
test_x = test_x.reshape(testrun*len(phi1),length)
test_y = [np.repeat(0,testrun),np.repeat(1,testrun*7),
          np.repeat(0,testrun),np.repeat(1,testrun*7),
          np.repeat(0,testrun),np.repeat(1,testrun*7),
          np.repeat(0,testrun),np.repeat(1,testrun*7),
          np.repeat(0,testrun),np.repeat(1,testrun*7),]
test_y = np.concatenate(test_y)
test_y = test_y.reshape(1000,1)

test_x = torch.FloatTensor(test_x).to(device)
test_y = torch.FloatTensor(test_y).to(device)




valid_x = totaldat(run = validrun, length = length)
valid_x = valid_x.reshape(validrun*len(phi1),length)
valid_y = [np.repeat(0,testrun),np.repeat(1,testrun*7),
            np.repeat(0,testrun),np.repeat(1,testrun*7),
            np.repeat(0,testrun),np.repeat(1,testrun*7),
            np.repeat(0,testrun),np.repeat(1,testrun*7),
            np.repeat(0,testrun),np.repeat(1,testrun*7),]
valid_y = np.concatenate(valid_y)
valid_y = valid_y.reshape(1000,1)

train_x = torch.FloatTensor(train_x).to(device)
train_y = torch.FloatTensor(train_y).to(device)
test_x = torch.FloatTensor(test_x).to(device)
test_y = torch.FloatTensor(test_y).to(device)
valid_x = torch.FloatTensor(valid_x).to(device)
valid_y = torch.FloatTensor(valid_y).to(device)

trainset = TensorDataset(train_x, train_y)
trainloader = DataLoader(trainset, shuffle=True)
testset = TensorDataset(test_x, test_y)
testloader = DataLoader(testset,shuffle = False)
validset = TensorDataset(valid_x, valid_y)
validloader = DataLoader(validset,shuffle = False)




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
        self.fc = nn.Sequential(nn.Linear(hidden_size,hidden_size2),
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




model = NeuralNetwork(input_size = length, hidden_size = hidden_size1 ,num_layers = 1, device = device).to(device)


optimizer = optim.Adam(model.parameters(),lr = learning_rate)
criterion = nn.MSELoss()



loss_ = []
n = len(trainloader)
valoss_ = []
logger = {"train_loss": list(),
          "validation_loss": list(),

          }




#training
def training(epochs) :

    for epoch in range(epochs):
        running_train_loss = 0.0
        running_vall_loss = 0.0
        total = 0

        for data in trainloader:
            model.train()
            inputs, outputs = data
            optimizer.zero_grad()# zero the parameter gradients
            predicted_outputs = model(inputs)  # predict output from the model
            train_loss = criterion(predicted_outputs, outputs)  # calculate loss for the predicted output
            train_loss.backward()  # backpropagate the loss
            optimizer.step()  # adjust parameters based on the calculated gradients
            running_train_loss += train_loss.item()  # track the loss value
        loss_.append(running_train_loss / n)
        with torch.no_grad():
            model.eval()
            for data in validloader:
                inputs, outputs = data
                predicted_outputs = model(inputs)
                val_loss = criterion(predicted_outputs, outputs)

                # The label with the highest value will be our prediction
                _, predicted = torch.max(predicted_outputs,1)
                running_vall_loss += val_loss.item()
                total += outputs.size(0)
                val_loss_value = running_vall_loss/len(validloader)
        valoss_.append(val_loss_value)

        avgtrainloss = np.mean(loss_)
        avgvalidloss = np.mean(valoss_)
        print('epoch', epoch + 1)
        print(f'train loss : {avgtrainloss}, validation loss : {avgvalidloss}')
        early_stopping(avgvalidloss, model)
        if early_stopping.early_stop:  # 조건 만족 시 조기 종료
            break
    saveModel()
training(epochs = epoch)


plt.plot(loss_)
plt.plot(valoss_)
plt.legend(['Train','Valid'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#evaluation

def evaluation(dataloader):

  predictions = torch.tensor([], dtype=torch.float64,device = device) # 예측값을 저장하는 텐서.
  actual = torch.tensor([], dtype=torch.float64, device = device) # 실제값을 저장하는 텐서.

  with torch.no_grad():
    model.eval() # 평가를 할 땐 반드시 eval()을 사용해야 한다.

    for data in dataloader:
        inputs, values = data



        outputs = model(inputs)

        predictions = torch.cat((predictions, outputs), 0) # cat함수를 통해 예측값을 누적.
        actual = torch.cat((actual, values), 0) # cat함수를 통해 실제값을 누적.
  predictions =predictions.to(device= "cpu")
  predictions = predictions.numpy() # 넘파이 배열로 변경.
  actual = actual.to(device= "cpu")
  actual = actual.numpy() # 넘파이 배열로 변경.
  rmse = np.sqrt(mean_squared_error(predictions, actual)) # sklearn을 이용해 RMSE를 계산.

  return rmse,actual,predictions



test_rmse,actual,pred = evaluation(testloader)
test_rmse





