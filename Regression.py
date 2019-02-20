import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import LinearModel
import NonLinearModel
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filename):
    file = open(filename, "r")
    #lines = file.readlines()

    dataset = []

    for line in file:
        temp = line.rstrip('\n').split(' ')
        curr = []
        for j in temp:
            curr.append(float(j))

        dataset.append(curr)

    file.close()
    return dataset

def split(data):
    x = [row[:-1] for row in data]
    y = [row[-1] for row in data]

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

# Hyper Parameters
input_size = 10
output_size = 1
num_epochs = 500
learning_rate = 0.001

#w = Variable( torch.randn(25), requires_grad = True )


if __name__ == '__main__':
    file = "Train.txt"
    Data = load_data(file)
    Xtrain, Ytrain = split(Data)

    file = "Test.txt"
    Data = load_data(file)
    Xtest, Ytest  = split(Data)

    model = NonLinearModel.Network(input_size, output_size)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Convert numpy array to torch Variable
        inputs = Variable(torch.from_numpy(Xtrain))
        targets = Variable(torch.from_numpy(Ytrain))

        # Forward + Backward + Optimize
        optimizer.zero_grad()  
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    predicted = model( Variable( torch.from_numpy(Xtest) ) )
    Ypredicted = predicted.data.numpy()
    print("Mean squared error: %.2f" % mean_squared_error(Ytest, Ypredicted ))
    print('Variance score: %.2f' % r2_score(Ytest, Ypredicted))