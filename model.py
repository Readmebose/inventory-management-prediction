#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 00:34:34 2018

@author: vk001716
"""
import os 
os.chdir("/home/vk001716/Desktop/bose_dell")
import random
import torch
import torch.nn as nn
from  torch.autograd  import Variable
import torch.optim as optim 
import pandas as pd

if __name__ == '__main__':
    class model(nn.Module):
        
        def __init__(self):
            super(model , self ).__init__()
            self.layers = nn.Sequential(
                    nn.Linear(36, 72),
                    nn.ReLU(),
                    nn.Linear(72, 144),
                    nn.ReLU(),
                    nn.Linear(144, 36),
                    nn.ReLU(),
                    nn.Linear(36, 18),
                    nn.ReLU(),
                    nn.Linear(18 , 9),
                    nn.ReLU(),
                    nn.Linear(9 , 1),                
                    nn.Sigmoid())
            
        
        def forward(self , x ):
            return self.layers(x)
            
    
    def encode_locations(loc = None):
        encode = [0 for i in range(34)]
        flag = False
        for i in range(len(locations)):
            if locations[i] == loc :
                flag = True
                break
        if flag :
            encode[i] = 1
        return encode
    
    def encode_logistic(x):
        encode = [0 for i in range(10)]
        encode[x] = 1
        return encode
    
    
    locations = pd.read_csv('location.csv' , header = None).values.tolist()[0]
    ratings = pd.read_csv('rating.csv' , header = None).values.tolist()[0]
    dataset = []
    for i in range(10):
        dat = pd.read_csv('l{0}.csv'.format(i+1) , header = None).values.tolist()
        dataset.append(dat)
    models  = []
    optimizers = []
    
    for i in range(10):
        mod = model()
        opt = optim.SGD(mod.parameters(), lr=0.0001)
        optimizers.append(opt)
        models.append(mod)
        
        
    criterion = nn.MSELoss()   
    
    epochs = 2
    for epoch in range(epochs):
        for i in range(8000):
            for j in range(10):
                input = []
                input.append(dataset[j][i][7]/312499) #price
                input.extend(encode_locations(dataset[j][i][11])) #location
                input.append(ratings[j] / 5) #rating
                target= []
                target.append( dataset[j][i][10] /120)#No of days item was in logistic
                input = Variable(torch.FloatTensor(input))
                target = Variable(torch.FloatTensor(target))
                optimizers[j].zero_grad()
                loss = criterion(models[j].forward(input) , target )
                loss.backward()
                optimizers[j].step()
                
                
                if i % 100 == 0 :
                    print( 'Epoch {0} / {1}  Loss of Logistic {2} at row {3} =  {4}'.format(epoch , epochs , j , i ,  loss.data[0]))
def mod(x = None , y = None):
    return random.randint(0 , 33) , random.randint(90 , 120) , random.randint(0 , 9)
    class master_model(nn.Module):
        
        def __init__(self):
            
            super(master_model ,self ).__init__()
            self.layers = nn.Sequential(
                                nn.Linear(2 , 11),
                                nn.ReLU(),
                                nn.Linear(11 , 22),
                                nn.ReLU(),
                                nn.Linear(22 , 44),
                                nn.ReLU(),
                                nn.Linear(44 , 88),
                                nn.ReLU(),
                                nn.Linear(88 , 44),
                                nn.Softmax()
                    )
            
                
        def forward ( self , x ):
            return self.layers( x )
        
        
        
    
    main_model = master_model()
    optimizer_main = optim.SGD(main_model.parameters(), lr=0.0001)
    
    
    # collection main dataset
    
    dataset_main = []
    for i in range(10):
        temp = []
        for j in range(len(dataset[0])):
            temp2 = []
            if dataset[i][j][10] < 40 :
                temp2.append(dataset[i][j][7])
                temp2.append(dataset[i][j][11])
                temp.append(temp2)
        dataset_main.append(temp)
        
        
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for j in range(8000):
            for i in range(10):
                if len(dataset_main[i]) > j:
                    input = []
                    input.append(dataset_main[i][j][0]/312499) #price
                    input.append(ratings[i] / 5)
                    target= []
                    target.extend(encode_locations(dataset_main[i][j][1])) #location
                    target.extend(encode_logistic(i)) 
                    input = Variable(torch.FloatTensor(input))
                    target = Variable(torch.FloatTensor(target))
                    optimizer_main.zero_grad()
                    loss = criterion(main_model.forward(input) , target )
                    loss.backward()
                    optimizer_main.step()
                    if j % 100 == 0 :
                        print( 'Epoch {0} / {1}  Loss of main_model {2} of logistic {3} =  {4}'.format(epoch , epochs , j , i ,  loss.data[0]))
        
    for i in range(10):
        torch.save(models[i].state_dict(), 'saved_state_of_logistic{0}.pkl'.format(i+1))
    torch.save(main_model.state_dict() , 'main_model_saved.pkl')
