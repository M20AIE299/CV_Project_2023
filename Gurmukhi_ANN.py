# -*- coding: utf-8 -*-
"""


@author: saumitra tarey
"""


import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

b = np.load('gurmukhi2.npy')
v = np.load('gurmukhi_val.npy')

print(b.shape)
print(v.shape)

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])


temp = torch.from_numpy(b)
image_2_npArray_2_tensor = temp.float()
#print(image_2_npArray_2_tensor)
print('the shape of numpy array transformed into tensor: {}'.format(np.shape(image_2_npArray_2_tensor)))
#print('transformed numpy array: {}'.format(image_2_npArray_2_tensor))

image_2_npArray_2_tensor = torch.utils.data.DataLoader(image_2_npArray_2_tensor, batch_size=10, shuffle=False)

temp2 = torch.from_numpy(v)
image_val_2_npArray_2_tensor = temp2.float()
#print(image_val_2_npArray_2_tensor)
print('the shape of numpy val array transformed into tensor: {}'.format(np.shape(image_val_2_npArray_2_tensor)))
#print('transformed numpy val array: {}'.format(image_val_2_npArray_2_tensor))

image_val_2_npArray_2_tensor = torch.utils.data.DataLoader(image_val_2_npArray_2_tensor, batch_size=10, shuffle=False)


# targets
targets = []
for x in range(10):
    for y in range(100):
        targets.append(x)
        
print(len(targets))
#print(targets)

targets = np.array(targets)
targets_2_npArray_2_tensor = torch.from_numpy(targets)
print(targets_2_npArray_2_tensor.shape)

targets_2_npArray_2_tensor = torch.utils.data.DataLoader(targets_2_npArray_2_tensor, batch_size=10, shuffle=False)
#labels = targets_2_npArray_2_tensor

# targets for val

targets_val = []
for l in range(10):
    for m in range(20):
        targets_val.append(l)
        
print(len(targets_val))
#print(targets_val)

targets_val = np.array(targets_val)
targets_val_2_npArray_2_tensor = torch.from_numpy(targets_val)
print(targets_val_2_npArray_2_tensor.shape)

targets_val_2_npArray_2_tensor = torch.utils.data.DataLoader(targets_val_2_npArray_2_tensor, batch_size=10, shuffle=False)


input_size = 1024
hidden_sizes = [512, 256]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
 
for e in range(epochs):
    running_loss = 0
    for images, labels in zip(image_2_npArray_2_tensor, targets_2_npArray_2_tensor):
        # Flatten MNIST images into a 1024 long vector
        images = images.view(images.shape[0], -1)
            
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
                
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()

    else:
      print("Epoch {} - Training loss: {}".format(e, running_loss/len(image_2_npArray_2_tensor)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)


correct_count, all_count = 0, 0
for images2,labels2 in zip(image_val_2_npArray_2_tensor, targets_val_2_npArray_2_tensor):
  for i in range(len(labels2)):
    img = images2[i].view(1, 1024)
    #img = images2[i].view(images2[i].shape[0], -1)
    with torch.no_grad():
        logps = model(img)

    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels2.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("number of correctly classified images =", correct_count)
print("\nModel Accuracy =", (correct_count/all_count))