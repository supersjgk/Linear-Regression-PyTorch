import torch
import torch.nn as nn
import numpy  as np
from sklearn import datasets
import matplotlib.pyplot as plt

#data
x_numpy,y_numpy=datasets.make_regression(n_samples=100,n_features=1,noise=15,random_state=1)
x=torch.from_numpy(x_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))
y=y.view(y.shape[0],1)
n_samples,n_features=x.shape

#learning rate
lrate=0.005

#model
model=nn.Linear(n_features,n_features)

#loss
loss=nn.MSELoss()

#optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=lrate)

#training
epochs=1000

for epoch in range(epochs):
	#forward pass
	y_pred=model(x)
	#loss
	l=loss(y,y_pred)
	#backward pass-calculate gradients
	l.backward()
	#update weights
	optimizer.step()
	#empty gradients
	optimizer.zero_grad()
	
	if (epoch+1)%100==0:
		print(f'epoch={epoch+1} loss={l:.4f} ')

#plot
pred=model(x).detach().numpy()
plt.plot(x_numpy,y_numpy, 'ro')
plt.plot(x_numpy,pred,'b')
plt.show()