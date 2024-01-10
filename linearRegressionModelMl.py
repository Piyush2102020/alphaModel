#import libraries torch ,nn and matplot for analysis
import torch
from torch import nn
import matplotlib.pyplot as plt

#create a random weight and bias for a dataset
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.02
epochs=300 # hyper parameter number of training loops and learning rate change these to improve accuracy
lr=0.01
X = torch.arange(start, end, step).unsqueeze(dim=1) #here we are creating a range of x value at a step size of 0.02
Y = weight * X + bias #the basic linear regression formula to create the Y for a train and test data set
print(len(X), len(Y))


#Model class import nn.Modeule to get all the predefined functions
class LinearRegression(nn.Module):
    #init function to make prediction
    def __init__(self):
        super().__init__()
        #the below will create a two params that are weights and bias which will be adjusted for predicting the further value based on the input
        self.weigths=nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
        self.bias=nn.Parameter(torch.rand(1,requires_grad=True,dtype=torch.float)) #the dtype is float 
        
        
        #this function will return the ouput predictions that means it will multiply the wights with the inputs and will add a bias to it
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.weigths *x +self.bias
        
        
        
        

#here we are splitting the data into test and train split
train_split = int(0.8 * len(X)) # get the 80% of the data
x_train, y_train = X[:train_split], Y[:train_split] #this will return two dataset the x train and the y train in the ratio of 80 and 80
x_test, y_test = X[train_split:], Y[train_split:] # this will return the test dataset for visualizing
print(len(x_train), len(y_train), len(x_test), len(y_test))


#matplot function to visualize the data
def plot_predict(train_data=x_train, train_label=y_train, test_data=x_test, test_label=y_test, prediction=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_label, c="b", s=4, label="Training Data")
    plt.scatter(test_data, test_label, c="g", s=4, label="Testing Data")
    if prediction is not None:
        plt.scatter(test_data, prediction, c="r", label="Prediction")

    plt.legend(prop={"size": 14})



#add a manual seed 
torch.manual_seed(42)
#create an instance of the model class
model=LinearRegression()
loss_fn=nn.L1Loss() #create an instance of the loss function we are using L1Loss this will return the MAE the difference between the required and the predicted output
optimizer=torch.optim.SGD(params=model.parameters(),lr=lr) #heres the SGD optimizer that will optimize the value are a learning rate mention above
print(list(model.parameters())) #printing model parameters for the first time this will return weights and bias
print(model.state_dict())


    
    
#training loop this will train the model
for epoch in range(epochs):
    #set the model to train with the .train() function
    model.train()
    #make predictions with the model based on the input of train
    y_preds=model(x_train)
    loss=loss_fn(y_preds,y_train) #check on the loss and prediction and calculate the loss
    optimizer.zero_grad() #set the gradients to zero
    print(loss) #print the loss
    loss.backward() #do  back propagation
    optimizer.step() #step the optimizer
    
    

#after training set the model to evaluation mode
model.eval()
#inference mode wil turn off all the grads and other things that can cause an issue
with torch.inference_mode():
    #make predictions with the trained model based on the test value of x
    y_preds=model(x_test) 
    loss=loss_fn(y_preds,y_test)
    print(f"test loss : {loss}")
    #print the prediction and pass them to the predict functions
    print(y_preds)
    plot_predict(prediction=y_preds)
    
    
#plot the model
plt.show()