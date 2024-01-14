import torch
from torch import nn
from helper_functions import plot_predictions

data=[]
out=512
lr=0.0008
epochs=200

vocab="abcdefghijklmnopqrstuvwxyz1234567890"
def encode(x):
    return vocab.index(x)

torch.manual_seed(42)

def decode(x):
    if x<int(len(vocab)):
        return vocab[x]
    else:
        return "not in range"
    

for a in vocab:
    data.append(encode(a))
    
    
    
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(in_features=1,out_features=out)
        self.layer2=nn.Linear(in_features=out,out_features=out)
        self.layer3=nn.Linear(in_features=out,out_features=out)
        self.layer4=nn.Linear(in_features=out,out_features=out)
        self.layer5=nn.Linear(in_features=out,out_features=out)
        self.layer6=nn.Linear(in_features=out,out_features=1)
        
    def forward(self,x):
        return self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))
    
    


X=torch.tensor(data).unsqueeze(1)
y=[]
for a in range(len(X)-1):
    y.append(a+1)
    
Y=torch.tensor(y).unsqueeze(1).type(torch.float)
print(X.ndim,X.shape)
split=int(0.8*len(X))
x_train,y_train=X[:split],Y[:split]
x_test,y_test=X[split:],Y[split:]
x_test=x_test[:len(x_test)-1]
x_test=x_test.type(torch.float)

x_train=x_train.type(torch.float)
model=LinearRegressionModel()
loss_fn=nn.MSELoss()
optimizer=torch.optim.Adam(params=model.parameters(),lr=lr)


#train loop
model.train()
for epoch in range(epochs):
    predictions=model(x_train)
    loss=loss_fn(predictions,y_train)
    optimizer.zero_grad()
    if epoch %10==0:
        print(loss)
    loss.backward()
    optimizer.step()
    

with torch.inference_mode():
    model.eval()
    pred=torch.round(model(x_test)).type(torch.int)

    
req_val=[]
for a in x_test:
    req_val.append(int(torch.round(a[0])))
    
pred_val=[]
for a in pred:
    pred_val.append(decode(a[0]))

    
for a in range(len(req_val)):
    print(f"For the Required value {decode(req_val[a])} the prediction is {pred_val[a]}")
    

    
while True:
    user_input = input("Enter the letter ")
    dat = [encode(user_input)]

    pred = model(torch.tensor(dat).type(torch.float).unsqueeze(dim=1))
    predicted_index = int(torch.round(pred).item())
    predicted_word = decode(predicted_index)
    print(f"The next word to your letter {user_input} is {predicted_word}")

    
    
