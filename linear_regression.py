import torch 
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.], [2.], [3.]])
y_data = torch.tensor([[2.], [4.], [6.]])

class LinearModule(torch.nn.Module):
    def __init__(self):
        super(LinearModule, self).__init__()
        self.linear = torch.nn.Linear(in_features = 1, out_features = 1, bias = True)
        
    def forward(self, x):
        pred_y = self.linear(x)
        return pred_y
    
episode_l = []
w_l = []

def dl():
    model = LinearModule()
    criterion = torch.nn.MSELoss(size_average = False)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    for episode in np.arange(1, 101, 1):
        pred_y = model(x_data)
        loss = criterion(pred_y, y_data)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        episode_l.append(episode)
        w_l.append(model.linear.weight.item())
    
        
def show():
    plt.xlabel('epoch')
    plt.ylabel('w')
    plt.plot(episode_l, w_l)
    plt.show()
    
    
if __name__ == "__main__":
    dl()
    show()