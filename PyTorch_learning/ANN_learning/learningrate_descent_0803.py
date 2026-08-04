import torch
from torch import optim
import matplotlib.pyplot as plt

def dm01():
    #1.
    lr, epochs, iteration = 0.1, 200, 10
    #2.dataset
    #true data
    y_true = torch.tensor([0])
    x = torch.tensor([1.0], dtype=torch.float32)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    #3.optimizer
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    #4.learning rate descent
    #4.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    #5.
    lr_list, epoch_list = [], []
    #6.epoch loop
    for epoch in range(epochs):
        #7.get epoch and learning rate
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())
        #8.interation loop
        for batch in range(iteration):
            #9.pred
            y_pred = w * x
            #10.loss
            criterion = ((y_pred - y_true) ** 2)
            #11.zero grad + backward + step
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()
        #12.update learning rate
        scheduler.step()
    #13.print
    print(f'lr_list:{lr_list}')
    
    #14.plot
    plt.plot(epoch_list, lr_list)
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.legend()
    plt.title('learning rate descent')
    plt.show()
    
def dm02():
    #1.
    lr, epochs, iteration = 0.1, 200, 10
    #2.dataset
    #true data
    y_true = torch.tensor([0])
    x = torch.tensor([1.0], dtype=torch.float32)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    #3.optimizer
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    #4.learning rate descent
    #4.1
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    #4.2
    milestones = [50, 125, 160]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    #5.
    lr_list, epoch_list = [], []
    #6.epoch loop
    for epoch in range(epochs):
        #7.get epoch and learning rate
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())
        #8.interation loop
        for batch in range(iteration):
            #9.pred
            y_pred = w * x
            #10.loss
            criterion = ((y_pred - y_true) ** 2)
            #11.zero grad + backward + step
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()
        #12.update learning rate
        scheduler.step()
    #13.print
    print(f'lr_list:{lr_list}')
    
    #14.plot
    plt.plot(epoch_list, lr_list)
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.legend()
    plt.title('learning rate descent')
    plt.show()

def dm03():
    #1.
    lr, epochs, iteration = 0.1, 200, 10
    #2.dataset
    #true data
    y_true = torch.tensor([0])
    x = torch.tensor([1.0], dtype=torch.float32)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    #3.optimizer
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    #4.learning rate descent
    #4.1
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    #4.2
    #milestones = [50, 125, 160]
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    #4.3
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    #5.
    lr_list, epoch_list = [], []
    #6.epoch loop
    for epoch in range(epochs):
        #7.get epoch and learning rate
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())
        #8.interation loop
        for batch in range(iteration):
            #9.pred
            y_pred = w * x
            #10.loss
            criterion = ((y_pred - y_true) ** 2)
            #11.zero grad + backward + step
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()
        #12.update learning rate
        scheduler.step()
    #13.print
    print(f'lr_list:{lr_list}')
    
    #14.plot
    plt.plot(epoch_list, lr_list)
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.legend()
    plt.title('learning rate descent')
    plt.show()
  
    
if __name__ == '__main__':
    #dm01()
    #dm02()
    dm03()