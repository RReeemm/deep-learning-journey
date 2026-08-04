import torch
import torch.nn as nn
import torch.optim as optim

#1.momentum
'''
    s_t = s_t-1 * momentum + (1 - momentum) * g_t
    w_t = w_t-1 - lr * s_t
    
    s_t: velocity
    g_t: gradient
    w_t: weight
'''

def dm01_momentum():
    #1.
    w = torch.tensor([1.0], requires_grad=True,dtype=torch.float32)
    #2.optimizer(SGD+momentum)
    optimizer = optim.SGD([w], lr=0.1, momentum=0.9)
    #3.compute gradient and update weigh
    for i in range(10):
        optimizer.zero_grad()
        criterion = ((w ** 2) / 2)
        criterion.backward()
        optimizer.step()
        print(f'epoch:{i + 1}, w:{w.item()}')

#2.AdaGrad
'''
    lr = lr / (sqrt(s_t) + eps)
    s_t = s_t-1 + g_t^2
    w_t = w_t-1 - lr * g_t / (sqrt(s_t) + eps)
    
    s_t: sum of squared gradients
    g_t: gradient
'''
def dm02_adagrad():
    #1.
    w = torch.tensor([1.0], requires_grad=True,dtype=torch.float32)
    #2.optimizer(SGD+AdaGrad)
    optimizer = optim.Adagrad([w], lr=0.1, eps=1e-8)
    #3.compute gradient and update weight
    for i in range(10):
        optimizer.zero_grad()
        criterion = ((w ** 2) / 2)
        criterion.sum().backward()
        optimizer.step()
        print(f'epoch:{i + 1}, w:{w.item()}')

#3.RMSProp
'''
    lr = lr / (sqrt(s_t) + eps)
    s_t = decay * s_t-1 + (1 - decay) * g_t^2
    w_t = w_t-1 - lr * g_t / (sqrt(s_t) + eps)
    
    s_t: sum of squared gradients
    g_t: gradient
'''
def dm03_rmsprop():
    #1.
    w = torch.tensor([1.0], requires_grad=True,dtype=torch.float32)
    #2.optimizer(SGD+RMSProp)
    optimizer = optim.RMSprop([w], lr=0.1, alpha=0.9, eps=1e-8)
    #3.compute gradient and update weight
    for i in range(10):
        optimizer.zero_grad()
        criterion = ((w ** 2) / 2)
        criterion.sum().backward()
        optimizer.step()
        print(f'epoch:{i + 1}, w:{w.item()}')
        
#4.Adam
'''
    lr = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
    m_t = beta1 * m_t-1 + (1 - beta1) * g_t
    s_t = beta2 * s_t-1 + (1 - beta2) * g_t^2
    m_t^ = m_t / (1 - beta1^t)
    s_t^ = s_t / (1 - beta2^t)
    w_t = w_t-1 - lr * m_t^ / (sqrt(s_t^) + eps)
    
    m_t: first moment estimate
    s_t: second moment estimate
    g_t: gradient
'''
def dm04_adam():
    #1.
    w = torch.tensor([1.0], requires_grad=True,dtype=torch.float32)
    #2.optimizer(SGD+Adam)
    optimizer = optim.Adam([w], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    #3.compute gradient and update weight
    for i in range(10):
        optimizer.zero_grad()
        criterion = ((w ** 2) / 2)
        criterion.sum().backward()
        optimizer.step()
        print(f'epoch:{i + 1}, w:{w.item()}')
        
if __name__ == '__main__':
    print('1.momentum')
    dm01_momentum()

    # print('2.AdaGrad')
    # dm02_adagrad()
    # print('3.RMSProp')
    # dm03_rmsprop()
    # print('4.Adam')
    # dm04_adam()