import torch

w = torch.tensor(data=10, requires_grad=True, dtype=torch.float)
loss = w ** 2 + 20
print(f'w: {w}, loss: {loss}')

for i in range(1,101):
    loss = w ** 2 + 20
    if w.grad is not None:
        w.grad.zero_()
    loss.sum().backward()
    w.data -= 0.01 * w.grad
    print(f'epoch: {i}, w: {w:.2f}, loss: {loss:.2f}')
print(f'final w: {w:.2f}, final loss: {loss:.2f}')