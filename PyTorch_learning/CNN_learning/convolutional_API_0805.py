'''
convolutional neural network 
    convolutional layer
    pooling layer
    fully connected layer
'''

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def dm01():
    img = plt.imread('D:\GitHub\deep-learning-journey\PyTorch_learning\Project_exercise\data\img.jpg')
    #print(f'img: {img}, shape: {img.shape}')
    # HWC->CHW
    img2 = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    img3 = img2.unsqueeze(0)  # Add batch dimension
    
    #create convolutional layer
    conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0)

    #apply convolutional layer
    conv_img = conv(img3)
    #print(f'output: {conv_img}, shape: {conv_img.shape}')
    
    img4 = conv_img[0].permute(1, 2, 0)
    feature1 = img4[:, :, 0].detach().numpy()   
    feature2 = img4[:, :, 1].detach().numpy()
    
    plt.imshow(feature2)
    plt.axis('off')  # Hide axis
    plt.show()

if __name__ == "__main__":
    dm01()