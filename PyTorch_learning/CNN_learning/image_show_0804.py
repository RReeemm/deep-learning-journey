import numpy as np
import matplotlib.pyplot as plt
import torch

def dm01():
    img1 = np.zeros((200,200,3))
    print(f'img1: {img1}')
    plt.imshow(img1)
    plt.axis('off')  # Hide axis
    plt.show()
    
def dm02():
    img2 = torch.full((200,200,3), fill_value= 255)
    plt.imshow(img2)
    plt.axis('off')  # Hide axis
    
    plt.show()
    
def dm03():
    img1 = plt.imread('data/1.jpg')
    plt.imsave('data/1_copy.jpg', img1)
    plt.imshow(img1)
    plt.axis('off')  # Hide axis
    plt.show()

if __name__ == "__main__":
    dm01()
    dm02()