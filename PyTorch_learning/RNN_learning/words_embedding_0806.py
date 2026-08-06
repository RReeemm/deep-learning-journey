'''
    Recurrent neural network

'''

import torch
import jieba
import torch.nn as nn

def dm01():
    test = '北京东奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'
    words = jieba.lcut(test)
    print(f'分词结果:{words}')
    
    embed = nn.Embedding(len(words),4)
    for i, word in enumerate(words):
        #print(i, word)
        word_vector = embed(torch.tensor(i))
        print(f'词:{word},\t\t词向量:{word_vector}')
        


if __name__ == '__main__':
    dm01()
