import jieba

def dm01():
    vocabs = {'庄方宜','佩丽卡','陈千语','诀'}
    word2index = {vocab: i for i, vocab in enumerate(vocabs)}
    print(word2index)
    for vocab in vocabs:
        zero_list = [0] * len(vocabs)
        idx = word2index[vocab]
        zero_list[idx] = 1
        print(f'{vocab}的one-hot编码为:{zero_list}')
    
if __name__ == '__main__':
    dm01()