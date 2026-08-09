import jieba

#1.精确模式，文本分析
def dm01():
    content = '春江潮水连海平'
    result1 = jieba.cut(content, cut_all=False)
    for item in result1:
        print(item)
    list = jieba.lcut(content, cut_all = False)
    
#2.全模式，关键词提取
def dm02():
    content = '春江潮水连海平'
    result1 = jieba.cut(content, cut_all=True)
    for item in result1:
       print(item)
    list = jieba.lcut(content, cut_all = True)
    
#搜索引擎模式，适用于搜索引擎分析，文本匹配
def dm03():
    content = '春江潮水连海平'
    result1 = jieba.cut_for_search(content)
    for item in result1:
       print(item)
    list = jieba.lcut_for_search(content)
    
#4.繁体字
def dm04():
    content = '春江潮水连海平'
    result1 = jieba.cut(content)
    for item in result1:
       print(item)
    list = jieba.lcut(content)
    
#5.演示自定义词典
#格式：词语 词频（可选）词性（可选）
def dm05():
    content = '庄方宜和管理员是武陵管代'
    list1 = jieba.lcut(content)
    print(f'{list1}')
    jieba.load_userdict('deep-learning-journey/NLP_learning/data/userdict.txt')
    list2 = jieba.lcut(content)
    print(f'{list2}')
    
if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    dm05()