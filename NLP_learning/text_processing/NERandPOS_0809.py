''' 
    文本处理基本方法：
    分词


    Named Entity Recognition
    命名实体识别(NER)
    识别出可能存在的命名实体
    命名实体：人名，地名，机构等

    Part-Of-Speech tagging
    词性标注(POS)
    
    名词，动词，形容词……
    
'''

import jieba.posseg as pseg
content = '我爱庄方宜和佩丽卡'
result = pseg.lcut(content)
print(f'{result}')
for word, flag in result:
    print(f'{word},{flag}')
