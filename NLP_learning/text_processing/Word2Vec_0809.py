'''
    COBW(Continuous bag of words)
    两侧预测中间
    
    Skip-gram
    中间预测两边
'''
 
import fasttext
 
def dm01_train_save():
    my_model = fasttext.train_unsupervised()
     