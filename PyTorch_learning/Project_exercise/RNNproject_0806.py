import torch
import jieba
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import time

def build_vocab():
    unique_words, all_words = [],[]
    for line in open('D:/GitHub/deep-learning-journey/PyTorch_learning/Project_exercise/data/jaychou_lyrics.txt', encoding='utf-8'):
        words = jieba.lcut(line)
        all_words.append(words)
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    word_count = len(unique_words)      #number of unique words
    word_to_index = {word:i for i, word in enumerate(unique_words)} #index of unique_words
    corpus_idx = []     #index of words in txt
    for words in all_words:
        tmp = []
        for word in words:
            tmp.append(word_to_index[word])
        tmp.append(word_to_index[' '])
        corpus_idx.extend(tmp)
    return unique_words, word_to_index, word_count, corpus_idx

class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_idx, num_chars):
        self.corpus_idx = corpus_idx    #index of words in txt
        self.num_chars = num_chars      #the number of words in every lyriccs
        self.word_count = len(self.corpus_idx)      #number of words in txt
        self.number = self.word_count // self.num_chars #number of sentense in txt
        
    def __len__(self):
        return self.number
    def __getitem__(self, idx):
        start = min(max(idx, 0), self.word_count - self.num_chars - 1)
        end = start + self.num_chars
        x = self.corpus_idx[start:end]
        y = self.corpus_idx[start + 1:end + 1]
        return torch.tensor(x), torch.tensor(y)
    
class TextGenerator(nn.Module):
    def __init__(self, unique_word_count):
        super().__init__()
        self.ebd = nn.Embedding(unique_word_count, 128) #embedding words
        self.rnn = nn.RNN(128, 256, 1)
        self.out = nn.Linear(256, unique_word_count)
        
    def forward(self, inputs, hidden):
        batch_size = inputs.size(0)
        if hidden is None or hidden.size(1) != batch_size:
            hidden = torch.zeros(1, batch_size, 256, device=inputs.device)
        embd = self.ebd(inputs)
        output, hidden = self.rnn(embd.transpose(0, 1), hidden)
        output = self.out(output.reshape(-1, output.shape[-1]))
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1,batch_size, 256)
        
def train():
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    lyrics = LyricsDataset(corpus_idx, 32)
    model = TextGenerator(unique_word_count)
    lyrics_dataloader = DataLoader(lyrics, batch_size=5, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    epochs = 10
    for epoch in range(epochs):
        start, iter_num, total_loss = time.time(), 0, 0.0
        for x, y in lyrics_dataloader:
            hidden = model.init_hidden(5)
            output, hidden = model(x, hidden)
            y = torch.transpose(y, 0, 1).reshape(shape=(-1,))
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iter_num += 1
        print(f'epoch:{epoch + 1},time:{time.time() - start:.2f}s, loss:{total_loss / iter_num:.4f}')
    torch.save(model.state_dict(), 'D:/GitHub/deep-learning-journey/PyTorch_learning/Project_exercise/model/text_generator.pth')
    
def evaluate(start_word, sentence_length):
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    model = TextGenerator(unique_word_count)    
    model.load_state_dict(torch.load('D:/GitHub/deep-learning-journey/PyTorch_learning/Project_exercise/model/text_generator.pth'))
    hidden = model.init_hidden(1)
    word_idx = word_to_index[start_word]
    generate_sentence = [word_idx]
    for i in range(sentence_length):
        output, hidden = model(torch.tensor([[word_idx]]), hidden)
        word_idx = torch.argmax(output)
        generate_sentence.append(word_idx)
    for idx in generate_sentence:
        print(unique_words[idx], end='')
        
if __name__ == '__main__':
    #unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    
    #dataset = LyricsDataset(corpus_idx, 5)

    #train()
    evaluate('好',50)