import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

class MorphologyLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 num_pos_tags, dropout = 0.5):
        super(MorphologyLSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bi-LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                           num_layers=2, bidirectional=True, 
                           dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.hidden2pos = nn.Linear(hidden_dim, num_pos_tags)
        
    def forward(self, sentences, lengths):
        embeds = self.word_embeddings(sentences)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.dropout(lstm_out)
        
        pos_space = self.hidden2pos(lstm_out)
        pos_scores = torch.log_softmax(pos_space, dim=2)
        
        return pos_scores

class FeatureVectorizer:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.pos2idx = {}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.idx2pos = {}
        self.feature_dim = 0
        
    def fit(self, sentences, pos_tags):
        """Построение словарей"""
        word_idx = 2
        for sentence in sentences:
            for word in sentence:
                norm_word = word.lower().replace('ё', 'е')
                if norm_word not in self.word2idx:
                    self.word2idx[norm_word] = word_idx
                    self.idx2word[word_idx] = norm_word
                    word_idx += 1
        
        pos_idx = 0
        all_pos = set()
        for tags in pos_tags:
            all_pos.update(tags)
        
        for pos in sorted(all_pos):
            self.pos2idx[pos] = pos_idx
            self.idx2pos[pos_idx] = pos
            pos_idx += 1
            
        print(f"Размер словаря: {len(self.word2idx)}")
        print(f"Количество POS тегов: {len(self.pos2idx)}")
        
    def vectorize_sentence(self, sentence):
        indices = []
        for word in sentence:
            norm_word = word.lower().replace('ё', 'е')
            indices.append(self.word2idx.get(norm_word, 1))  # 1 = UNK
        return torch.LongTensor(indices)
    
    def vectorize_pos(self, pos_tags):
        indices = [self.pos2idx[pos] for pos in pos_tags]
        return torch.LongTensor(indices)
    
    def save(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'pos2idx': self.pos2idx,
                'idx2word': self.idx2word,
                'idx2pos': self.idx2pos
            }, f)
    
    def load(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.pos2idx = data['pos2idx']
            self.idx2word = data['idx2word']
            self.idx2pos = data['idx2pos']