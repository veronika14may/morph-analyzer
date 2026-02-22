import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

class MorphologyDataset(Dataset):
    def __init__(self, sentences, pos_tags, vectorizer):
        self.sentences = sentences
        self.pos_tags = pos_tags
        self.vectorizer = vectorizer
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        pos_tags = self.pos_tags[idx]
        
        sentence_vec = self.vectorizer.vectorize_sentence(sentence)
        pos_vec = self.vectorizer.vectorize_pos(pos_tags)
        
        return sentence_vec, pos_vec, len(sentence)

def collate_fn(batch):
    sentences, pos_tags, lengths = zip(*batch)
    
    lengths = torch.LongTensor(lengths)
    sorted_idx = torch.argsort(lengths, descending=True)
    
    sentences = [sentences[i] for i in sorted_idx]
    pos_tags = [pos_tags[i] for i in sorted_idx]
    lengths = lengths[sorted_idx]
    
    max_len = lengths[0].item()
    padded_sentences = torch.zeros(len(sentences), max_len, dtype=torch.long)
    padded_pos = torch.zeros(len(sentences), max_len, dtype=torch.long)
    
    for i, (sent, pos) in enumerate(zip(sentences, pos_tags)):
        padded_sentences[i, :len(sent)] = sent
        padded_pos[i, :len(pos)] = pos
    
    return padded_sentences, padded_pos, lengths

class MorphologyTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        for sentences, pos_tags, lengths in tqdm(train_loader, desc="Training"):
            sentences = sentences.to(self.device)
            pos_tags = pos_tags.to(self.device)
            lengths = lengths.to(self.device)
            
            optimizer.zero_grad()
            
            pos_scores = self.model(sentences, lengths)
            
            pos_scores = pos_scores.view(-1, pos_scores.shape[-1])
            pos_tags = pos_tags.view(-1)
            
            loss = criterion(pos_scores, pos_tags)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(pos_scores, dim=1)
            
            mask = pos_tags != 0
            total_correct += ((predictions == pos_tags) & mask).sum().item()
            total_tokens += mask.sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for sentences, pos_tags, lengths in tqdm(val_loader, desc="Validation"):
                sentences = sentences.to(self.device)
                pos_tags = pos_tags.to(self.device)
                lengths = lengths.to(self.device)
                
                pos_scores = self.model(sentences, lengths)
                
                pos_scores_flat = pos_scores.view(-1, pos_scores.shape[-1])
                pos_tags_flat = pos_tags.view(-1)
                
                loss = criterion(pos_scores_flat, pos_tags_flat)
                
                total_loss += loss.item()
                predictions = torch.argmax(pos_scores_flat, dim=1)
                
                mask = pos_tags_flat != 0
                total_correct += ((predictions == pos_tags_flat) & mask).sum().item()
                total_tokens += mask.sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs, lr=0.001):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.NLLLoss(ignore_index=0)  # Игнорируем PAD
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=1
        )
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f"\nЭпоха {epoch + 1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            scheduler.step(val_acc)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Текущий Learning Rate: {current_lr}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pt')
                print(f"Сохранена лучшая модель с точностью {val_acc:.4f}")
        
        print(f"\nЛучшая точность на валидации: {best_val_acc:.4f}")