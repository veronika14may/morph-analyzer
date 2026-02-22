import torch
import re
from typing import List, Tuple

class MorphologyPredictor:
    def __init__(self, model, vectorizer, preprocessor, device='cpu'):
        self.model = model
        self.vectorizer = vectorizer
        self.preprocessor = preprocessor
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def normalize_word(self, word):
        return word.lower().replace('ё', 'е')
    
    def tokenize_sentence(self, sentence):
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        return [w for w in sentence.split() if w]
    
    def predict_sentence(self, sentence):
        tokens = self.tokenize_sentence(sentence)
        if not tokens:
            return []
        
        sentence_vec = self.vectorizer.vectorize_sentence(tokens).unsqueeze(0)
        length = torch.LongTensor([len(tokens)])
        
        sentence_vec = sentence_vec.to(self.device)
        length = length.to(self.device)
        
        with torch.no_grad():
            pos_scores = self.model(sentence_vec, length)
            pos_predictions = torch.argmax(pos_scores, dim=2).squeeze(0)
        
        results = []
        for i, token in enumerate(tokens):
            pos_idx = pos_predictions[i].item()
            pos_tag = self.vectorizer.idx2pos[pos_idx]
            norm_token = self.normalize_word(token)
            
            if re.match(r'^[.,!?]$', token):
                results.append((token, token, 'PUNCT'))
                continue
            
            lemma = norm_token # Значение по умолчанию
            
            if norm_token in self.preprocessor.word_lemma_pos:
                # Достаем дефолтную лемму из словаря
                default_lemma, dict_pos = self.preprocessor.word_lemma_pos[norm_token]
                
                # Пробуем найти лемму, которая идеально совпадает с предсказанным тегом (для омонимов)
                candidates = self.preprocessor.get_candidates(token)
                lemma_found = False
                
                for cand_lemma, cand_pos in candidates:
                    if cand_pos == pos_tag:
                        lemma = cand_lemma
                        lemma_found = True
                        break
                
                # ЕСЛИ идеального совпадения нет (нейросеть ошиблась тегом или в словаре другой тег),
                # мы все равно берем нормальную начальную форму (default_lemma), а НЕ исходное слово!
                if not lemma_found:
                    lemma = default_lemma
            
            results.append((token, lemma, pos_tag))
        
        return results
    
    def format_output(self, results):
        formatted = []
        for token, lemma, pos in results:
            if pos == 'PUNCT':
                formatted.append(token)
            else:
                formatted.append(f"{token}{{{lemma}={pos}}}")
        return ' '.join(formatted)
    
    def predict_file(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    results = self.predict_sentence(sentence)
                    output = self.format_output(results)
                    f.write(output + '\n')