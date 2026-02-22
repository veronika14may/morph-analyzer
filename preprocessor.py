import json
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import List, Tuple

class DataPreprocessor:
    def __init__(self):
        self.pos_tags = {
            'NOUN': 'S', 'ADJF': 'A', 'ADJS': 'A', 'COMP': 'A',
            'VERB': 'V', 'INFN': 'V', 'PRTF': 'V', 'PRTS': 'V', 'GRND': 'V',
            'NUMR': 'NUM', 'ADVB': 'ADV', 'NPRO': 'NI', 'PRED': 'ADV',
            'PREP': 'PR', 'CONJ': 'CONJ', 'PRCL': 'PART', 'INTJ': 'INTJ',
        }
        
        self.word_forms = defaultdict(lambda: defaultdict(set))
        self.lemma_pos = {}
        self.word_lemma_pos = {}
        
    def normalize_word(self, word):
        return word.lower().replace('ё', 'е')
    
    def parse_opencorpora_xml(self, zip_file_path, max_words = None):
        print(f"Загрузка словаря из {zip_file_path}")
        try:
            word_count = 0
            
            with zipfile.ZipFile(zip_file_path, 'r') as z:
                xml_filename = [name for name in z.namelist() if name.endswith('.xml')][0]
                
                with z.open(xml_filename) as xml_file:
                    context = ET.iterparse(xml_file, events=('end',))
                    
                    for event, elem in context:
                        if elem.tag == 'lemma':
                            if max_words and word_count >= max_words:
                                break
                                
                            lemma_text = None
                            pos_tag = None
                            
                            # Ищем тег <l> (базовая форма слова)
                            l_node = elem.find('l')
                            if l_node is not None:
                                lemma_text = l_node.get('t') # Достаем само слово из атрибута 't'
                                # Собираем все граммемы из тегов <g>
                                grammemes = [g.get('v') for g in l_node.findall('g')]
                                if grammemes:
                                    pos_tag = self.pos_tags.get(grammemes[0], 'S') # 'S' по умолчанию
                            
                            if lemma_text and pos_tag:
                                norm_lemma = self.normalize_word(lemma_text)
                                self.lemma_pos[norm_lemma] = pos_tag
                                self.word_forms[norm_lemma][norm_lemma].add(pos_tag)
                                
                                # Сохраняем только первое значение для избежания затирания частых слов редкими омонимами
                                if norm_lemma not in self.word_lemma_pos:
                                    self.word_lemma_pos[norm_lemma] = (norm_lemma, pos_tag)
                                
                                # Ищем все формы этого слова
                                for form in elem.findall('f'):
                                    word_form = form.get('t')
                                    if word_form:
                                        norm_form = self.normalize_word(word_form)
                                        self.word_forms[norm_form][norm_lemma].add(pos_tag)
                                        
                                        if norm_form not in self.word_lemma_pos:
                                            self.word_lemma_pos[norm_form] = (norm_lemma, pos_tag)
                                
                                word_count += 1
                            
                            elem.clear()
                            
            print(f"Успешно загружено {word_count} лемм из OpenCorpora.")
            
        except Exception as e:
            print(f"Ошибка: {e}")
            raise e
            
    def get_candidates(self, word):
        norm_word = self.normalize_word(word)
        candidates = []
        if norm_word in self.word_forms:
            for lemma, pos_set in self.word_forms[norm_word].items():
                for pos in pos_set:
                    candidates.append((lemma, pos))
        return candidates if candidates else [(norm_word, 'S')] # Существительное по умолчанию
    
    def save_to_json(self, filename):
        data = {
            'word_lemma_pos': self.word_lemma_pos,
            'lemma_pos': self.lemma_pos
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_json(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.word_lemma_pos = {k: tuple(v) for k, v in data['word_lemma_pos'].items()}
        self.lemma_pos = data['lemma_pos']
        
        for word, (lemma, pos) in self.word_lemma_pos.items():
            self.word_forms[word][lemma].add(pos)