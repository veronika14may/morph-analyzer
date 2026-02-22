import zipfile
import xml.etree.ElementTree as ET
import os
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional

class OpenCorporaLoader:
    def __init__(self):
        self.pos_mapping = {
            'NOUN': 'S', # Существительное 
            'ADJF': 'A', 'ADJS': 'A', 'COMP': 'A', # Прилагательное
            'VERB': 'V', 'INFN': 'V', 'PRTF': 'V', 'PRTS': 'V', 'GRND': 'V', # Глагол
            'NUMR': 'NUM', # Числительное
            'ADVB': 'ADV', # Наречие
            'NPRO': 'NI', # Местоимение
            'PRED': 'ADV', # Наречие
            'PREP': 'PR', # Предлог
            'CONJ': 'CONJ', # Союз
            'PRCL': 'PART', # Частица
            'INTJ': 'INTJ', # Междометие
        }
    
    def load_and_split(self, zip_file_path: str, max_sentences: int = None, test_size: float = 0.2):
        sentences = []
        pos_tags = []
        sentence_count = 0
        
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as z:
                xml_filenames = [
                    name for name in z.namelist() 
                    if name.endswith('.xml') and '__MACOSX' not in name and not name.split('/')[-1].startswith('.')
                ]
                
                if not xml_filenames:
                    print("Ошибка: В архиве не найден валидный .xml файл.")
                    return None, None
                    
                xml_filename = xml_filenames[0]
                
                with z.open(xml_filename) as xml_file:
                    context = ET.iterparse(xml_file, events=('end',))
                    
                    for event, elem in context:
                        if elem.tag == 'sentence':
                            if max_sentences and sentence_count >= max_sentences:
                                break
                                
                            tokens = []
                            tags = []
                            
                            for token in elem.findall('.//token'):
                                text = token.get('text')
                                pos = 'UNKN' # по умолчанию часть речи "Неизвестна"
                                
                                g = token.find('.//g')
                                if g is not None:
                                    pos_raw = g.get('v')
                                    pos = self.pos_mapping.get(pos_raw, 'S') # по умолчанию для битых, редких, неизвестных, и тд.
                                else:
                                    pos = 'PUNCT'
                                    
                                tokens.append(text)
                                tags.append(pos)
                            
                            if tokens:
                                sentences.append(tokens)
                                pos_tags.append(tags)
                                sentence_count += 1
                                
                                if sentence_count % 10000 == 0:
                                    print(f"Обработано {sentence_count} предложений")
                            
                            elem.clear()
                            
        except Exception as e:
            print(f"Ошибка при загрузке: {e}")
            return None, None
            
        print(f"Всего загружено {len(sentences)} предложений.")
        print("Разделение на обучающую и валидационную выборки")
        
        if len(sentences) < 10:
             print("Ошибка: Загружено слишком мало данных для разделения.")
             return None, None
             
        train_sent, val_sent, train_pos, val_pos = train_test_split(
            sentences, pos_tags, test_size=test_size, random_state=42
        )
        
        return (train_sent, train_pos), (val_sent, val_pos)


def load_opencorpora_data(annot_file: str, dict_file: str, max_sentences: int = 50000):
    from preprocessor import DataPreprocessor
    
    print("\nЗагрузка словаря")
    preprocessor = DataPreprocessor()

    dict_json_path = 'dictionary.json'
    
    if os.path.exists(dict_json_path):
        print(f"Нашли готовый файл {dict_json_path}.")
        try:
            preprocessor.load_from_json(dict_json_path)
            print("Словарь успешно загружен.")
        except Exception as e:
            print(f"Файл сломан ({e}), придется парсить XML заново.")
            preprocessor.parse_opencorpora_xml(dict_file, max_words=None)
    else:
        print("Готового JSON нет. Начинаем парсинг XML.")
        try:
            preprocessor.parse_opencorpora_xml(dict_file, max_words=None)
        except Exception as e:
            print(f"Словарь не загружен: {e}")
            return None, None, preprocessor
        
    print("\nЗагрузка и разбиение размеченного корпуса")
    loader = OpenCorporaLoader()
    
    result = loader.load_and_split(annot_file, max_sentences=max_sentences)
    if result == (None, None):
        return None, None, preprocessor
        
    train_data, val_data = result
        
    print(f"Train: {len(train_data[0])} предложений")
    print(f"Val: {len(val_data[0])} предложений")
    
    return train_data, val_data, preprocessor