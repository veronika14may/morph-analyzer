import os
import sys
import torch
from torch.utils.data import DataLoader
from preprocessor import DataPreprocessor
from model import MorphologyLSTM, FeatureVectorizer
from trainer import MorphologyDataset, collate_fn, MorphologyTrainer
from predictor import MorphologyPredictor
from data_loader import load_opencorpora_data

def main():
    DATA_DIR = 'lab_1'
    ANNOT_FILE = os.path.join(DATA_DIR, 'annot.opcorpora.xml.zip')
    DICT_FILE = os.path.join(DATA_DIR, 'dict.opcorpora.xml.zip')
    
    if not (os.path.exists(ANNOT_FILE) and os.path.exists(DICT_FILE)):
        print("\nАрхивы OpenCorpora не найдены.")
        sys.exit(1)
        
    print("Найдены архивы OpenCorpora.")
    result = load_opencorpora_data(ANNOT_FILE, DICT_FILE, max_sentences=None)
    
    if not result or result[0] is None:
        print("\nОшибка: Не удалось прочитать данные из корпуса.")
        sys.exit(1)
        
    (train_sentences, train_pos), (val_sentences, val_pos), preprocessor = result
    epochs = 20
    
    # Векторизация
    vectorizer = FeatureVectorizer()
    vectorizer.fit(train_sentences + val_sentences, train_pos + val_pos)
    
    # Создание датасетов
    train_dataset = MorphologyDataset(train_sentences, train_pos, vectorizer)
    val_dataset = MorphologyDataset(val_sentences, val_pos, vectorizer)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Создание модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    model = MorphologyLSTM(
        vocab_size=len(vectorizer.word2idx),
        embedding_dim=100,
        hidden_dim=256,
        num_pos_tags=len(vectorizer.pos2idx),
        dropout=0.5
    )
    
    print(f"Параметров в модели: {sum(p.numel() for p in model.parameters())}")
    
    # Обучение
    # trainer = MorphologyTrainer(model, device=device)
    # trainer.train(train_loader, val_loader, epochs=epochs, lr=0.001)
    
    # Загрузка лучшей модели
    try:
        model.load_state_dict(torch.load('best_model.pt', weights_only=True))
    except FileNotFoundError:
        print("ВНИМАНИЕ: best_model.pt не найдена. Используем веса последней эпохи.")
    
    # Сохранение компонентов
    vectorizer.save('vectorizer.pkl')
    preprocessor.save_to_json('dictionary.json')
    
    # Тестирование
    predictor = MorphologyPredictor(model, vectorizer, preprocessor, device)
    
    test_sentences = [
        "Стала стабильнее экономическая и политическая обстановка, предприятия вывели из тени зарплаты сотрудников.",
        "Все Гришины одноклассники уже побывали за границей, он был чуть ли не единственным, кого не вывозили никуда дальше Красной Пахры.",
        "Мать начала печь пироги, а старая печь вдруг задымилась",
        "Мы стали делать детали из прочной стали",
        "Капля дождя стекла на лобовое стекло автомобиля",
        "Мой краш начал жестко рофлить над этой ситуацией, разрушив весь вайб",
        "Программист обещал быстро пофиксить баг, чтобы не кринжевать перед заказчиком"
    ]
    
    print("\nПримеры предсказаний:\n")
    for sent in test_sentences:
        results = predictor.predict_sentence(sent)
        output = predictor.format_output(results)
        print(f"Input: {sent}")
        print(f"Output: {output}\n")

if __name__ == "__main__":
    main()