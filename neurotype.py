import json
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from sklearn.model_selection import train_test_split
import os

# Путь для сохранения весов модели
BERT_WEIGHTS_PATH = './bert_weights.h5'

# Загрузка JSON данных из файла
with open('questions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Задаем список функций
functions = ["ЧИ", "БИ", "ЧС", "БС", "БЛ", "ЧЛ", "БЭ", "ЧЭ"]

# Задаем список типов и их функции
types_functions = {
    "Дон Кихот": ["ЧИ", "БЛ"],
    "Дюма": ["БС", "БЭ"],
    "Робеспьер": ["БЛ", "ЧИ"],
    "Гюго": ["ЧЭ", "БС"],
    "Жуков": ["ЧС", "БЛ"],
    "Есенин": ["БИ", "ЧЭ"],
    "Гамлет": ["ЧЭ", "БИ"],
    "Максим": ["БЛ", "ЧС"],
    "Гексли": ["ЧИ", "БЭ"],
    "Габен": ["БС", "ЧЛ"],
    "Драйзер": ["БЭ", "ЧС"],
    "Штирлиц": ["ЧЛ", "БС"],
    "Бальзак": ["БИ", "ЧЛ"],
    "Наполеон": ["ЧС", "БЭ"],
    "Достоевский": ["БЭ", "ЧИ"],
    "Джек Лондон": ["ЧЛ", "БИ"]
}

# Настраиваем гиперпараметры
learning_rate = 2e-5
batch_size = 8
epochs = 10
dropout_rate = 0.2
clipnorm = 1.0
early_stopping_patience = 3

# Извлечение вопросов и меток
list_of_questions = [entry['statement'] for entry in data]
labels = np.array([
    [entry['function_correlation'].get(func, 0.0) for func in functions]
    for entry in data
])

# Инициализируем токенизатор и модель BERT для русского языка
print("Загрузка токенизатора и модели BERT для русского языка...")
bert_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
bert_base_model = TFAutoModel.from_pretrained("DeepPavlov/rubert-base-cased", from_pt=True)

# Функция для создания модели
def create_bert_model():
    # Создаем входные слои
    input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name='attention_mask')

    # Получаем выходы BERT
    bert_outputs = bert_base_model(input_ids, attention_mask=attention_mask)
    sequence_output = bert_outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)

    # Используем выход [CLS] токена для регрессии
    cls_token = sequence_output[:, 0, :]  # (batch_size, hidden_size)

    # Добавляем плотные слои для регрессии
    x = tf.keras.layers.Dense(512, activation='relu')(cls_token)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    output = tf.keras.layers.Dense(len(functions), activation='linear')(x)

    # Создаем модель
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    # Создаем оптимизатор с клиппингом градиента
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)

    # Компилируем модель
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    return model

# Создание или загрузка модели
if os.path.exists(BERT_WEIGHTS_PATH):
    use_saved_model = input(f"Найдены сохраненные веса модели BERT в '{BERT_WEIGHTS_PATH}'. Использовать их? (y/n): ")
    if use_saved_model.lower() == 'y':
        print("Создание модели и загрузка сохраненных весов...")
        model = create_bert_model()
        model.load_weights(BERT_WEIGHTS_PATH)
    else:
        print("Создание новой модели и обучение...")
        model = create_bert_model()
else:
    print("Сохраненные веса модели не найдены. Создание и обучение новой модели...")
    model = create_bert_model()

# Функция подготовки данных
def encode_questions(questions, tokenizer, max_length=128):
    return tokenizer(
        questions,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors="np"
    )

# Подготовка данных
encoded_questions = encode_questions(list_of_questions, bert_tokenizer)
input_ids_data = encoded_questions["input_ids"]
attention_masks_data = encoded_questions["attention_mask"]

# Убедимся, что размеры данных согласованы
assert len(input_ids_data) == len(labels), "Количество вопросов не совпадает с количеством меток."

# Разделение данных на тренировочные и валидационные
train_input_ids, val_input_ids, train_attention_masks, val_attention_masks, train_labels, val_labels = train_test_split(
    input_ids_data, attention_masks_data, labels, test_size=0.2, random_state=42
)

# Создаем датасеты
train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"input_ids": train_input_ids, "attention_mask": train_attention_masks},
        train_labels
    )
).shuffle(len(train_input_ids)).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"input_ids": val_input_ids, "attention_mask": val_attention_masks},
        val_labels
    )
).batch(batch_size)

# Обучение модели, если она не была загружена
if not os.path.exists(BERT_WEIGHTS_PATH) or (os.path.exists(BERT_WEIGHTS_PATH) and use_saved_model.lower() != 'y'):
    # Добавляем раннюю остановку
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True
    )

    # Обучаем модель
    print("Обучение модели...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # Сохраняем веса модели
    print(f"Сохранение весов модели в '{BERT_WEIGHTS_PATH}'...")
    model.save_weights(BERT_WEIGHTS_PATH)

# Функция для предсказания корреляций и вывода таблицы
def predict_correlations(question):
    encoded_question = encode_questions([question], bert_tokenizer)
    predictions = model.predict({
        'input_ids': encoded_question['input_ids'],
        'attention_mask': encoded_question['attention_mask']
    })

    # Преобразуем предсказания в словарь
    correlations = dict(zip(functions, predictions[0]))

    # Сортируем корреляции от большего к меньшему
    sorted_correlations = dict(sorted(correlations.items(), key=lambda item: item[1], reverse=True))

    # Выводим таблицу
    print("\nКорреляции с функциями:")
    print("{:<5} {:<10}".format('Функция', 'Корреляция'))
    print("-" * 20)
    for func, corr in sorted_correlations.items():
        print("{:<5} {:<10.4f}".format(func, corr))

    # Определяем типы, которые согласились бы с утверждением
    type_scores = {}
    base_weight = 2    # Вес базовой функции
    creative_weight = 1  # Вес творческой функции

    for type_name, type_funcs in types_functions.items():
        base_func = type_funcs[0]
        creative_func = type_funcs[1]

        # Получаем корреляции для функций, если они отсутствуют, то считаем их равными 0
        base_corr = correlations.get(base_func, 0)
        creative_corr = correlations.get(creative_func, 0)

        # Расчитываем суммарный балл с учетом весов
        score = base_corr * base_weight + creative_corr * creative_weight
        type_scores[type_name] = score

    # Сортируем типы по их суммарным баллам
    sorted_types = sorted(type_scores.items(), key=lambda item: item[1], reverse=True)

    # Выводим типы, которые согласились бы с утверждением
    top_agree_types = [type_name for type_name, score in sorted_types[:3]]
    top_disagree_types = [type_name for type_name, score in sorted_types[-3:]]

    print(f"\nСкорее всего, с утверждением согласились бы типы: {', '.join(top_agree_types)}.")
    print(f"И не согласились бы типы: {', '.join(top_disagree_types)}.")

    return sorted_correlations

# Пример использования
new_question = input("\nВведите вопрос для предсказания корреляций:\n")
predict_correlations(new_question)
