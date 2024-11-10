import json
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import joblib  # Добавлен импорт joblib

# Константы
MODEL_PATH = 'function_correlation_model.keras'
LOG_DIR = 'logs/fit'
BATCH_SIZE = 8
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 5

# Загрузка данных
with open('talanovstatements.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Функции и определения
functions = ["ЧИ", "БИ", "ЧС", "БС", "БЛ", "ЧЛ", "БЭ", "ЧЭ", "БК", "ЧК", "БД", "ЧД"]
function_definitions = {
    "ЧИ": "воображение свободы / альтернатив",
    "БИ": "воображение повреждений / кризиса",
    "ЧС": "ощущение власти / доминирования",
    "БС": "ощущение комфорта / гомеостаза",
    "БЛ": "расчёт закона / права",
    "ЧЛ": "расчёт выгоды / дела",
    "БЭ": "внушение отношения",
    "ЧЭ": "внушение влечения, эмоций",
    "БК": "обособление индивидуальности",
    "ЧК": "обособление знатности, элитарности",
    "БД": "принятие подчинения, одинаковости",
    "ЧД": "принятие сотрудничества",
}

# Подготовка данных
list_of_questions = [entry['statement'] for entry in data]
labels = np.array([
    [entry['function_correlation'].get(func, 0.0) for func in functions]
    for entry in data
])

# Загрузка модели эмбеддингов предложений
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
embedding_model = SentenceTransformer(model_name)

# Вычисление эмбеддингов для утверждений
statement_embeddings = embedding_model.encode(list_of_questions)

# Нормализация меток в диапазон [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
labels_scaled = scaler.fit_transform(labels)

# Сохранение скейлера
import joblib
joblib.dump(scaler, 'label_scaler.pkl')
print("Скейлер сохранен в 'label_scaler.pkl'.")

# Разделение данных на обучающую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(
    statement_embeddings, labels_scaled, test_size=0.2, random_state=42
)

# Функция для создания модели
def create_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim, activation='sigmoid'))  # Выход в диапазоне [0, 1]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    return model

# Проверка, существует ли сохраненная модель
if os.path.exists(MODEL_PATH):
    use_saved_model = input(f"Найдена сохраненная модель в '{MODEL_PATH}'. Хотите загрузить ее? (y/n): ")
    if use_saved_model.lower() == 'y':
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = create_model(input_dim=statement_embeddings.shape[1], output_dim=len(functions))
else:
    model = create_model(input_dim=statement_embeddings.shape[1], output_dim=len(functions))

# Настройка коллбэков
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    MODEL_PATH,
    save_best_only=True,
    monitor='val_loss'
)
tensorboard_callback = TensorBoard(
    log_dir=LOG_DIR,
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# Обучение модели, если она не была загружена
if not os.path.exists(MODEL_PATH) or (os.path.exists(MODEL_PATH) and use_saved_model.lower() != 'y'):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, model_checkpoint, tensorboard_callback],
        verbose=1
    )

    # Сохранение модели
    model.save(MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")

# Функция для предсказания корреляций
def predict_correlations(statement):
    emb = embedding_model.encode([statement])
    prediction = model.predict(emb)[0]
    # Обратное преобразование меток к исходному диапазону [-1, 1]
    prediction_rescaled = scaler.inverse_transform([prediction])[0]
    correlations = dict(zip(functions, prediction_rescaled))
    sorted_correlations = dict(sorted(correlations.items(), key=lambda item: item[1], reverse=True))

    print("\nПредсказанные корреляции:")
    for func, corr in sorted_correlations.items():
        print(f"{func}: {corr:.4f}")

# Пример использования
new_statement = input("\nВведите утверждение для предсказания корреляций:\n")
predict_correlations(new_statement)
