import datetime
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# Загрузка модели RuBERT и токенизатора
model_name = 'cointegrated/rubert-tiny'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model2 = SentenceTransformer(model_name)
# Путь к папке с документами
data_dir = 'texts'

# Путь к файлу для сохранения векторных представлений
output_file = 'vectors2.npy'

# Получение списка файлов в папке
file_list = os.listdir(data_dir)

# Создание пустого массива для векторных представлений
vectors = np.zeros((len(file_list), model.config.hidden_size))
# Итерация по каждому файлу в папке
start=datetime.datetime.now()
for i, file_name in enumerate(file_list):
    # Чтение файла и токенизация его содержимого
    document_embedding=None
    start=datetime.datetime.now()
    # with open(os.path.join(data_dir, file_name), 'r') as f:
    #     text = f.read()
    #     encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    #     output = model(**encoded_input)[1]
    #     document_embedding = output.detach().numpy()
    #     print(document_embedding)
    print(file_name)
    with open(os.path.join(data_dir, file_name), 'r') as f:
        text = f.read()
        document_embedding = model2.encode(text)
    vectors[i, :] = document_embedding

print(datetime.datetime.now()-start)
# Сохранение векторных представлений в файл
# np.save(output_file, vectors)

