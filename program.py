import datetime
import os

import faiss
import numpy as np
from memory_profiler import profile
from sentence_transformers import SentenceTransformer


def Message(s: str, beforeTimestamp):
    currentTime=datetime.datetime.now()
    print(s,currentTime, f"({(currentTime-beforeTimestamp)})")
    return datetime.datetime.now()

@profile
def mainFunc():
    currTime=Message(f"Started...",datetime.datetime.now())
    # загрузка предобученной модели и создание индекса faiss
    model = SentenceTransformer('cointegrated/rubert-tiny')
    index = faiss.IndexHNSWFlat(312, 5)

    currTime=Message(f"Начало добавления векторов в индекс...",currTime)
    # создание вектора из эмбеддингов по каждому файлу и добавление в индекс
    for vector in GetVectorsByFiles("texts", model):
        index.add(vector)

    # осуществить поиск по индексу из содержимого заданного файла
    currTime=Message(f"Загрузка содержимого файла для поиска...",currTime)
    with open("ex.txt", 'r') as f:
        query_for_search = model.encode([f.read()])

    # index.add(np.load("vs_dir.npy"))
    currTime=Message(f"Начало поиска...",currTime)
    res = index.search(query_for_search, 3)  # index.ntotal)
    currTime=Message(f"Поиск занял...",currTime)
    # # print(res[0])
    print(res[1])


def CreateVectors(model, filename: str = 'vectors2.npy', data_dir: str = "texts"):
    """
    Создание файла с векторами для каждого файла из папки.
    :param filename: Имя для файла с векторами.
    :param data_dir: Папка, из которой будут сканироваться все файлы.
    :return: None, но создает файл.
    """
    file_list = os.listdir(data_dir)
    # Создание пустого массива для векторных представлений
    vectors = np.zeros((len(file_list), 312))
    for i, file_name in enumerate(file_list):
        with open(os.path.join(data_dir, file_name), 'r') as f:
            text = f.read()
            vectors[i, :] = model.encode(text)
    np.save(filename, vectors)

def GetVectorsByFiles(dir, model: SentenceTransformer):
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            fullPath = os.path.join(dirpath, filename)
            with open(fullPath, 'r') as f:
                yield np.reshape(model.encode(f.read(), convert_to_numpy=True), (1, -1))


if __name__ == "__main__":
    # process = psutil.Process()
    # start_memory = process.memory_info().rss
    mainFunc()
    # end_memory = process.memory_info().rss
    # print(f"Memory usage: {(end_memory - start_memory) / 1024 / 1024} MB")
