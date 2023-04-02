import datetime
import os

import faiss
import numpy as np
# from memory_profiler import profile
import psutil
from sentence_transformers import SentenceTransformer


# @profile
def mainFunc():
    model_name = 'cointegrated/rubert-tiny'
    model = SentenceTransformer(model_name)
    index = faiss.IndexHNSWFlat(312, 5)
    # CreateVectors(model=model,filename="vs_dir.npy",data_dir="dir")
    # for i, (file, emb) in enumerate(GetEmbeddings(model,"texts")):
    #     print(i, file)

    query_for_search = None
    with open("ex.txt", 'r') as f:
        query_for_search = model.encode([f.read()])

    index.add(np.load("vs_dir.npy"))
    start = datetime.datetime.now()
    res = index.search(query_for_search, index.ntotal)
    print("Время на поиск среди всех:", datetime.datetime.now() - start)

    start = datetime.datetime.now()
    res = index.search(query_for_search, 3)
    print("Время на поиск среди 10 первых:", datetime.datetime.now() - start)

def CreateVectors(model,filename: str = 'vectors2.npy', data_dir: str = "texts"):
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

def GetEmbeddings(model,dir):
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            # Получение полного пути к файлу
            file_path = os.path.join(dirpath, filename)
            # Обработка файла
            with open(file_path, 'r') as f:
                text = f.read()
                yield file_path, model.encode(text)

if __name__=="__main__":
    process = psutil.Process()
    start_memory = process.memory_info().rss
    mainFunc()
    end_memory = process.memory_info().rss
    print(f"Memory usage: {(end_memory - start_memory) / 1024 / 1024} MB")


