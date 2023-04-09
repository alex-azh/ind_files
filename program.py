import datetime
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
# import psutil
from sentence_transformers import SentenceTransformer

from FileReaders import PDFReader, DOCReader


def SaveArray(arr: list, name: str):
    with open(name + '.pkl', 'wb') as file:
        pickle.dump(arr, file)


def OpenArray(name: str) -> list:
    with open(name + '.pkl', 'rb') as file:
        loaded_array = pickle.load(file)
    return loaded_array


def Message(s: str, beforeTimestamp):
    currentTime = datetime.datetime.now()
    print(s, currentTime, f"({(currentTime - beforeTimestamp)})")
    return datetime.datetime.now()


# @profile
def mainFunc():
    currTime = Message(f"Started...", datetime.datetime.now())
    # загрузка предобученной модели и создание индекса faiss
    model = SentenceTransformer('cointegrated/rubert-tiny')
    indexName = "myindex"
    if os.path.isfile(indexName):
        currTime = Message(f"Инициализация индекса из файла...", currTime)
        index = faiss.read_index(indexName)
    else:
        currTime = Message(f"Начало добавления векторов в индекс...", currTime)
        if indexName.endswith("2"):
            index = faiss.IndexFlatL2(312)
        else:
            index = faiss.IndexHNSWFlat(312, 5)
        readedFiles = []
        nonReadedFiles = []
        # создание вектора из эмбеддингов по каждому файлу и добавление в индекс
        for vector in GetVectorsByFiles("texts", model, readedFiles, nonReadedFiles):
            index.add(vector)
        print(nonReadedFiles)
        currTime = Message(f"Сохранение индекса...", currTime)
        faiss.write_index(index, indexName)

    # осуществить поиск по индексу из содержимого заданного файла
    currTime = Message(f"Загрузка содержимого файла для поиска...", currTime)
    with open("ex.txt", 'r') as f:
        query_for_search = model.encode([f.read()])

    currTime = Message(f"Начало поиска...", currTime)
    res = index.search(query_for_search, 3)  # index.ntotal)
    currTime = Message(f"Поиск занял...", currTime)
    print(res[1])
    with open("ex2.txt", 'r') as f:
        query_for_search = model.encode([f.read()])
    currTime = Message(f"Начало поиска...", currTime)
    res = index.search(query_for_search, 3)  # index.ntotal)
    currTime = Message(f"Поиск занял...", currTime)
    print(res[1])
    text = "The analytic continuation is an old yet persistent problem,"
    currTime = Message(f"Начало поиска 3...", currTime)
    res = index.search(model.encode([text]), 3)
    currTime = Message(f"Конец поиска 3...", currTime)
    # readedFiles=[]
    # for i, file in enumerate(os.listdir("texts")):
    #     readedFiles.append(file)


def GetVectorsByFiles(dir, model: SentenceTransformer, readedFiles: list, nonReadedFiles: list):
    i = 0
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            fullPath = os.path.join(dirpath, filename)
            text = GetTextFromFile(fullPath)
            if text == None:
                nonReadedFiles.append(fullPath)
            else:
                yield np.reshape(model.encode(text, convert_to_numpy=True), (1, -1))
                readedFiles.append(fullPath)
                i += 1
            SaveArray()


def GetTextFromFile(path: str):
    ppath = Path(path)
    suffix = ppath.suffix
    notSupported = [".sys", ".exe"]
    # более 500 мб не проверять
    if suffix in notSupported or ppath.stat().st_size > 524288000:
        return None
    if suffix == ".pdf":
        return PDFReader(path)
    elif suffix in [".doc", ".docx"]:
        return DOCReader(path)
    else:
        try:
            with open(path, 'r') as f:
                text = f.read()
            return text
        except:
            text = PDFReader(path)
            if text == None:
                return DOCReader(path)


if __name__ == "__main__":
    # process = psutil.Process()
    # start_memory = process.memory_info().rss
    mainFunc()
    # end_memory = process.memory_info().rss
    # print(f"Memory usage: {(end_memory - start_memory) / 1024 / 1024} MB")
