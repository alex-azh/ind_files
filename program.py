import datetime
import os
from pathlib import Path

import faiss
import numpy as np
# import psutil
from memory_profiler import profile
from sentence_transformers import SentenceTransformer

import MyDB.db as db
from FileReaders import PDFReader, DOCReader


# faiss_index = faiss.IndexFlatL2(128)
# faiss_index.metric_type = faiss.METRIC_INNER_PRODUCT
# // normlize matrix
# faiss_index.add(np.asmatrix(candidates, np.float32))
# I was reading the doc, can I just replace the first two line with
# faiss_index = faiss.IndexFlatIP(128)


def Message(s: str, beforeTimestamp):
    currentTime = datetime.datetime.now()
    print(s, currentTime, f"({(currentTime - beforeTimestamp)})")
    return datetime.datetime.now()


@profile
def mainFunc():
    currTime = Message(f"Started...", datetime.datetime.now())
    # загрузка предобученной модели и создание индекса faiss
    model = SentenceTransformer('cointegrated/rubert-tiny')
    indexName = "myindex2"
    tmpP=False
    if os.path.isfile(indexName) and tmpP:
        currTime = Message(f"Инициализация индекса из файла...", currTime)
        index = faiss.read_index(indexName)
    else:
        currTime = Message(f"Начало добавления векторов в индекс...", currTime)
        if indexName.endswith("2"):
            index = faiss.IndexFlatL2(312)
        else:
            index = faiss.IndexHNSWFlat(312, 8)
        # создание вектора из эмбеддингов по каждому файлу и добавление в индекс
        for vector in GetVectorsByFiles("C:/", model):
            # currTime = Message(f"Добавление в индекс", currTime)
            index.add(vector)
            faiss.write_index(index, indexName)
        # currTime = Message(f"Сохранение индекса...", currTime)
        # faiss.write_index(index, indexName)

    # # осуществить поиск по индексу из содержимого заданного файла
    # currTime = Message(f"Загрузка содержимого файла для поиска...", currTime)
    # with open("ex.txt", 'r') as f:
    #     query_for_search = model.encode([f.read()])

    # currTime = Message(f"Начало поиска...", currTime)
    # res = index.search(query_for_search, 3)  # index.ntotal)
    # currTime = Message(f"Поиск занял...", currTime)
    # print(res[1])
    # with open("ex2.txt", 'r',encoding="utf-8") as f:
    #     query_for_search = model.encode([f.read()])
    # currTime = Message(f"Начало поиска...", currTime)
    # res = index.search(query_for_search, 3)  # index.ntotal)
    # currTime = Message(f"Поиск занял...", currTime)
    # print(res[1])
    # text = "The analytic continuation is an old yet persistent problem,"
    # currTime = Message(f"Начало поиска 3...", currTime)
    # res = index.search(model.encode([text]), 3)
    # currTime = Message(f"Конец поиска 3...", currTime)


i = 0

def GetVectorsByFiles(dir, model: SentenceTransformer):
    global i
    for dirpath, dirs, filenames in os.walk(dir):
        if dirpath.startswith("C:/Windows") or dirpath.startswith("C:/$RECYCLE.BIN") or dirpath.startswith("C:/AMD"):
                continue
        for filename in filenames:
            fullPath = os.path.join(dirpath, filename)
            text = GetTextFromFile(fullPath)
            if text == None:
                db.AddRow(fullPath, None)
            else:
                yield np.reshape(model.encode(text, convert_to_numpy=True), (1, -1))
                db.AddRow(fullPath, i)
                i += 1
        for dir in dirs:
            GetVectorsByFiles(Path(r"{dirpath+dir}"), model)


def GetTextFromFile(path: str):
    ppath = Path(path)
    suffix = ppath.suffix
    notSupported = [".sys", ".exe",".dll", ".bin", ".iso"]
    # более 500 мб не проверять
    if suffix in notSupported or ppath.stat().st_size > 524288000:
        return None
    if suffix == ".pdf":
        return PDFReader(path)
    elif suffix in [".doc", ".docx"]:
        return DOCReader(path)
    else:
        try:
            with open(path, 'r',encoding="utf-8") as f:
                text = f.read()
            return text
        except Exception as ex:
            # print(ex)
            text = PDFReader(path)
            return text if text!=None else DOCReader(path)


if __name__ == "__main__":
    # process = psutil.Process()
    # start_memory = process.memory_info().rss
    mainFunc()
    # end_memory = process.memory_info().rss
    # print(f"Memory usage: {(end_memory - start_memory) / 1024 / 1024} MB")
