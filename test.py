import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import faiss
from pathlib import Path
model_name='cointegrated/rubert-tiny'

start=datetime.datetime.now()

with open(Path(r"C:/Program Files\dotnet\packs\Microsoft.Android.Ref.33\33.0.4\ref\net7.0\Mono.Android.dll"), 'r',encoding="UTF-8") as f:
    text=f.read()
model = SentenceTransformer(model_name)
m=model.encode([text])
index = faiss.read_index("myindex2")
res=index.search(m, 12)  # index.ntotal)
print(res[0])
print(res[1])
# model=AutoModel.from_pretrained(model_name)
# tk = AutoTokenizer.from_pretrained(model_name)
# start=datetime.datetime.now()
# encoded=tk(text,padding=True,truncation = True, return_tensors='pt')
# output=model(**encoded)[1]
# emd=output.detach().numpy()
# end=datetime.datetime.now()
# print(end-start)

