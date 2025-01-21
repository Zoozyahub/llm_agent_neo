from sentence_transformers import SentenceTransformer
import pandas as pd
from chromadb import Client, Settings

# Инициализация эмбеддера
def initialize_embedder():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Загрузка данных
def load_dataset(split="test"):
    splits = {
        'test': 'data/test-00000-of-00001-d519841742f463e6.parquet',
        'dev': 'data/dev-00000-of-00001-d7e3040a344e1e68.parquet'
    }
    return pd.read_parquet("hf://datasets/d0rj/RuBQ_2.0/" + splits[split])

# Инициализация базы данных Chroma
def initialize_chroma(persist_directory="./chroma_db"):
    client = Client(settings=Settings(persist_directory=persist_directory))
    return client.create_collection(name="rubq_light")

def delete_chroma(name, persist_directory="./chroma_db"):
    client = Client(settings=Settings(persist_directory=persist_directory))
    client.delete_collection(name)
    
def open_chrome(persist_directory="./chroma_db"):
    client = Client(settings=Settings(persist_directory))
    return client.get_collection(name="rubq_light")

# Добавление данных в коллекцию
def populate_collection(collection, df, embedder):
    for idx, row in df[['question_text', 'answer_text']].iterrows():
        question = row['question_text']
        answer = row['answer_text']
        embedding = embedder.encode(question)
        
        collection.add(
            ids=[str(idx)],
            embeddings=[embedding],
            documents=[answer],
            metadatas=[{"question": question}]
        )
