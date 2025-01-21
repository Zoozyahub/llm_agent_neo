from langchain_community.tools import DuckDuckGoSearchRun

def retrieve_from_db(collection, query, embedder, n_results=1):
    # Генерируем эмбеддинг для пользовательского запроса
    query_embedding = embedder.encode(query)
    
    # Выполняем запрос в ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # Получаем документы, метаданные (в данном случае это вопросы), и соответствующие идентификаторы
    documents = results.get('documents', [])  # Список ответов
    metadatas = results.get('metadatas', [])  # Список вопросов
    ids = results.get('ids', [])              # Список идентификаторов
    
    # Формируем список вопрос-ответ пар
    qa_pairs = [
        {"id": doc_id, "question": meta, "answer": doc}
        for doc_id, meta, doc in zip(ids, metadatas, documents)
    ]
    
    res = {}
    for item in qa_pairs:
        res["Вопрос:"] =  item['question'][0]['question']
        res["Ответ:"] = item['answer'][0]
    
    return res


def search_duckduckgo(query, max_results=1): 
    tool = DuckDuckGoSearchRun(max_results=max_results)
    results = tool.invoke(query)
    return results