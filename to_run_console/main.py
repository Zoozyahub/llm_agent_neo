from database import initialize_embedder, initialize_chroma, load_dataset, populate_collection
from langchain_openai import ChatOpenAI
from agent import Agent




if __name__ == "__main__":
    print('loading...')
    embedder = initialize_embedder()
    df = load_dataset(split="test")
    collection = initialize_chroma()
    populate_collection(collection, df, embedder)
    
    model_mistral = ChatOpenAI(base_url="http://localhost:11434/v1", api_key='1', model="mistral")
    
    # Инициализация модели и коллекции
    model = model_mistral
    collection = collection
    embedder = embedder  

    # Создание агента
    agent = Agent(model, collection, embedder)
    state = {
    "messages": [{"content": "На чем играл Джимми Хендрикс?"}],
    "memory": [],
    "db_results": None,
    "web_results": None
    }
    
    print("Чат-бот готов к работе! Введите 'exit' для завершения.")
    
    while True:
        # Получаем ввод пользователя
        user_input = input("Вы: ")
        
        # Проверяем команду выхода
        if user_input.lower() == 'exit':
            print("Чат завершен. До свидания!")
            break
        
        # Добавляем сообщение пользователя в состояние
        state["messages"] = [{"content": user_input}]
        
        # Передаем состояние агенту
        try:
            result = agent.invoke(state)
            
            # Получаем последний ответ из памяти
            bot_response = result["memory"][-1][7:]  # Пропускаем служебную информацию
            
            # Выводим ответ бота
            print("Бот:", bot_response)
        except Exception as e:
            print("Произошла ошибка:", e)