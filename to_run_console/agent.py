from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Any
import operator
from tools import search_duckduckgo, retrieve_from_db
from prompts import RELEVANCE_PROMPT, REFINE_QUERY_PROMPT, REDACT_PROMPT


from dotenv import load_dotenv
_ = load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[Any], operator.add]  # Сообщения (вопросы и ответы)
    memory: list[str]  # Память (история диалога)
    db_results: dict  # Результаты поиска в БД
    web_results: str  # Результаты поиска в интернете


class Agent:
    def __init__(self, model, collection, embedder, system=""):
        self.model = model
        self.collection = collection
        self.embedder = embedder
        self.system = system
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)

        # Добавляем узлы
        graph.add_node("refine_query", self.refine_query)  # Новый узел
        graph.add_node("search_db", self.search_db)
        graph.add_node("evaluate_relevance", self.evaluate_relevance)
        graph.add_node("search_web", self.search_web)
        graph.add_node("edit_web_response", self.edit_web_response)

        # Добавляем переходы
        graph.add_edge("refine_query", "search_db")  # После уточнения ищем в БД
        graph.add_edge("search_db", "evaluate_relevance")
        graph.add_conditional_edges(
            "evaluate_relevance",
            self.decide_next_step,
            {"end": END, "search_web": "search_web"}
        )
        graph.add_edge("search_web", "edit_web_response")
        graph.add_edge("edit_web_response", END)

        # Устанавливаем начальный узел
        graph.set_entry_point("refine_query")

        return graph.compile()
    
    def refine_query(self, state: AgentState):
        """
        Уточняет запрос пользователя на основе контекста.
        """
        user_query = state["messages"][-1]["content"]
        context = "\n".join(state["memory"])  # История диалога
        
        if 'По поиску в RuBQ Ответ:' in context:
            context = context.replace('По поиску в RuBQ Ответ:', '')
        elif 'По поиску в DuckDuckGo:' in context:
            context = context.replace('По поиску в DuckDuckGo:') 
        # print('----')
        # print('Что в истории до редактирования', context )
        # print('----')        

        # Формируем промпт для уточнения запроса
        prompt = REFINE_QUERY_PROMPT.format(
            context=context,
            human_message=user_query
        )

        # Запрашиваем уточнение у LLM
        response = self.model.invoke(prompt)
        refined_query = response.content.strip()

        # Сохраняем уточнённый запрос в состоянии
        print('Отредактированый вопрос:', refined_query)
        state["messages"][-1]["content"] = refined_query
        return state

    def search_db(self, state: AgentState):
        """
        Ищет в базе данных и сохраняет результаты в состоянии.
        """
        user_query = state["messages"][-1]["content"]
        db_results = retrieve_from_db(self.collection, user_query, self.embedder)
        state["db_results"] = db_results
        return state

    def evaluate_relevance(self, state: AgentState):
        """
        Оценивает релевантность ответа из БД с помощью LLM.
        """
        user_query = state["messages"][-1]["content"]
        db_question = state["db_results"]["Вопрос:"]
        db_answer = state["db_results"]["Ответ:"]

        # Формируем промпт для оценки релевантности
        prompt = RELEVANCE_PROMPT.format(
            user_query=user_query,
            db_question=db_question,
            db_answer=db_answer
        )

        # Запрашиваем оценку у LLM
        response = self.model.invoke(prompt)
        # print(f'Вопрос бд:{db_question}, Ответ бд:{db_answer}')
        # print(f"Оценка релевантности: {response.content}")  # Отладочный вывод

        # Если ответ релевантен, добавляем его в память
        if "да" in response.content.lower():
            # state["memory"].append(f"По поиску в RuBQ Ответ: {db_answer}")
            self.update_memory(state, user_query, f"По поиску в RuBQ Ответ: {db_answer}")
            state["next_step"] = "end"  # Завершаем выполнение
        else:
            state["next_step"] = "search_web"  # Переходим к поиску в интернете

        return state

    def search_web(self, state: AgentState):
        """
        Ищет в интернете с помощью DuckDuckGo.
        """
        user_query = state["messages"][-1]["content"]
        web_results = self.search_duckduckgo(user_query)
        state["web_results"] = web_results
        return state

    def edit_web_response(self, state: AgentState):
        """
        Редактирует ответ из интернета с помощью LLM.
        """
        user_query = state["messages"][-1]["content"]
        web_results = state["web_results"]

        # Формируем промпт для редактирования

        # Запрашиваем редактирование у LLM
        response = self.model.invoke(
            REDACT_PROMPT.format(user_query=user_query, web_results=web_results)
        )


        # Добавляем отредактированный ответ в память
        # state["memory"].append(f"По поиску в DuckDuckGo: {response.content}")
        self.update_memory(state, user_query, f"По поиску в DuckDuckGo: {response.content}")
        return state

    def decide_next_step(self, state: AgentState):
        """
        Определяет следующий шаг на основе состояния.
        """
        return state.get("next_step", "search_web")

    def search_duckduckgo(self, query):
        """
        Ищет в интернете с помощью DuckDuckGo.
        """
        result = search_duckduckgo(query, max_results=1)
        return result

    def update_memory(self, state: AgentState, user_message, agent_message):
        """
        Обновляет память агента.
        """
        state["memory"].append(f"User: {user_message}")
        state["memory"].append(f"Agent: {agent_message}")

    def invoke(self, state: AgentState):
        """
        Запускает агента с начальным состоянием.
        """
        return self.graph.invoke(state)