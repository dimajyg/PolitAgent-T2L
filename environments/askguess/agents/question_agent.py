from llm.agent import BaseAgent
from environments.askguess.utils.prompt import get_questioner_prompt_template
from environments.askguess.utils.utils import create_message
from typing import Dict, List, Any, Optional, Tuple

class QuestionAgent(BaseAgent):
    """
    Агент-вопрошающий для игры AskGuess.
    
    Args:
        llm: LangChain-совместимая модель
        object_name: Слово, которое нужно угадать
        args: Аргументы игры
    """
    def __init__(self, llm, object_name, args) -> None:
        self.object_name = object_name
        self.mode = args.mode
        self.prompt_template = get_questioner_prompt_template(self.mode)
        
        # Получаем шаблон как строку для BaseAgent
        template_str = self.prompt_template.template
        
        # Инициализируем базовый класс
        super().__init__("Questioner", llm, template_str)
        
        # Инициализируем private_history для хранения диалога
        self.private_history = []
        
        # Формируем system prompt через PromptTemplate
        system_prompt = self.prompt_template.format(word=self.object_name)
        role_message = create_message("system", system_prompt)
        self.private_history.append(role_message)
    
    def get_role_description(self) -> str:
        """Возвращает описание роли агента."""
        return "The questioner in the Ask & Guess game"

    def chat(self, context: str) -> Tuple[str, Dict[str, Any]]:
        """Отправляет сообщение пользователя и получает ответ."""
        messages = self.private_history + [create_message("user", context)]
        response = super().chat(messages)
        return response, {"answer": response}

    def play(self) -> str:
        """Генерирует ход агента на основе текущей истории."""
        messages = self.private_history.copy()
        response = super().chat(messages)
        return response
        
    def update_history(self, message: Dict[str, str]) -> None:
        """Добавляет сообщение в историю диалога."""
        self.private_history.append(message)



