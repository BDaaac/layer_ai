"""
Интеграция модели Saiga-2 для юридического ассистента.
Использует llama.cpp для работы с GGUF моделями.
"""

import os
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Проверяем доступность llama_cpp
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama_cpp not available. Install with: pip install llama-cpp-python")


class SaigaNotInstalledError(Exception):
    """Исключение когда модель Saiga-2 не найдена"""
    pass


class SaigaFallbackLawyer:
    """
    Fallback класс для работы без llama_cpp.
    Генерирует ответы на основе найденного контекста.
    """
    
    def __init__(self):
        self.is_loaded = True  # Всегда готов к работе
        
    def load_model(self) -> bool:
        """Fallback модель всегда готова"""
        return True
        
    def is_model_available(self) -> bool:
        """Fallback модель всегда доступна"""
        return True
        
    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Генерирует ответ на основе контекста без использования LLM.
        
        Args:
            question: Вопрос пользователя
            context_chunks: Найденные фрагменты
            
        Returns:
            Dict: Ответ с метаданными
        """
        if not context_chunks:
            return {
                'success': True,
                'answer': 'Извините, не удалось найти релевантную информацию в базе данных для ответа на ваш вопрос. Попробуйте переформулировать запрос.',
                'context_used': [],
                'tokens_used': 0,
                'model': 'Fallback (без LLM)'
            }
        
        # Составляем ответ на основе найденных фрагментов
        context_text = "\n\n".join([chunk['chunk'] for chunk in context_chunks[:3]])
        sources = list(set([chunk['metadata']['source'] for chunk in context_chunks[:3]]))
        
        answer = f"""На основе найденной информации в правовых документах:

{context_text}

**Источники:** {', '.join(sources)}

*Примечание: Этот ответ составлен на основе найденных фрагментов документов. Для получения более детального анализа установите llama-cpp-python и модель Saiga-2.*"""
        
        return {
            'success': True,
            'answer': answer,
            'context_used': context_chunks[:3],
            'tokens_used': len(answer.split()),
            'model': 'Fallback (поиск по документам)'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Информация о fallback модели"""
        return {
            'model_type': 'Fallback',
            'model_loaded': True,
            'model_file_exists': False,
            'model_size_mb': 0,
            'context_length': 0,
            'description': 'Режим работы без LLM модели'
        }


class SaigaLawyer:
    """
    Юридический ассистент на базе модели Saiga-2.
    Использует GGUF модель через llama_cpp для генерации ответов.
    """
    
    def __init__(self, model_path: str = "models/saiga/saiga2.gguf"):
        """
        Инициализация Saiga-2 модели.
        
        Args:
            model_path: Путь к файлу модели GGUF
        """
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Загружает модель Saiga-2.
        
        Returns:
            bool: True если модель успешно загружена
            
        Raises:
            SaigaNotInstalledError: Если модель не найдена
        """
        if not LLAMA_CPP_AVAILABLE:
            raise SaigaNotInstalledError(
                "llama_cpp не установлен. Установите: pip install llama-cpp-python"
            )
        
        if not os.path.exists(self.model_path):
            raise SaigaNotInstalledError(
                f"Модель Saiga-2 не найдена: {self.model_path}\n"
                "Скачайте модель с: https://huggingface.co/IlyaGusev/saiga2_7b_gguf"
            )
        
        try:
            logger.info(f"Загружаю модель Saiga-2 из {self.model_path}...")
            
            # Параметры модели для оптимальной работы
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=4096,          # Контекстное окно
                n_threads=4,         # Количество потоков
                n_gpu_layers=0,      # GPU слои (0 = только CPU)
                verbose=False,       # Отключаем подробный вывод
                use_mmap=True,       # Используем memory mapping
                use_mlock=False,     # Не блокируем память
            )
            
            self.is_loaded = True
            logger.info("✅ Модель Saiga-2 успешно загружена!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {str(e)}")
            raise SaigaNotInstalledError(f"Ошибка загрузки модели: {str(e)}")
    
    def _build_prompt(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Создает промпт для модели в нужном формате.
        
        Args:
            question: Вопрос пользователя
            context_chunks: Найденные фрагменты документов
            
        Returns:
            str: Сформированный промпт
        """
        # Системная инструкция
        system_prompt = """Ты — опытный юрист, специализирующийся на законодательстве Республики Казахстан.

ПРАВИЛА:
- Отвечай строго на основе предоставленного контекста
- Используй только факты из законодательных актов
- Указывай конкретные статьи и источники
- Если информации недостаточно, честно скажи об этом
- Не придумывай несуществующие нормы
- Отвечай четко и структурированно на русском языке

ФОРМАТ ОТВЕТА:
1. Краткий ответ на вопрос
2. Ссылки на конкретные статьи
3. Объяснение применимых норм
4. При необходимости - практические рекомендации"""

        # Формируем контекст из найденных документов
        context_text = ""
        if context_chunks:
            context_text = "КОНТЕКСТ ИЗ ЗАКОНОДАТЕЛЬСТВА:\n\n"
            for i, chunk in enumerate(context_chunks, 1):
                source = chunk['metadata']['source']
                content = chunk['chunk']
                relevance = chunk['score']
                
                context_text += f"[Источник {i}: {source} (релевантность: {relevance:.2f})]\n"
                context_text += f"{content}\n\n"
        else:
            context_text = "КОНТЕКСТ: Релевантная информация в базе законов не найдена.\n\n"
        
        # Собираем финальный промпт
        full_prompt = f"""<s>system
{system_prompt}</s>

<s>user
{context_text}

ВОПРОС КЛИЕНТА:
{question}

Проанализируй предоставленную правовую информацию и дай подробный, юридически грамотный ответ.</s>

<s>assistant
"""
        
        return full_prompt
    
    def answer(self, question: str, context_chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Генерирует ответ на юридический вопрос.
        
        Args:
            question: Вопрос пользователя
            context_chunks: Найденные релевантные фрагменты
            
        Returns:
            Dict: Ответ с метаданными
        """
        if not self.is_loaded:
            try:
                self.load_model()
            except SaigaNotInstalledError:
                raise
        
        try:
            # Строим промпт
            prompt = self._build_prompt(question, context_chunks or [])
            
            logger.info("🤖 Генерирую ответ с помощью Saiga-2...")
            
            # Генерируем ответ
            response = self.model(
                prompt,
                max_tokens=500,        # Максимум токенов
                temperature=0.2,       # Низкая температура для точности
                top_p=0.9,            # Top-p sampling
                top_k=40,             # Top-k sampling
                repeat_penalty=1.1,   # Штраф за повторения
                stop=["</s>", "<s>", "user:", "system:", "assistant:"],  # Стоп-токены
                echo=False,           # Не повторять промпт
            )
            
            # Извлекаем текст ответа
            answer_text = response['choices'][0]['text'].strip()
            
            # Очищаем ответ от служебных токенов
            for stop_token in ["</s>", "<s>", "user", "system", "assistant"]:
                answer_text = answer_text.replace(stop_token, "").strip()
            
            logger.info("✅ Ответ сгенерирован!")
            
            return {
                'answer': answer_text,
                'context_used': context_chunks or [],
                'question': question,
                'model_used': 'Saiga-2 7B',
                'context_count': len(context_chunks) if context_chunks else 0,
                'tokens_used': response['usage']['total_tokens'],
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {str(e)}")
            return {
                'answer': f'Произошла ошибка при генерации ответа: {str(e)}',
                'context_used': context_chunks or [],
                'question': question,
                'model_used': 'Saiga-2 7B (ERROR)',
                'error': str(e),
                'success': False
            }
    
    def is_model_available(self) -> bool:
        """
        Проверяет доступность модели без попытки загрузки.
        
        Returns:
            bool: True если модель доступна
        """
        return (LLAMA_CPP_AVAILABLE and 
                os.path.exists(self.model_path) and 
                os.path.getsize(self.model_path) > 1000000)  # Больше 1MB
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о модели.
        
        Returns:
            Dict: Информация о модели
        """
        info = {
            'model_path': self.model_path,
            'llama_cpp_available': LLAMA_CPP_AVAILABLE,
            'model_file_exists': os.path.exists(self.model_path),
            'model_loaded': self.is_loaded,
            'model_size_mb': 0
        }
        
        if os.path.exists(self.model_path):
            info['model_size_mb'] = os.path.getsize(self.model_path) / (1024 * 1024)
        
        return info


# Глобальный экземпляр для использования в других модулях (инициализируется по требованию)
saiga_lawyer = None

def initialize_saiga():
    """Initialize Saiga model (call this from Streamlit app)"""
    global saiga_lawyer
    if saiga_lawyer is None:
        if not LLAMA_CPP_AVAILABLE:
            logger.info("llama_cpp not available, using fallback mode")
            saiga_lawyer = SaigaFallbackLawyer()
            return True
        
        saiga_lawyer = SaigaLawyer()
        
        try:
            success = saiga_lawyer.load_model()
            if success:
                logger.info("Saiga model initialized successfully")
                return True
            else:
                logger.warning("Failed to initialize Saiga model, using fallback")
                saiga_lawyer = SaigaFallbackLawyer()
                return True
        except Exception as e:
            logger.error(f"Error initializing Saiga: {e}, using fallback")
            saiga_lawyer = SaigaFallbackLawyer()
            return True
    return True


def generate_answer_with_saiga(question: str, context_chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Удобная функция для генерации ответов с Saiga-2.
    
    Args:
        question: Вопрос пользователя
        context_chunks: Найденные фрагменты
        
    Returns:
        Dict: Ответ с метаданными
        
    Raises:
        SaigaNotInstalledError: Если модель недоступна
    """
    global saiga_lawyer
    
    # Инициализируем модель если не инициализирована
    if saiga_lawyer is None:
        if not initialize_saiga():
            return {
                'success': False,
                'error': 'Failed to initialize Saiga model',
                'answer': 'Извините, модель Saiga-2 недоступна. Проверьте установку llama-cpp-python.'
            }
    
    return saiga_lawyer.generate_answer(question, context_chunks)


def is_saiga_available() -> bool:
    """
    Проверяет доступность Saiga-2 модели (включая fallback режим).
    
    Returns:
        bool: True если модель доступна (включая fallback)
    """
    global saiga_lawyer
    
    # Fallback режим всегда доступен
    if not LLAMA_CPP_AVAILABLE:
        return True
    
    # Проверяем существование файла модели
    model_path = "models/saiga/saiga2.gguf"
    if not os.path.exists(model_path):
        return True  # Fallback режим доступен
    
    # Инициализируем если нужно
    if saiga_lawyer is None:
        return initialize_saiga()
    
    return saiga_lawyer.is_model_available()


if __name__ == "__main__":
    # Тестирование интеграции
    print("🧪 Тестирование Saiga-2 интеграции...")
    
    # Проверяем доступность
    if is_saiga_available():
        print("✅ Модель Saiga-2 доступна")
        
        # Тестовый контекст
        test_context = [{
            'chunk': 'Собственнику принадлежат права владения, пользования и распоряжения своим имуществом.',
            'metadata': {'source': 'civil_code_kz.txt'},
            'score': 0.9,
            'rank': 1
        }]
        
        try:
            # Тестовый вопрос
            result = generate_answer_with_saiga(
                "Что такое право собственности?", 
                test_context
            )
            
            if result['success']:
                print("✅ Тест прошел успешно!")
                print(f"Ответ: {result['answer'][:100]}...")
                print(f"Использовано токенов: {result.get('tokens_used', 'N/A')}")
            else:
                print(f"❌ Ошибка в ответе: {result.get('error', 'Unknown')}")
                
        except SaigaNotInstalledError as e:
            print(f"❌ {e}")
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}")
    else:
        print("❌ Модель Saiga-2 недоступна")
        
        # Показываем информацию о модели
        info = saiga_lawyer.get_model_info()
        print("📊 Информация о модели:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        if not info['llama_cpp_available']:
            print("\n💡 Установите llama-cpp-python: pip install llama-cpp-python")
        
        if not info['model_file_exists']:
            print(f"\n💡 Скачайте модель в: {info['model_path']}")
            print("   Ссылка: https://huggingface.co/IlyaGusev/saiga2_7b_gguf")