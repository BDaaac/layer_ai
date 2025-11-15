"""
Saiga-2 7B model integration for the AI Lawyer project.
Handles model loading, prompt engineering, and answer generation.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from ctransformers import AutoModelForCausalLM
from rag import search_law

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SaigaLawyer:
    """
    AI Lawyer using Saiga-2 7B model for legal consultation.
    """
    
    def __init__(self, model_path: str = "models/saiga/saiga2.gguf"):
        """
        Initialize the Saiga lawyer.
        
        Args:
            model_path (str): Path to the Saiga-2 GGUF model file
        """
        self.model_path = model_path
        self.model = None
        self.system_prompt = self._get_system_prompt()
        
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the legal assistant.
        
        Returns:
            str: System prompt in Russian
        """
        return """Ты — профессиональный юридический ИИ-ассистент, специализирующийся на законодательстве Республики Казахстан.

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай СТРОГО на основе предоставленного законодательного контекста
2. Всегда указывай конкретные статьи, пункты и названия нормативных актов
3. Если в контексте нет информации для ответа на вопрос, четко скажи: "В предоставленных документах нет информации по данному вопросу"
4. Не выдумывай и не интерпретируй законы - только цитируй и объясняй имеющиеся нормы
5. При неопределенности рекомендуй обратиться к юристу
6. Отвечай на русском языке четко и структурированно
7. Выделяй ключевые правовые понятия и требования

ФОРМАТ ОТВЕТА:
- Краткий ответ на вопрос
- Ссылки на конкретные статьи
- Объяснение применимых норм
- При необходимости - рекомендации по дальнейшим действиям

Помни: твоя цель — предоставить точную правовую информацию, основанную исключительно на действующем законодательстве."""

    def load_model(self) -> bool:
        """
        Load the Saiga-2 model.
        
        Returns:
            bool: Success status
        """
        if self.model is not None:
            logger.info("Model already loaded")
            return True
            
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            logger.info("Please download Saiga-2 7B GGUF model and place it in models/saiga/saiga2.gguf")
            logger.info("You can download it from: https://huggingface.co/IlyaGusev/saiga2_7b_gguf")
            return False
            
        try:
            logger.info(f"Loading Saiga-2 model from {self.model_path}...")
            
            # Model configuration for optimal performance
            config = {
                'model_path': self.model_path,
                'model_type': 'llama',
                'gpu_layers': 0,  # Set to > 0 if you have GPU with enough VRAM
                'context_length': 4096,
                'max_new_tokens': 1024,
                'temperature': 0.1,  # Low temperature for more factual responses
                'top_p': 0.9,
                'top_k': 40,
                'repetition_penalty': 1.1,
                'threads': 4,
                'batch_size': 8
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                model_type='llama',
                gpu_layers=config['gpu_layers'],
                context_length=config['context_length'],
                max_new_tokens=config['max_new_tokens'],
                temperature=config['temperature'],
                top_p=config['top_p'],
                top_k=config['top_k'],
                repetition_penalty=config['repetition_penalty'],
                threads=config['threads'],
                batch_size=config['batch_size']
            )
            
            logger.info("Saiga-2 model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Format context chunks into a readable string.
        
        Args:
            context_chunks (List[Dict]): Context chunks from RAG
            
        Returns:
            str: Formatted context
        """
        if not context_chunks:
            return "Контекст не найден."
        
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk['metadata']['source']
            content = chunk['chunk']
            score = chunk['score']
            
            context_parts.append(f"[Документ {i}: {source} (релевантность: {score:.2f})]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Build the complete prompt for the model.
        
        Args:
            question (str): User question
            context_chunks (List[Dict]): Context from RAG
            
        Returns:
            str: Complete prompt
        """
        context = self._format_context(context_chunks)
        
        prompt = f"""<s>system
{self.system_prompt}</s>

<s>user
КОНТЕКСТ ИЗ ЗАКОНОДАТЕЛЬСТВА:
{context}

ВОПРОС:
{question}</s>

<s>assistant"""
        
        return prompt
    
    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]] = None, 
                       use_rag: bool = True, k: int = 3) -> Dict[str, Any]:
        """
        Generate an answer to a legal question.
        
        Args:
            question (str): User's legal question
            context_chunks (List[Dict]): Pre-retrieved context chunks
            use_rag (bool): Whether to use RAG for context retrieval
            k (int): Number of context chunks to retrieve if using RAG
            
        Returns:
            Dict: Answer with metadata
        """
        if self.model is None:
            success = self.load_model()
            if not success:
                return {
                    'answer': 'Модель не загружена. Пожалуйста, загрузите модель Saiga-2.',
                    'context_used': [],
                    'error': 'Model not loaded'
                }
        
        # Get context if not provided
        if context_chunks is None and use_rag:
            try:
                context_chunks = search_law(question, k=k)
                logger.info(f"Retrieved {len(context_chunks)} context chunks")
            except Exception as e:
                logger.error(f"Error retrieving context: {str(e)}")
                context_chunks = []
        elif context_chunks is None:
            context_chunks = []
        
        # Build prompt
        prompt = self._build_prompt(question, context_chunks)
        
        try:
            logger.info("Generating answer...")
            
            # Generate response
            response = self.model(
                prompt,
                max_new_tokens=800,
                temperature=0.1,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                stop=['</s>', '<s>', 'user', 'system']
            )
            
            # Clean up the response
            answer = response.strip()
            
            # Remove any remaining special tokens
            for token in ['<s>', '</s>', 'assistant', 'user', 'system']:
                answer = answer.replace(token, '')
            
            answer = answer.strip()
            
            logger.info("Answer generated successfully")
            
            return {
                'answer': answer,
                'context_used': context_chunks,
                'question': question,
                'model_used': 'Saiga-2 7B',
                'context_count': len(context_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                'answer': f'Произошла ошибка при генерации ответа: {str(e)}',
                'context_used': context_chunks,
                'error': str(e)
            }
    
    def chat(self, question: str) -> str:
        """
        Simple chat interface (convenience method).
        
        Args:
            question (str): User question
            
        Returns:
            str: Generated answer
        """
        result = self.generate_answer(question)
        return result['answer']


# Global lawyer instance
saiga_lawyer = SaigaLawyer()


def load_lawyer_model() -> bool:
    """
    Load the Saiga lawyer model (convenience function).
    
    Returns:
        bool: Success status
    """
    return saiga_lawyer.load_model()


def generate_answer(question: str, context_chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate an answer using the loaded model (convenience function).
    
    Args:
        question (str): Legal question
        context_chunks (List[Dict]): Optional pre-retrieved context
        
    Returns:
        Dict: Answer with metadata
    """
    return saiga_lawyer.generate_answer(question, context_chunks)


def quick_ask(question: str) -> str:
    """
    Quick question-answer interface (convenience function).
    
    Args:
        question (str): Legal question
        
    Returns:
        str: Answer
    """
    return saiga_lawyer.chat(question)


if __name__ == "__main__":
    # Test the lawyer
    print("Testing Saiga Lawyer...")
    
    # Try to load model
    model_loaded = load_lawyer_model()
    
    if model_loaded:
        print("Model loaded successfully!")
        
        # Test questions
        test_questions = [
            "Что такое гражданские права по законодательству Казахстана?",
            "Каковы основания возникновения права собственности?",
            "Как применяются международные договоры в гражданском праве?"
        ]
        
        for question in test_questions:
            print(f"\nВопрос: {question}")
            print("Ответ:")
            
            result = generate_answer(question)
            print(result['answer'])
            print(f"Использовано контекста: {result['context_count']}")
            print("-" * 50)
    else:
        print("Model not available. Testing with mock responses...")
        
        # Mock response for testing
        mock_answer = """
        К сожалению, модель Saiga-2 не загружена. 
        
        Для полноценной работы системы необходимо:
        1. Скачать модель Saiga-2 7B в формате GGUF
        2. Поместить её в папку models/saiga/saiga2.gguf
        3. Перезапустить систему
        
        Модель можно скачать с: https://huggingface.co/IlyaGusev/saiga2_7b_gguf
        """
        
        print("Mock answer:", mock_answer)