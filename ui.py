"""
Streamlit UI for the AI Lawyer project.
Provides a user-friendly web interface for legal consultation.
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List
import logging

# Configure page
st.set_page_config(
    page_title="AI Lawyer - Юридический ассистент",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f4e79;
    }
    
    .answer-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #0066cc;
    }
    
    .context-box {
        background-color: #fff2cc;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #ff9900;
        font-size: 0.9rem;
    }
    
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ff4444;
        color: #cc0000;
    }
    
    .status-good {
        color: #00aa00;
        font-weight: bold;
    }
    
    .status-bad {
        color: #cc0000;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ff8800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def get_system_status():
    """Get system status from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


@st.cache_data(ttl=300)
def get_system_stats():
    """Get detailed system statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None


def ask_question(question: str, use_context: bool = True, max_context: int = 3) -> Dict[str, Any]:
    """Send question to API and get response."""
    try:
        payload = {
            "question": question,
            "use_context": use_context,
            "max_context": max_context
        }
        
        with st.spinner("Обработка вашего вопроса..."):
            response = requests.post(
                f"{API_BASE_URL}/ask",
                json=payload,
                timeout=60
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = "Неизвестная ошибка"
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", error_detail)
            except:
                pass
            
            return {
                "error": f"Ошибка API (HTTP {response.status_code}): {error_detail}",
                "answer": "",
                "context_chunks": []
            }
            
    except requests.exceptions.Timeout:
        return {
            "error": "Превышено время ожидания ответа от сервера",
            "answer": "",
            "context_chunks": []
        }
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Ошибка соединения: {str(e)}",
            "answer": "",
            "context_chunks": []
        }


def search_documents(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search legal documents."""
    try:
        params = {"query": query, "limit": limit}
        response = requests.post(f"{API_BASE_URL}/search", params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
            
    except requests.exceptions.RequestException:
        return []


def display_status_indicator(status_data: Dict[str, Any]):
    """Display system status indicator."""
    if status_data.get("status") == "error":
        st.markdown(
            f'<div class="status-bad">❌ Сервер недоступен: {status_data.get("message", "")}</div>',
            unsafe_allow_html=True
        )
    elif status_data.get("status") == "healthy":
        st.markdown('<div class="status-good">✅ Система работает</div>', unsafe_allow_html=True)
    elif status_data.get("status") == "degraded":
        st.markdown('<div class="status-warning">⚠️ Ограниченная функциональность</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">❓ Неизвестный статус</div>', unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">⚖️ AI Lawyer</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">Юридический ИИ-ассистент на базе Saiga-2 и RAG</div>', unsafe_allow_html=True)
    
    # Sidebar with system info
    with st.sidebar:
        st.header("📊 Статус системы")
        
        status_data = get_system_status()
        display_status_indicator(status_data)
        
        if status_data.get("status") != "error":
            st.write("**Модель ИИ:**", 
                    "✅ Загружена" if status_data.get("model_loaded") else "❌ Не загружена")
            st.write("**RAG система:**", 
                    "✅ Инициализирована" if status_data.get("rag_initialized") else "❌ Не инициализирована")
        
        # Detailed stats
        stats = get_system_stats()
        if stats:
            st.header("📈 Статистика")
            
            if "rag_status" in stats and "details" in stats["rag_status"]:
                rag_details = stats["rag_status"]["details"]
                st.write(f"**Документов:** {len(rag_details.get('sources', []))}")
                st.write(f"**Фрагментов:** {rag_details.get('total_chunks', 0)}")
                st.write(f"**Символов:** {rag_details.get('total_characters', 0):,}")
        
        # Settings
        st.header("⚙️ Настройки")
        use_context = st.checkbox("Использовать контекст RAG", value=True, 
                                 help="Включить поиск по базе правовых документов")
        max_context_chunks = st.slider("Максимум фрагментов контекста", 1, 10, 3,
                                      help="Количество релевантных фрагментов для анализа")
        show_context = st.checkbox("Показывать найденные фрагменты", value=True)
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Задайте юридический вопрос")
        
        # Question input
        question = st.text_area(
            "Введите ваш вопрос:",
            placeholder="Например: Что такое гражданские права по законодательству Казахстана?",
            height=100,
            help="Формулируйте вопросы четко и конкретно для получения наиболее точных ответов"
        )
        
        # Action buttons
        col_ask, col_search = st.columns([1, 1])
        
        with col_ask:
            ask_button = st.button("🤖 Получить консультацию", type="primary", use_container_width=True)
        
        with col_search:
            search_button = st.button("🔍 Поиск в документах", use_container_width=True)
    
    with col2:
        st.header("📋 Примеры вопросов")
        
        example_questions = [
            "Что такое гражданские права?",
            "Основания возникновения права собственности",
            "Международные договоры в гражданском праве",
            "Обязательства по договору",
            "Защита нарушенных прав"
        ]
        
        for i, example in enumerate(example_questions):
            if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
                st.rerun()
    
    # Process question
    if ask_button and question.strip():
        if status_data.get("status") == "error":
            st.error("❌ Сервер недоступен. Пожалуйста, проверьте подключение к API.")
        else:
            # Ask question
            response = ask_question(question, use_context, max_context_chunks)
            
            # Display question
            st.markdown(f'<div class="question-box"><strong>❓ Ваш вопрос:</strong><br>{question}</div>', 
                       unsafe_allow_html=True)
            
            # Display error if any
            if response.get("error"):
                st.markdown(f'<div class="error-box"><strong>⚠️ Ошибка:</strong><br>{response["error"]}</div>',
                           unsafe_allow_html=True)
            
            # Display answer
            if response.get("answer"):
                processing_time = response.get("processing_time", 0)
                st.markdown(
                    f'<div class="answer-box"><strong>🤖 Ответ ИИ-юриста:</strong><br>{response["answer"]}</div>',
                    unsafe_allow_html=True
                )
                
                st.caption(f"⏱️ Время обработки: {processing_time:.2f} сек | "
                          f"🧠 Модель: {response.get('model_info', {}).get('model_name', 'Неизвестно')}")
            
            # Display context if available and enabled
            if show_context and response.get("context_chunks"):
                st.header("📚 Найденные правовые документы")
                
                for i, chunk in enumerate(response["context_chunks"], 1):
                    with st.expander(f"📄 Документ {i}: {chunk['source']} (релевантность: {chunk['relevance_score']:.3f})"):
                        st.markdown(f'<div class="context-box">{chunk["content"]}</div>', 
                                   unsafe_allow_html=True)
    
    # Process search
    elif search_button and question.strip():
        if status_data.get("status") == "error":
            st.error("❌ Сервер недоступен. Пожалуйста, проверьте подключение к API.")
        else:
            # Search documents
            with st.spinner("Поиск в правовых документах..."):
                search_results = search_documents(question, limit=max_context_chunks)
            
            st.header(f"🔍 Результаты поиска: '{question}'")
            
            if search_results:
                for i, result in enumerate(search_results, 1):
                    with st.expander(f"📄 Результат {i}: {result['source']} (релевантность: {result['relevance_score']:.3f})"):
                        st.markdown(f'<div class="context-box">{result["content"]}</div>', 
                                   unsafe_allow_html=True)
            else:
                st.info("🤷‍♂️ Релевантных документов не найдено. Попробуйте изменить запрос.")
    
    # Footer with disclaimers
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <strong>⚠️ Важно:</strong> Данная система предоставляет справочную информацию и не заменяет профессиональную юридическую консультацию.
        Для решения конкретных правовых вопросов обратитесь к квалифицированному юристу.
    </div>
    """, unsafe_allow_html=True)
    
    # Store question in session state for examples
    if 'selected_example' in st.session_state:
        question = st.session_state.selected_example
        del st.session_state.selected_example


if __name__ == "__main__":
    main()