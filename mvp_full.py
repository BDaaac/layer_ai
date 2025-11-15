"""
Полный MVP Streamlit приложения для AI Lawyer с интеграцией Saiga-2
"""

import os
import streamlit as st
import time
import subprocess
from datetime import datetime

# Основные импорты
from rag import build_index, search_law, get_rag_stats, initialize_rag

# Simple lawyer fallback
try:
    from simple_lawyer import generate_answer as simple_generate_answer
    SIMPLE_LAWYER_AVAILABLE = True
except ImportError:
    SIMPLE_LAWYER_AVAILABLE = False

# Saiga-2 integration
try:
    from model_saiga import generate_answer_with_saiga, is_saiga_available, SaigaNotInstalledError, initialize_saiga
    SAIGA_AVAILABLE = True
except ImportError as e:
    SAIGA_AVAILABLE = False
    print(f"Saiga-2 недоступна: {e}")

# Конфигурация страницы
st.set_page_config(
    page_title="AI Lawyer - Full MVP",
    page_icon="⚖️",
    layout="wide"
)

# CSS для улучшенного дизайна
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1e3a5f;
        margin-bottom: 2rem;
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 1rem;
    }
    
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 2px solid #e6e6e6;
        border-radius: 15px;
        background: linear-gradient(to bottom, #f8f9fa, #ffffff);
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #007bff, #0056b3);
        color: white;
        padding: 12px 18px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        margin-left: 25%;
        text-align: right;
        box-shadow: 0 2px 4px rgba(0, 123, 255, 0.3);
        animation: slideInRight 0.3s ease;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #e9ecef, #f8f9fa);
        color: #333;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        margin-right: 25%;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        animation: slideInLeft 0.3s ease;
    }
    
    .saiga-message {
        background: linear-gradient(135deg, #fff3cd, #fef6d1);
        color: #856404;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        margin-right: 20%;
        border-left: 4px solid #ffc107;
        box-shadow: 0 2px 4px rgba(255, 193, 7, 0.3);
        animation: slideInLeft 0.3s ease;
    }
    
    .search-results {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border: 2px solid #17a2b8;
        border-radius: 12px;
        padding: 12px;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    
    .timestamp {
        font-size: 0.75rem;
        color: #6c757d;
        text-align: center;
        margin: 5px 0;
        font-style: italic;
    }
    
    .status-badge {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 5px 0;
        text-align: center;
    }
    
    .status-success {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        border: 2px solid #28a745;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        color: #856404;
        border: 2px solid #ffc107;
    }
    
    .status-error {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        border: 2px solid #dc3545;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .download-section {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border: 2px solid #2196f3;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .model-info {
        background: linear-gradient(135deg, #f3e5f5, #e1bee7);
        border: 2px solid #9c27b0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Инициализация состояния сессии"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            'type': 'ai',
            'content': '👋 Добро пожаловать в AI Lawyer! Я ваш юридический ИИ-ассистент, специализирующийся на законодательстве Республики Казахстан. Задайте мне любой правовой вопрос!',
            'timestamp': datetime.now().strftime("%H:%M")
        })
    
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
        st.session_state.rag_status = "Проверка..."
    
    if 'saiga_initialized' not in st.session_state:
        st.session_state.saiga_initialized = False
        st.session_state.saiga_status = "Проверка..."


def display_chat():
    """Отображение истории чата с улучшенным дизайном"""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        timestamp_html = f'<div class="timestamp">🕐 {message["timestamp"]}</div>'
        
        if message['type'] == 'user':
            st.markdown(f"""
            <div class="user-message">
                👤 {message['content']}
            </div>
            {timestamp_html}
            """, unsafe_allow_html=True)
        
        elif message['type'] == 'ai':
            st.markdown(f"""
            <div class="ai-message">
                🤖 {message['content']}
            </div>
            {timestamp_html}
            """, unsafe_allow_html=True)
        
        elif message['type'] == 'saiga':
            st.markdown(f"""
            <div class="saiga-message">
                🧠 <strong>Saiga-2:</strong> {message['content']}
            </div>
            {timestamp_html}
            """, unsafe_allow_html=True)
        
        elif message['type'] == 'search':
            st.markdown(f"""
            <div class="search-results">
                🔍 <strong>Результаты поиска:</strong><br>
                {message['content']}
            </div>
            {timestamp_html}
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def add_message(msg_type, content):
    """Добавление сообщения в чат"""
    st.session_state.messages.append({
        'type': msg_type,
        'content': content,
        'timestamp': datetime.now().strftime("%H:%M")
    })


def generate_ai_response(question, search_results):
    """Генерирует ответ используя доступные модели"""
    
    # Пытаемся использовать Saiga-2 если доступна
    if st.session_state.get('saiga_initialized', False) and SAIGA_AVAILABLE:
        try:
            with st.spinner("🧠 Генерация ответа с помощью Saiga-2..."):
                result = generate_answer_with_saiga(question, search_results)
                if result.get('success', False):
                    return result['answer'], 'saiga'
                else:
                    st.warning(f"⚠️ Ошибка Saiga-2: {result.get('error', 'Unknown')}")
        except Exception as e:
            st.warning(f"⚠️ Ошибка Saiga-2: {str(e)}")
    
    # Fallback на простого ассистента
    if SIMPLE_LAWYER_AVAILABLE:
        with st.spinner("🤖 Генерация ответа..."):
            result = simple_generate_answer(question, search_results)
            return result['answer'], 'ai'
    else:
        return "❌ Ни один из ИИ-ассистентов не доступен.", 'ai'


def display_search_results_summary(results):
    """Отображает краткую сводку результатов поиска"""
    if not results:
        return "❌ Релевантные документы не найдены"
    
    summary = f"✅ Найдено {len(results)} релевантных фрагментов:\n\n"
    
    for i, result in enumerate(results[:3], 1):
        source = result['metadata']['source']
        score = result['score']
        summary += f"📄 **{i}.** {source} (релевантность: {score:.2f})\n"
    
    if len(results) > 3:
        summary += f"\n➕ И еще {len(results) - 3} документов..."
    
    return summary


def main():
    # Инициализация
    init_session_state()
    
    # Инициализация RAG системы
    if not st.session_state.rag_initialized:
        with st.spinner("Инициализация RAG системы..."):
            try:
                success = initialize_rag()
                if success:
                    st.session_state.rag_initialized = True
                    st.session_state.rag_stats = get_rag_stats()
                    st.success("RAG система успешно инициализирована!")
                else:
                    st.error("Ошибка при инициализации RAG системы")
            except Exception as e:
                st.error(f"Ошибка инициализации RAG: {e}")
    
    # Заголовок
    st.markdown('<h1 class="main-header">⚖️ AI Lawyer - Полный MVP</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Юридический ИИ-ассистент с интеграцией Saiga-2 и RAG системой</p>', unsafe_allow_html=True)
    
    # Создание колонок
    sidebar, main_col = st.columns([1, 3])
    
    with sidebar:
        st.header("🔧 Панель управления")
        
        # RAG система
        st.subheader("📚 RAG система")
        if not st.session_state.rag_initialized:
            if st.button("🚀 Инициализировать RAG", use_container_width=True):
                with st.spinner("Инициализация RAG системы..."):
                    try:
                        success = initialize_rag()
                        st.session_state.rag_initialized = success
                        st.session_state.rag_status = "✅ Готова" if success else "❌ Ошибка"
                        st.rerun()
                    except Exception as e:
                        st.session_state.rag_status = f"❌ Ошибка: {str(e)}"
                        st.error(st.session_state.rag_status)
        
        if st.session_state.rag_initialized:
            st.markdown('<div class="status-badge status-success">📚 RAG система готова</div>', unsafe_allow_html=True)
            
            try:
                stats = get_rag_stats()
                st.write("**📊 Статистика базы:**")
                st.write(f"• 📄 Документов: {len(stats.get('sources', []))}")
                st.write(f"• 🧩 Фрагментов: {stats.get('total_chunks', 0)}")
                st.write(f"• 📝 Символов: {stats.get('total_characters', 0):,}")
            except:
                st.write("Статистика недоступна")
        else:
            st.markdown('<div class="status-badge status-error">📚 RAG не готова</div>', unsafe_allow_html=True)
        
        # Saiga-2 модель
        st.subheader("🧠 Модель Saiga-2")
        
        if SAIGA_AVAILABLE:
            if not st.session_state.saiga_initialized:
                if st.button("🔍 Инициализировать Saiga-2", use_container_width=True):
                    with st.spinner("Инициализация Saiga-2..."):
                        try:
                            if initialize_saiga():
                                st.session_state.saiga_initialized = True
                                st.session_state.saiga_status = "✅ Готова"
                                st.success("Saiga-2 успешно инициализирована!")
                            else:
                                st.session_state.saiga_status = "❌ Ошибка инициализации"
                                st.error("Не удалось инициализировать Saiga-2")
                        except Exception as e:
                            st.session_state.saiga_status = f"❌ Ошибка: {str(e)}"
                            st.error(f"Ошибка: {str(e)}")
                        st.rerun()
            
            if st.session_state.saiga_initialized:
                st.markdown('<div class="status-badge status-success">🧠 Saiga-2 готова</div>', unsafe_allow_html=True)
                
                # Проверяем тип модели
                try:
                    from model_saiga import LLAMA_CPP_AVAILABLE
                    if LLAMA_CPP_AVAILABLE:
                        st.write("**📋 Статус модели:**")
                        st.write("• 🔧 Полная модель Saiga-2 загружена")
                        if os.path.exists("models/saiga/saiga2.gguf"):
                            file_size = os.path.getsize("models/saiga/saiga2.gguf") / (1024*1024)
                            st.write(f"• 💾 Размер файла: {file_size:.1f} МБ")
                    else:
                        st.write("**📋 Режим работы:**")
                        st.write("• 🔄 Fallback режим (без LLM)")
                        st.write("• 📚 Ответы на основе поиска по документам")
                        st.info("💡 Для полной функциональности установите llama-cpp-python")
                except:
                    st.write("• ✅ Модель готова к работе")
            else:
                st.markdown('<div class="status-badge status-error">🧠 Saiga-2 недоступна</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="model-info">
                <strong>💡 Для использования Saiga-2:</strong><br>
                1. Установите: <code>pip install llama-cpp-python</code><br>
                2. Скачайте модель в папку <code>models/saiga/</code><br>
                3. Нажмите "Проверить Saiga-2"
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-warning">🧠 llama_cpp недоступен</div>', unsafe_allow_html=True)
            st.info("Установите llama-cpp-python для использования Saiga-2")
        
        # Скачивание законов
        st.subheader("📥 Обновление данных")
        
        st.markdown("""
        <div class="download-section">
        <strong>🏛️ Автоматическое скачивание законов РК:</strong><br>
        • Конституция РК<br>
        • Гражданский кодекс<br>
        • Закон о защите прав потребителей<br>
        • Административный кодекс<br>
        • Трудовой кодекс
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📥 Скачать законы КZ", use_container_width=True):
            with st.spinner("Скачивание законов Казахстана..."):
                try:
                    result = subprocess.run(['python', 'download_kazakh_laws.py'], 
                                          capture_output=True, text=True, cwd='.')
                    if result.returncode == 0:
                        st.success("✅ Законы скачаны успешно!")
                        st.session_state.rag_initialized = False
                        st.info("🔄 Перезапустите RAG для индексации новых документов")
                    else:
                        st.error(f"❌ Ошибка скачивания: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        # Примеры вопросов
        st.subheader("💡 Примеры вопросов")
        example_questions = [
            "Что такое право собственности?",
            "Права потребителя при покупке товара",
            "Как заключить договор?",
            "Защита трудовых прав",
            "Административная ответственность",
            "Гражданские права и свободы"
        ]
        
        for question in example_questions:
            if st.button(f"💬 {question}", key=f"example_{hash(question)}", use_container_width=True):
                add_message('user', question)
                st.rerun()
    
    with main_col:
        # Отображение чата
        display_chat()
        
        # Поле ввода
        st.markdown("### 💬 Задайте ваш юридический вопрос:")
        
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                label="question",
                placeholder="Например: Какие права имеет собственник недвижимости в Казахстане?",
                height=80,
                label_visibility="collapsed"
            )
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col2:
                submit_button = st.form_submit_button("📤 Отправить", type="primary", use_container_width=True)
            
            with col3:
                search_only = st.form_submit_button("🔍 Только поиск", use_container_width=True)
            
            with col4:
                clear_button = st.form_submit_button("🗑️ Очистить", use_container_width=True)
        
        # Обработка действий
        if (submit_button or search_only) and user_input.strip():
            if not st.session_state.rag_initialized:
                st.error("❌ RAG система не инициализирована. Используйте боковую панель.")
            else:
                # Добавляем вопрос пользователя
                add_message('user', user_input)
                
                # Поиск документов
                with st.spinner("🔍 Поиск релевантной информации..."):
                    search_results = search_law(user_input, k=5)
                
                # Показываем результаты поиска
                search_summary = display_search_results_summary(search_results)
                add_message('search', search_summary)
                
                # Генерируем ответ если не только поиск
                if submit_button:
                    answer, msg_type = generate_ai_response(user_input, search_results)
                    add_message(msg_type, answer)
                
                st.rerun()
        
        # Очистка чата
        if clear_button:
            st.session_state.messages = []
            add_message('ai', '👋 Чат очищен! Готов ответить на новые вопросы.')
            st.rerun()
    
    # Подвал с информацией
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔧 Технологии:**
        - 🧠 Saiga-2 7B (GGUF)
        - 📚 RAG + FAISS
        - 🌐 Streamlit UI
        - 🇰🇿 Законы РК
        """)
    
    with col2:
        st.markdown("""
        **📊 Статус системы:**
        - RAG: """ + ("✅" if st.session_state.rag_initialized else "❌") + """
        - Saiga-2: """ + ("✅" if st.session_state.get('saiga_initialized') else "❌") + """
        - Данные: Готовы
        """)
    
    with col3:
        st.markdown("""
        **⚠️ Важно:**
        Система предоставляет справочную 
        информацию. Для решения конкретных 
        правовых вопросов обратитесь к 
        квалифицированному юристу.
        """)


if __name__ == "__main__":
    main()