"""
MVP Streamlit приложение для AI Lawyer
Чат-интерфейс для тестирования RAG системы
"""

import streamlit as st
import time
from rag import build_index, search_law, get_rag_stats
from simple_lawyer import generate_answer as simple_generate_answer
from datetime import datetime

# Saiga-2 integration
try:
    from model_saiga import generate_answer_with_saiga, is_saiga_available, SaigaNotInstalledError, saiga_lawyer
    SAIGA_AVAILABLE = True
except ImportError as e:
    SAIGA_AVAILABLE = False
    print(f"Saiga-2 недоступна: {e}")

# Конфигурация страницы
st.set_page_config(
    page_title="AI Lawyer Chat",
    page_icon="⚖️",
    layout="wide"
)

# CSS для улучшения дизайна
st.markdown("""
<style>

body {
    font-family: 'Inter', sans-serif;
}

/* Контейнер чата */
.chat-wrapper {
    height: 70vh;
    overflow-y: auto;
    padding: 20px;
    background-color: #f7f9fc;
    border-radius: 12px;
    border: 1px solid #e6e9ef;
}

/* Сообщение пользователя */
.user-msg {
    background-color: #2d7cff;
    color: white;
    padding: 12px 16px;
    border-radius: 16px 16px 4px 16px;
    margin: 12px 0;
    max-width: 70%;
    margin-left: auto;
    font-size: 15px;
}

/* Сообщение ассистента */
.ai-msg {
    background-color: #ffffff;
    color: #1e1e1e;
    padding: 12px 16px;
    border-radius: 16px 16px 16px 4px;
    margin: 12px 0;
    max-width: 70%;
    margin-right: auto;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
    font-size: 15px;
}

/* Аватарки */
.avatar-ai {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    margin-right: 8px;
}
.avatar-user {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    margin-left: 8px;
}

/* Контейнер сообщения + аватар */
.msg-block {
    display: flex;
    align-items: flex-start;
    gap: 8px;
}

/* Нижняя панель ввода */
.input-box {
    position: fixed;
    bottom: 15px;
    left: 50%;
    transform: translateX(-50%);
    width: 75%;
    background: white;
    padding: 12px;
    border-radius: 12px;
    box-shadow: 0px 3px 14px rgba(0,0,0,0.1);
}

/* Кнопка отправки */
.send-btn button {
    background-color: #2d7cff !important;
    color: white !important;
    border-radius: 8px !important;
}

</style>
""", unsafe_allow_html=True)


def add_message(msg_type, content):
    """Добавление сообщения в чат"""
    st.session_state.messages.append({
        'type': msg_type,
        'content': content,
        'timestamp': datetime.now().strftime("%H:%M")
    })

def generate_ai_response(question, search_results):
    """Генерирует ответ ИИ на основе найденных документов"""
    
    # Пытаемся использовать Saiga-2, если доступна
    if st.session_state.get('saiga_initialized', False) and SAIGA_AVAILABLE:
        try:
            result = generate_answer_with_saiga(question, search_results)
            if result.get('success', False):
                return result['answer']
            else:
                st.warning(f"⚠️ Ошибка Saiga-2: {result.get('error', 'Unknown')}")
        except Exception as e:
            st.warning(f"⚠️ Ошибка Saiga-2: {str(e)}")
    
    # Fallback на простого ассистента
    result = simple_generate_answer(question, search_results)
    return result['answer']

def display_search_results_summary(results):
    """Отображает краткую сводку результатов поиска"""
    if not results:
        return "Релевантные документы не найдены"
    
    summary = f"Найдено {len(results)} релевантных фрагментов:\n"
    
    for i, result in enumerate(results[:3], 1):
        source = result['metadata']['source']
        score = result['score']
        summary += f"• {source} (релевантность: {score:.2f})\n"
    
    return summary

def main():
    # Инициализация
    init_session_state()
    
    # Заголовок
    st.markdown('<h1 class="main-header">⚖️ AI Lawyer - Юридический Чат-Ассистент</h1>', unsafe_allow_html=True)
    
    # Боковая панель со статистикой
    with st.sidebar:
        st.header("🔧 Статус системы")
        
        # Инициализация RAG системы
        if not st.session_state.rag_initialized:
            with st.spinner("Инициализация RAG системы..."):
                try:
                    success = build_index()
                    st.session_state.rag_initialized = success
                    st.session_state.rag_status = "✅ Готова" if success else "❌ Ошибка"
                except Exception as e:
                    st.session_state.rag_status = f"❌ Ошибка: {str(e)}"
        
        st.markdown(f'<div class="status-indicator status-{"success" if st.session_state.rag_initialized else "info"}">🔍 RAG: {st.session_state.rag_status}</div>', unsafe_allow_html=True)
        
        # Проверка Saiga-2
        if SAIGA_AVAILABLE and not st.session_state.saiga_initialized:
            if is_saiga_available():
                st.session_state.saiga_status = "✅ Доступна"
                st.session_state.saiga_initialized = True
            else:
                st.session_state.saiga_status = "❌ Модель не найдена"
        elif not SAIGA_AVAILABLE:
            st.session_state.saiga_status = "❌ Библиотека недоступна"
        
        # Кнопка загрузки модели
        if st.session_state.get('saiga_initialized', False):
            st.markdown(f'<div class="status-indicator status-success">🤖 Saiga-2: {st.session_state.saiga_status}</div>', unsafe_allow_html=True)
            if st.button("🔄 Перезагрузить модель", use_container_width=True):
                st.session_state.saiga_initialized = False
                st.rerun()
        else:
            st.markdown(f'<div class="status-indicator status-info">🤖 Saiga-2: {st.session_state.saiga_status}</div>', unsafe_allow_html=True)
            
            if st.button("📥 Скачать законы КZ", use_container_width=True):
                with st.spinner("Скачивание законов..."):
                    import subprocess
                    result = subprocess.run(['python', 'download_kazakh_laws.py'], 
                                          capture_output=True, text=True, cwd='.')
                    if result.returncode == 0:
                        st.success("✅ Законы скачаны!")
                        st.session_state.rag_initialized = False  # Пересоздать индекс
                        st.rerun()
                    else:
                        st.error(f"❌ Ошибка: {result.stderr}")
        
        # Статистика
        if st.session_state.rag_initialized:
            try:
                stats = get_rag_stats()
                st.write("**📊 Статистика базы:**")
                st.write(f"• Документов: {len(stats.get('sources', []))}")
                st.write(f"• Фрагментов: {stats.get('total_chunks', 0)}")
                st.write(f"• Символов: {stats.get('total_characters', 0):,}")
                
                st.write("**📄 Источники:**")
                for source in stats.get('sources', []):
                    st.write(f"• {source}")
            except:
                st.write("Статистика недоступна")
        
        st.markdown("---")
        st.write("**💡 Примеры вопросов:**")
        example_questions = [
            "Что такое право собственности?",
            "Что такое обязательства?", 
            "Как защищаются гражданские права?",
            "Что такое договор?",
            "Свобода договора"
        ]
        
        for question in example_questions:
            if st.button(f"💬 {question}", key=f"example_{question}", use_container_width=True):
                # Добавляем вопрос в чат
                add_message('user', question)
                st.rerun()
    
    # Основной чат
    col1, col2 = st.columns([1, 5])
    
    with col2:
        # Отображение чата
        display_chat()
        
        # Поле ввода и кнопка отправки
        st.markdown("### 💬 Задайте ваш вопрос:")
        
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                label="question",
                placeholder="Введите ваш правовой вопрос здесь...",
                height=100,
                label_visibility="collapsed"
            )
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col2:
                submit_button = st.form_submit_button("📤 Отправить", type="primary", use_container_width=True)
            
            with col3:
                clear_button = st.form_submit_button("🗑️ Очистить", use_container_width=True)
        
        # Обработка отправки сообщения
        if submit_button and user_input.strip():
            if not st.session_state.rag_initialized:
                st.error("❌ RAG система не инициализирована")
            else:
                # Добавляем сообщение пользователя
                add_message('user', user_input)
                
                # Ищем релевантные документы
                with st.spinner("🔍 Поиск релевантной информации..."):
                    search_results = search_law(user_input, k=5)
                
                # Добавляем результаты поиска
                search_summary = display_search_results_summary(search_results)
                add_message('search', search_summary)
                
                # Генерируем ответ ИИ
                with st.spinner("🤖 Генерация ответа..."):
                    ai_response = generate_ai_response(user_input, search_results)
                
                # Добавляем ответ ИИ
                add_message('ai', ai_response)
                
                # Обновляем интерфейс
                st.rerun()
        
        # Обработка очистки чата
        if clear_button:
            st.session_state.messages = []
            # Добавляем приветственное сообщение
            add_message('ai', '👋 Чат очищен! Задайте новый вопрос.')
            st.rerun()
    
    # Информация внизу
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <strong>⚠️ Важно:</strong> Данная система предоставляет справочную информацию и не заменяет профессиональную юридическую консультацию.
        <br>Для решения конкретных правовых вопросов обратитесь к квалифицированному юристу.
    </div>
    """, unsafe_allow_html=True)
def render_message(msg_type, content):
    """Красивая визуализация каждого сообщения"""
    
    if msg_type == "user":
        st.markdown(f"""
        <div class="msg-block" style="justify-content: flex-end;">
            <div class="user-msg">{content}</div>
            <img src="https://i.imgur.com/9Xn0XKp.png" class="avatar-user">
        </div>
        """, unsafe_allow_html=True)

    elif msg_type == "ai":
        st.markdown(f"""
        <div class="msg-block">
            <img src="https://i.imgur.com/1o1h8Gf.png" class="avatar-ai">
            <div class="ai-msg">{content}</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()