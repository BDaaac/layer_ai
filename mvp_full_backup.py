"""
Полный MVP Streamlit приложения для AI Lawyer с интеграцией Saiga-2
Улучшенный дизайн
"""

import streamlit as st
import time
import subprocess
from datetime import datetime

# Основные импорты
from rag import build_index, search_law, get_rag_stats

# Simple lawyer fallback
try:
    from simple_lawyer import generate_answer as simple_generate_answer
    SIMPLE_LAWYER_AVAILABLE = True
except ImportError:
    SIMPLE_LAWYER_AVAILABLE = False

# Saiga-2 integration
try:
    from model_saiga import generate_answer_with_saiga, is_saiga_available, SaigaNotInstalledError, saiga_lawyer
    SAIGA_AVAILABLE = True
except ImportError as e:
    SAIGA_AVAILABLE = False
    print(f"Saiga-2 недоступна: {e}")

# Конфигурация страницы
st.set_page_config(
    page_title="AI Lawyer - Юридический ИИ-ассистент",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS для кардинально улучшенного дизайна
st.markdown("""
<style>
    /* Основные стили */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        text-align: center;
        color: #4B296B;
        margin-bottom: 2rem;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

    /* Контейнер чата */
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 2rem;
        border-radius: 25px;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    /* Сообщения пользователя */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 25px 25px 5px 25px;
        margin: 1rem 0;
        margin-left: 15%;
        text-align: right;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .user-message::before {
        position: absolute;
        right: -45px;
        top: 50%;
        transform: translateY(-50%);
        background: #667eea;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Сообщения ИИ */
    .ai-message {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        color: #2d3748;
        padding: 1.2rem 1.5rem;
        border-radius: 25px 25px 25px 5px;
        margin: 1rem 0;
        margin-right: 15%;
        border-left: 5px solid #48bb78;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        animation: slideInLeft 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    .ai-message::before {
        position: absolute;
        left: -45px;
        top: 50%;
        transform: translateY(-50%);
        background: #48bb78;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);
    }
    
    /* Сообщения Saiga */
    .saiga-message {
        background: linear-gradient(135deg, #fff9db 0%, #fff3bf 100%);
        color: #5f3f16;
        padding: 1.2rem 1.5rem;
        border-radius: 25px 25px 25px 5px;
        margin: 1rem 0;
        margin-right: 10%;
        border-left: 5px solid #f59f00;
        box-shadow: 0 8px 25px rgba(245, 159, 0, 0.15);
        animation: slideInLeft 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
        border: 1px solid rgba(245, 159, 0, 0.2);
    }
    
    .saiga-message::before {
        position: absolute;
        left: -45px;
        top: 50%;
        transform: translateY(-50%);
        background: #f59f00;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(245, 159, 0, 0.3);
    }
    
    /* Результаты поиска */
    .search-results {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 2px solid #2196f3;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        font-size: 0.95rem;
        box-shadow: 0 8px 25px rgba(33, 150, 243, 0.15);
        animation: fadeIn 0.5s ease;
        position: relative;
        overflow: hidden;
    }
    
    .search-results::before {
        position: absolute;
        top: 15px;
        right: 15px;
        font-size: 1.5rem;
        opacity: 0.3;
    }
    
    .timestamp {
        font-size: 0.8rem;
        color: #718096;
        text-align: center;
        margin: 0.5rem 0;
        font-style: italic;
        opacity: 0.8;
    }
    
    /* Статус баджи */
    .status-badge {
        display: inline-block;
        padding: 0.8rem 1.2rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .status-success {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
    }
    
    .status-error {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
    }
    
    .status-info {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
    }
    
    /* Анимации */
    @keyframes slideInRight {
        from { 
            transform: translateX(50px) scale(0.95); 
            opacity: 0; 
        }
        to { 
            transform: translateX(0) scale(1); 
            opacity: 1; 
        }
    }
    
    @keyframes slideInLeft {
        from { 
            transform: translateX(-50px) scale(0.95); 
            opacity: 0; 
        }
        to { 
            transform: translateX(0) scale(1); 
            opacity: 1; 
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Боковая панель */
    .css-1d391kg {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%) !important;
    }
    
    .sidebar .sidebar-content {
        background: transparent !important;
    }
    
    /* Кнопки */
    .stButton button {
        border-radius: 15px !important;
        border: none !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Поля ввода */
    .stTextArea textarea {
        border-radius: 20px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Карточки */
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Прогресс бары */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Уведомления */
    .stAlert {
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Инициализация состояния сессии"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        welcome_msg = """
         **Добро пожаловать в AI Lawyer - ваш персональный юридический ИИ-ассистент!**
        
         **Что я умею:**
        •  Анализировать законодательство РК
        •  Находить релевантные правовые нормы
        •  Объяснять сложные юридические понятия
        •  Помогать с правовыми вопросами
        
         **Задайте ваш вопрос ниже и получите профессиональный ответ!**
        """
        st.session_state.messages.append({
            'type': 'ai',
            'content': welcome_msg,
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
                {message['content']}
            </div>
            {timestamp_html}
            """, unsafe_allow_html=True)
        
        elif message['type'] == 'ai':
            st.markdown(f"""
            <div class="ai-message">
                {message['content']}
            </div>
            {timestamp_html}
            """, unsafe_allow_html=True)
        
        elif message['type'] == 'saiga':
            st.markdown(f"""
            <div class="saiga-message">
                <strong>🧠 Saiga-2:</strong><br>
                {message['content']}
            </div>
            {timestamp_html}
            """, unsafe_allow_html=True)
        
        elif message['type'] == 'search':
            st.markdown(f"""
            <div class="search-results">
                <strong>🔍 Найдено в законодательстве:</strong><br><br>
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
        with st.spinner(" Генерация ответа..."):
            result = simple_generate_answer(question, search_results)
            return result['answer'], 'ai'
    else:
        return "❌ Ни один из ИИ-ассистентов не доступен.", 'ai'


def display_search_results_summary(results):
    """Отображает краткую сводку результатов поиска"""
    if not results:
        return "❌ Релевантные документы не найдены в текущей базе данных."
    
    summary = f"🎯 **Найдено {len(results)} релевантных документов:**\n\n"
    
    for i, result in enumerate(results[:4], 1):
        source = result['metadata']['source']
        score = result['score']
        # Создаем визуальный индикатор релевантности
        relevance_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        summary += f"**{i}. {source}**\n"
        summary += f"   Релевантность: {score:.2f} {relevance_bar}\n\n"
    
    if len(results) > 4:
        summary += f"*... и еще {len(results) - 4} документов*"
    
    return summary


def create_feature_card(title, description, icon, color):
    """Создает красивую карточку функции"""
    return f"""
    <div class="feature-card" style="border-left: 5px solid {color};">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <h4 style="margin: 0 0 0.5rem 0; color: #2d3748;">{title}</h4>
        <p style="margin: 0; color: #718096; font-size: 0.9rem;">{description}</p>
    </div>
    """


def main():
    # Инициализация
    init_session_state()
    
    # Главный заголовок
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 3rem;">⚖️ AI Lawyer</h1>
            <p style="margin: 0; font-size: 1.2rem; opacity: 0.9;">Умный юридический ассистент с искусственным интеллектом</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Создание колонок
    sidebar, main_col = st.columns([1, 2])
    
    with sidebar:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 20px; margin-bottom: 2rem;">
            <h3 style="color: white; margin-bottom: 1rem;">🔧 Панель управления</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # RAG система
        st.markdown("### 📚 База знаний")
        if not st.session_state.rag_initialized:
            if st.button("🚀 Инициализировать систему", use_container_width=True, type="primary"):
                with st.spinner("Загрузка базы знаний..."):
                    try:
                        success = build_index()
                        st.session_state.rag_initialized = success
                        st.session_state.rag_status = "✅ Готова" if success else "❌ Ошибка"
                        if success:
                            st.success("✅ База знаний загружена!")
                        st.rerun()
                    except Exception as e:
                        st.session_state.rag_status = f"❌ Ошибка: {str(e)}"
                        st.error(st.session_state.rag_status)
        
        if st.session_state.rag_initialized:
            st.markdown('<div class="status-badge status-success">📚 База знаний активна</div>', unsafe_allow_html=True)
            
            try:
                stats = get_rag_stats()
                st.markdown(f"""
                **📊 Статистика:**
                ```
                📄 Документов: {len(stats.get('sources', []))}
                🧩 Фрагментов: {stats.get('total_chunks', 0)}
                📝 Символов: {stats.get('total_characters', 0):,}
                ```
                """)
            except:
                st.info("📊 Статистика временно недоступна")
        else:
            st.markdown('<div class="status-badge status-error">📚 Требуется инициализация</div>', unsafe_allow_html=True)
        
        # Saiga-2 модель
        st.markdown("### 🧠 ИИ Модель")
        
        if SAIGA_AVAILABLE:
            if not st.session_state.saiga_initialized:
                if st.button("🔍 Проверить Saiga-2", use_container_width=True):
                    with st.spinner("Проверка модели..."):
                        if is_saiga_available():
                            st.session_state.saiga_initialized = True
                            st.session_state.saiga_status = "✅ Готова"
                            st.success("🎉 Saiga-2 доступна!")
                        else:
                            st.session_state.saiga_status = "❌ Модель не найдена"
                            st.error("Модель Saiga-2 не найдена")
                        st.rerun()
            
            if st.session_state.saiga_initialized:
                st.markdown('<div class="status-badge status-success">🧠 Saiga-2 активна</div>', unsafe_allow_html=True)
                
                model_info = saiga_lawyer.get_model_info()
                st.markdown(f"""
                **📋 Информация:**
                ```
                💾 Размер: {model_info['model_size_mb']:.1f} МБ
                🔧 Статус: Загружена
                ⚡ Производительность: Оптимальная
                ```
                """)
            else:
                st.markdown('<div class="status-badge status-warning">🧠 Saiga-2 недоступна</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-info">🧠 Используется базовый ИИ</div>', unsafe_allow_html=True)
        
        # Обновление данных
        st.markdown("### 📥 Данные")
        if st.button("🔄 Обновить законы", use_container_width=True):
            with st.spinner("Скачивание актуальных данных..."):
                try:
                    result = subprocess.run(['python', 'download_kazakh_laws.py'], 
                                          capture_output=True, text=True, cwd='.')
                    if result.returncode == 0:
                        st.success("✅ Данные обновлены!")
                        st.session_state.rag_initialized = False
                    else:
                        st.error(f"❌ Ошибка обновления: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        # Примеры вопросов
        st.markdown("### 💡 Быстрые вопросы")
        example_questions = [
            "Что такое право собственности?",
            "Права потребителя при покупке товара",
            "Как заключить договор?",
            "Защита трудовых прав"
        ]
        
        for question in example_questions:
            if st.button(f"💬 {question}", key=f"example_{hash(question)}", use_container_width=True):
                add_message('user', question)
                st.rerun()
    
    with main_col:
        # Отображение чата
        display_chat()
        
        # Поле ввода
        st.markdown("### 💬 Ваш юридический вопрос:")
        
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                label="question",
                placeholder="Опишите вашу правовую ситуацию или задайте вопрос о законодательстве РК...",
                height=100,
                label_visibility="collapsed"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                submit_button = st.form_submit_button(
                    "🚀 Получить юридическую консультацию", 
                    type="primary", 
                    use_container_width=True
                )
            
            with col2:
                search_only = st.form_submit_button(
                    "🔍 Только поиск", 
                    use_container_width=True
                )
            
            with col3:
                clear_button = st.form_submit_button(
                    "🗑️ Очистить чат", 
                    use_container_width=True
                )
        
        # Обработка действий
        if (submit_button or search_only) and user_input.strip():
            if not st.session_state.rag_initialized:
                st.error("❌ База знаний не инициализирована. Используйте панель управления.")
            else:
                # Добавляем вопрос пользователя
                add_message('user', user_input)
                
                # Поиск документов
                with st.spinner("🔍 Поиск в законодательстве РК..."):
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
            add_message('ai', '💬 Чат очищен! Я готов ответить на ваши юридические вопросы.')
            st.rerun()
        
        # Информационные карточки
        st.markdown("---")
        st.markdown("### 🎯 Возможности системы")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(create_feature_card(
                "Анализ законодательства",
                "Глубокий анализ законов и нормативных актов РК",
                "📚", "#667eea"
            ), unsafe_allow_html=True)
            
            st.markdown(create_feature_card(
                "Правовые консультации",
                "Ответы на конкретные юридические вопросы",
                "💡", "#48bb78"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_feature_card(
                "Поиск прецедентов",
                "Нахождение релевантных правовых норм",
                "🔍", "#ed8936"
            ), unsafe_allow_html=True)
            
            st.markdown(create_feature_card(
                "Объяснение понятий",
                "Простое объяснение сложных юридических терминов",
                "⚖️", "#9f7aea"
            ), unsafe_allow_html=True)
    
    # Подвал
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #718096; padding: 2rem 0;">
        <p>⚖️ <strong>AI Lawyer</strong> - Интеллектуальная система юридической поддержки</p>
        <p style="font-size: 0.9rem; opacity: 0.7;">
            ⚠️ Внимание: Система предоставляет справочную информацию. 
            Для решения конкретных правовых вопросов обратитесь к квалифицированному юристу.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()