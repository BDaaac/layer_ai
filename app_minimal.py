"""
AI Lawyer - Минималистичный дизайн
Красивый и функциональный интерфейс с логотипом
"""

import streamlit as st
import time
from datetime import datetime
import base64
import os

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
    from model_saiga import generate_answer_with_saiga, is_saiga_available, saiga_lawyer
    SAIGA_AVAILABLE = True
except ImportError as e:
    SAIGA_AVAILABLE = False

# Конфигурация страницы
st.set_page_config(
    page_title="AI Lawyer",
    page_icon="⚖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def load_logo():
    """Загружает логотип"""
    try:
        if os.path.exists("Frame 2 (1).svg"):
            with open("Frame 2 (1).svg", "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
        pass
    return None

# Минималистичный CSS
def load_minimal_css():
    logo_base64 = load_logo()
    
    logo_html = ""
    if logo_base64:
        logo_html = f"""
        .logo {{
            background-image: url("data:image/svg+xml;base64,{logo_base64}");
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            width: 60px;
            height: 60px;
            margin: 0 auto;
        }}
        """
    
    st.markdown(f"""
    <style>
        /* Общие стили */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {{
            --primary: #7A3EB1;
            --dark: #4B296B;
            --light: #F8F4FF;
            --white: #FFFFFF;
            --text: #2D2B32;
            --gray: #8E8E93;
            --border: #E5E5E7;
        }}
        
        .main {{
            background: linear-gradient(180deg, #FAFBFD 0%, #F5F7FA 100%);
            font-family: 'Inter', sans-serif;
        }}
        
        /* Скрыть элементы */
        #MainMenu, footer, header, .stDeployButton {{
            visibility: hidden;
        }}
        
        {logo_html}
        
        /* Заголовок */
        .header {{
            text-align: center;
            padding: 3rem 0 2rem 0;
            background: var(--white);
            border-radius: 24px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 24px rgba(0,0,0,0.04);
            border: 1px solid var(--border);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
            margin: 1rem 0 0.5rem 0;
            letter-spacing: -0.02em;
        }}
        
        .header p {{
            font-size: 1.1rem;
            color: var(--gray);
            margin: 0;
            font-weight: 400;
        }}
        
        /* Чат */
        .chat-box {{
            background: var(--white);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 2px 20px rgba(0,0,0,0.04);
            border: 1px solid var(--border);
            max-height: 600px;
            overflow-y: auto;
        }}
        
        /* Сообщения */
        .message {{
            margin: 1.5rem 0;
            animation: fadeIn 0.3s ease;
        }}
        
        .user-msg {{
            background: linear-gradient(135deg, var(--primary), var(--dark));
            color: var(--white);
            padding: 1rem 1.5rem;
            border-radius: 18px 18px 4px 18px;
            margin-left: 20%;
            font-weight: 500;
        }}
        
        .ai-msg {{
            background: var(--light);
            color: var(--text);
            padding: 1rem 1.5rem;
            border-radius: 18px 18px 18px 4px;
            margin-right: 20%;
            border-left: 3px solid var(--primary);
        }}
        
        .search-msg {{
            background: #F0F9FF;
            color: #0369A1;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            margin-right: 15%;
            border: 1px solid #BAE6FD;
            font-size: 0.9rem;
        }}
        
        .timestamp {{
            text-align: center;
            color: var(--gray);
            font-size: 0.8rem;
            margin: 0.5rem 0;
        }}
        
        /* Форма ввода */
        .input-box {{
            background: var(--white);
            border-radius: 20px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 20px rgba(0,0,0,0.04);
            border: 1px solid var(--border);
        }}
        
        .stTextArea textarea {{
            border: 2px solid var(--border);
            border-radius: 16px;
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            padding: 1rem;
            transition: all 0.2s ease;
        }}
        
        .stTextArea textarea:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(122, 62, 177, 0.1);
        }}
        
        /* Кнопки */
        .stButton button {{
            background: linear-gradient(135deg, var(--primary), var(--dark));
            border: none;
            border-radius: 12px;
            color: var(--white);
            font-weight: 500;
            font-size: 0.95rem;
            padding: 0.75rem 1.5rem;
            transition: all 0.2s ease;
            box-shadow: 0 2px 12px rgba(122, 62, 177, 0.2);
        }}
        
        .stButton button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 20px rgba(122, 62, 177, 0.3);
        }}
        
        /* Статус */
        .status {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            margin: 0.25rem;
        }}
        
        .status-ok {{
            background: #ECFDF5;
            color: #059669;
            border: 1px solid #A7F3D0;
        }}
        
        .status-error {{
            background: #FEF2F2;
            color: #DC2626;
            border: 1px solid #FECACA;
        }}
        
        .status-warning {{
            background: #FFFBEB;
            color: #D97706;
            border: 1px solid #FDE68A;
        }}
        
        /* Сайдбар */
        .sidebar {{
            background: var(--white);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 12px rgba(0,0,0,0.04);
            border: 1px solid var(--border);
        }}
        
        /* Анимации */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* Прокрутка */
        * {{
            scrollbar-width: thin;
            scrollbar-color: var(--primary) transparent;
        }}
        
        *::-webkit-scrollbar {{ width: 6px; }}
        *::-webkit-scrollbar-thumb {{
            background: var(--primary);
            border-radius: 3px;
        }}
        
        /* Адаптивность */
        @media (max-width: 768px) {{
            .user-msg {{ margin-left: 10%; }}
            .ai-msg {{ margin-right: 10%; }}
            .search-msg {{ margin-right: 5%; }}
            .header h1 {{ font-size: 2rem; }}
        }}
    </style>
    """, unsafe_allow_html=True)

load_minimal_css()

def init_session():
    """Инициализация сессии"""
    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            'type': 'ai',
            'content': 'Добро пожаловать! Я AI Lawyer - ваш юридический ассистент по законодательству Казахстана. Как могу помочь?',
            'time': datetime.now().strftime("%H:%M")
        }]
    
    if 'rag_ready' not in st.session_state:
        st.session_state.rag_ready = False
    
    if 'model_ready' not in st.session_state:
        st.session_state.model_ready = False

def display_chat():
    """Отображение чата"""
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        time_html = f'<div class="timestamp">{msg["time"]}</div>'
        
        if msg['type'] == 'user':
            st.markdown(f'''
            <div class="message">
                <div class="user-msg">{msg["content"]}</div>
                {time_html}
            </div>
            ''', unsafe_allow_html=True)
            
        elif msg['type'] == 'ai':
            st.markdown(f'''
            <div class="message">
                <div class="ai-msg">{msg["content"]}</div>
                {time_html}
            </div>
            ''', unsafe_allow_html=True)
            
        elif msg['type'] == 'search':
            st.markdown(f'''
            <div class="message">
                <div class="search-msg"><strong>Результаты поиска:</strong><br>{msg["content"]}</div>
                {time_html}
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def add_message(msg_type, content):
    """Добавить сообщение"""
    st.session_state.messages.append({
        'type': msg_type,
        'content': content,
        'time': datetime.now().strftime("%H:%M")
    })

def search_summary(results):
    """Краткий обзор результатов поиска"""
    if not results:
        return "Релевантные документы не найдены"
    
    summary = f"Найдено {len(results)} документов:<br>"
    for i, result in enumerate(results[:3], 1):
        source = result['metadata']['source']
        score = result['score']
        summary += f"{i}. {source} ({score:.2f})<br>"
    
    if len(results) > 3:
        summary += f"+ еще {len(results) - 3}"
    
    return summary

def generate_response(question, search_results):
    """Генерация ответа"""
    # Попытка использовать Saiga-2
    if st.session_state.model_ready and SAIGA_AVAILABLE:
        try:
            with st.spinner("Генерация ответа..."):
                result = generate_answer_with_saiga(question, search_results)
                if result.get('success'):
                    return result['answer']
                else:
                    st.warning(f"Ошибка модели: {result.get('error', 'Unknown')}")
        except Exception as e:
            st.warning(f"Ошибка: {str(e)}")
    
    # Fallback
    if SIMPLE_LAWYER_AVAILABLE:
        result = simple_generate_answer(question, search_results)
        return result['answer']
    
    return "Извините, сервис временно недоступен."

def main():
    # Инициализация
    init_session()
    
    # Заголовок
    st.markdown('''
    <div class="header">
        <div class="logo"></div>
        <h1>AI Lawyer</h1>
        <p>Юридический ИИ-ассистент для Казахстана</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Боковая панель
    with st.sidebar:
        st.markdown('<div class="sidebar">', unsafe_allow_html=True)
        st.markdown("### Система")
        
        # RAG статус
        if not st.session_state.rag_ready:
            if st.button("Инициализировать базу", use_container_width=True):
                with st.spinner("Загрузка..."):
                    success = build_index()
                    st.session_state.rag_ready = success
                    if success:
                        st.success("База готова!")
                    else:
                        st.error("Ошибка инициализации")
                    st.rerun()
        
        # Статус RAG
        if st.session_state.rag_ready:
            st.markdown('<div class="status status-ok">База готова</div>', unsafe_allow_html=True)
            try:
                stats = get_rag_stats()
                st.write(f"Документов: {len(stats.get('sources', []))}")
                st.write(f"Фрагментов: {stats.get('total_chunks', 0)}")
            except:
                pass
        else:
            st.markdown('<div class="status status-error">База не готова</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Модель Saiga-2
        if SAIGA_AVAILABLE:
            if not st.session_state.model_ready:
                if st.button("Проверить модель", use_container_width=True):
                    with st.spinner("Проверка..."):
                        ready = is_saiga_available()
                        st.session_state.model_ready = ready
                        if ready:
                            st.success("Модель готова!")
                        else:
                            st.error("Модель не найдена")
                        st.rerun()
            
            if st.session_state.model_ready:
                st.markdown('<div class="status status-ok">Saiga-2 готова</div>', unsafe_allow_html=True)
                try:
                    info = saiga_lawyer.get_model_info()
                    if info['model_file_exists']:
                        st.write(f"Размер: {info['model_size_mb']:.0f} MB")
                except:
                    pass
            else:
                st.markdown('<div class="status status-error">Модель не готова</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status status-warning">llama_cpp не установлен</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Примеры
        st.markdown("### Примеры вопросов")
        examples = [
            "Права собственника",
            "Защита потребителей",
            "Трудовые права",
            "Административные штрафы"
        ]
        
        for example in examples:
            if st.button(example, key=f"ex_{example}", use_container_width=True):
                add_message('user', example)
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Чат
    display_chat()
    
    # Ввод
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    
    with st.form("chat", clear_on_submit=True):
        user_input = st.text_area(
            "Задайте вопрос",
            placeholder="Например: Какие права имеет собственник недвижимости?",
            height=80,
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            send = st.form_submit_button("Отправить", type="primary", use_container_width=True)
        with col2:
            search_only = st.form_submit_button("Поиск", use_container_width=True)
        with col3:
            clear = st.form_submit_button("Очистить", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Обработка
    if clear:
        st.session_state.messages = [{
            'type': 'ai',
            'content': 'Чат очищен. Готов к новым вопросам!',
            'time': datetime.now().strftime("%H:%M")
        }]
        st.rerun()
    
    if (send or search_only) and user_input.strip():
        if not st.session_state.rag_ready:
            st.error("Сначала инициализируйте базу знаний")
            return
        
        # Добавить вопрос
        add_message('user', user_input)
        
        # Поиск
        with st.spinner("Поиск..."):
            results = search_law(user_input, k=5)
        
        # Результаты поиска
        summary = search_summary(results)
        add_message('search', summary)
        
        # Ответ (если не только поиск)
        if send:
            answer = generate_response(user_input, results)
            add_message('ai', answer)
        
        st.rerun()

if __name__ == "__main__":
    main()