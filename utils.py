"""
Utility functions for the AI Lawyer project.
Contains functions for loading and processing legal documents.
"""

import os
import glob
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_law_texts(data_dir: str = "data/laws_raw") -> List[Tuple[str, str]]:
    """
    Load all legal text files from the specified directory.
    
    Args:
        data_dir (str): Directory containing legal text files
        
    Returns:
        List[Tuple[str, str]]: List of tuples containing (filename, content)
    """
    law_texts = []
    
    # Get absolute path
    abs_data_dir = os.path.abspath(data_dir)
    
    if not os.path.exists(abs_data_dir):
        logger.warning(f"Directory {abs_data_dir} does not exist. Creating it...")
        os.makedirs(abs_data_dir, exist_ok=True)
        return law_texts
    
    # Supported file extensions
    extensions = ['*.txt', '*.md', '*.rtf']
    
    for extension in extensions:
        pattern = os.path.join(abs_data_dir, extension)
        files = glob.glob(pattern)
        
        for file_path in files:
            try:
                # Try different encodings
                encodings = ['utf-8', 'cp1251', 'windows-1251', 'latin-1']
                content = None
                
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content:
                    filename = os.path.basename(file_path)
                    law_texts.append((filename, content))
                    logger.info(f"Loaded {filename} ({len(content)} characters)")
                else:
                    logger.error(f"Could not decode file: {file_path}")
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
    
    logger.info(f"Total loaded documents: {len(law_texts)}")
    return law_texts


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better RAG performance.
    
    Args:
        text (str): Input text to split
        chunk_size (int): Maximum size of each chunk
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of the current chunk
        end = start + chunk_size
        
        # If we're not at the end of the text, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 200 characters
            sentence_endings = ['.', '!', '?', '\n']
            best_break = end
            
            for i in range(max(0, end - 200), min(len(text), end + 100)):
                if text[i] in sentence_endings and i > start + chunk_size // 2:
                    best_break = i + 1
                    break
            
            end = best_break
        
        # Extract the chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    logger.info(f"Split text into {len(chunks)} chunks (avg size: {sum(len(c) for c in chunks) // len(chunks) if chunks else 0})")
    return chunks


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text (str): Raw text content
        
    Returns:
        str: Cleaned text
    """
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might interfere with processing
    # Keep Cyrillic, Latin, numbers, and common punctuation
    import re
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'«»]', '', text)
    
    return text


def get_sample_legal_text() -> str:
    """
    Returns a sample legal text for testing purposes.
    
    Returns:
        str: Sample legal text in Russian
    """
    return """
    Гражданский кодекс Республики Казахстан
    
    Статья 1. Основные начала гражданского законодательства
    1. Гражданское законодательство основывается на признании равенства участников регулируемых им отношений, неприкосновенности собственности, свободы договора, недопустимости произвольного вмешательства кого-либо в частные дела, необходимости беспрепятственного осуществления гражданских прав, обеспечения восстановления нарушенных прав, их судебной защиты.
    
    Статья 2. Отношения, регулируемые гражданским законодательством
    1. Гражданское законодательство определяет правовое положение участников гражданского оборота, основания возникновения и порядок осуществления права собственности и других вещных прав, исключительных прав на результаты интеллектуальной деятельности (интеллектуальной собственности), регулирует договорные и иные обязательства, а также другие имущественные и связанные с ними личные неимущественные отношения.
    
    Статья 3. Гражданское законодательство и нормы международного права
    1. Если международным договором, ратифицированным Республикой Казахстан, установлены иные правила, чем те, которые предусмотрены гражданским законодательством, то применяются правила международного договора.
    """


if __name__ == "__main__":
    # Test the functions
    print("Testing utility functions...")
    
    # Test text splitting
    sample_text = get_sample_legal_text()
    chunks = split_text(sample_text, chunk_size=500, overlap=100)
    
    print(f"Original text length: {len(sample_text)}")
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} length: {len(chunk)}")
    
    # Test loading (will create empty directory if not exists)
    laws = load_law_texts()
    print(f"Loaded {len(laws)} legal documents")