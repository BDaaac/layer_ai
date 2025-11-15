#!/usr/bin/env python
"""
Quick setup script for the AI Lawyer project.
This script helps with initial setup and testing.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    print("=" * 60)
    print("🤖 AI Lawyer - Setup Script")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("✅ Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_virtual_env():
    """Check if virtual environment is active."""
    print("✅ Checking virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is active")
        return True
    else:
        print("⚠️ Virtual environment not detected")
        print("Please activate with: .\\venv\\Scripts\\activate")
        return False

def check_directories():
    """Check and create necessary directories."""
    print("✅ Checking directories...")
    
    directories = [
        "data/laws_raw",
        "models/saiga",
        "data"
    ]
    
    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 Created directory: {dir_path}")
        else:
            print(f"✅ Directory exists: {dir_path}")

def check_model_file():
    """Check if Saiga model file exists."""
    print("✅ Checking Saiga model...")
    model_path = "models/saiga/saiga2.gguf"
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Saiga model found ({size_mb:.1f} MB)")
        return True
    else:
        print("⚠️ Saiga model not found")
        print("📥 Please download from: https://huggingface.co/IlyaGusev/saiga2_7b_gguf")
        print(f"💾 Save as: {model_path}")
        return False

def check_legal_documents():
    """Check for legal documents."""
    print("✅ Checking legal documents...")
    
    laws_dir = "data/laws_raw"
    files = list(Path(laws_dir).glob("*.txt"))
    
    if files:
        print(f"✅ Found {len(files)} legal document(s)")
        for file in files[:3]:  # Show first 3 files
            print(f"  📄 {file.name}")
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more")
        return True
    else:
        print("⚠️ No legal documents found")
        print("📄 Please add .txt files to data/laws_raw/")
        
        # Create sample file
        sample_path = Path(laws_dir) / "sample_civil_code.txt"
        if not sample_path.exists():
            from utils import get_sample_legal_text
            with open(sample_path, 'w', encoding='utf-8') as f:
                f.write(get_sample_legal_text())
            print(f"📝 Created sample file: {sample_path}")
        
        return False

def test_imports():
    """Test if all required packages are installed."""
    print("✅ Testing package imports...")
    
    packages = [
        ("sentence_transformers", "SentenceTransformers"),
        ("faiss", "FAISS"),
        ("fastapi", "FastAPI"),
        ("streamlit", "Streamlit"),
        ("ctransformers", "CTransformers"),
        ("torch", "PyTorch"),
    ]
    
    all_good = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - not installed")
            all_good = False
    
    return all_good

def test_rag_system():
    """Test RAG system initialization."""
    print("✅ Testing RAG system...")
    
    try:
        from rag import build_index, get_rag_stats
        
        success = build_index(force_rebuild=False)
        if success:
            stats = get_rag_stats()
            print(f"✅ RAG system working ({stats.get('total_chunks', 0)} chunks)")
            return True
        else:
            print("⚠️ RAG system initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ RAG system error: {str(e)}")
        return False

def run_quick_test():
    """Run a quick functionality test."""
    print("✅ Running quick test...")
    
    try:
        from rag import search_law
        from lawyer import generate_answer
        
        # Test search
        results = search_law("гражданские права", k=1)
        if results:
            print("✅ Document search working")
        else:
            print("⚠️ Document search returned no results")
        
        # Test answer generation (without full model)
        test_question = "Что такое гражданские права?"
        answer = generate_answer(test_question, use_rag=True)
        
        if answer.get('answer'):
            print("✅ Answer generation working")
            print(f"📝 Sample answer length: {len(answer['answer'])} chars")
        else:
            print("⚠️ Answer generation failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def show_next_steps(model_available, docs_available):
    """Show next steps based on setup status."""
    print()
    print("🚀 Next Steps:")
    print("-" * 40)
    
    if not model_available:
        print("1. 📥 Download Saiga-2 model:")
        print("   https://huggingface.co/IlyaGusev/saiga2_7b_gguf")
        print("   Save as: models/saiga/saiga2.gguf")
        print()
    
    if not docs_available:
        print("2. 📄 Add legal documents:")
        print("   Place .txt files in data/laws_raw/")
        print("   (Civil code, Constitution, etc.)")
        print()
    
    print("3. 🚀 Start the system:")
    print("   Terminal 1: python api.py")
    print("   Terminal 2: streamlit run ui.py")
    print()
    
    print("4. 🌐 Access the system:")
    print("   Web UI: http://localhost:8501")
    print("   API docs: http://localhost:8000/docs")
    print()

def main():
    """Main setup function."""
    print_banner()
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_env),
        ("Directories", check_directories),
        ("Package Imports", test_imports),
    ]
    
    print("Running system checks...")
    print("-" * 40)
    
    all_passed = True
    for name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {name}: Error - {str(e)}")
            all_passed = False
        print()
    
    # Additional checks
    model_available = check_model_file()
    print()
    
    docs_available = check_legal_documents()
    print()
    
    if all_passed:
        print("Testing system functionality...")
        print("-" * 40)
        
        rag_working = test_rag_system()
        print()
        
        if rag_working:
            test_working = run_quick_test()
            print()
        
        if rag_working:
            print("🎉 System is ready to use!")
        else:
            print("⚠️ System partially ready (some issues detected)")
    else:
        print("❌ Please fix the issues above before proceeding")
    
    show_next_steps(model_available, docs_available)

if __name__ == "__main__":
    main()