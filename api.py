"""
FastAPI backend for the AI Lawyer project.
Provides REST API endpoints for legal consultation.
"""

import os
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager

from lawyer import generate_answer, load_lawyer_model, saiga_lawyer
from rag import build_index, search_law, get_rag_stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class LegalQuestion(BaseModel):
    """Request model for legal questions."""
    question: str = Field(..., description="Legal question in Russian", min_length=5, max_length=1000)
    use_context: bool = Field(True, description="Whether to use RAG context")
    max_context: int = Field(3, description="Maximum number of context chunks", ge=1, le=10)


class ContextChunk(BaseModel):
    """Model for context chunks in response."""
    content: str
    source: str
    relevance_score: float
    rank: int


class LegalAnswer(BaseModel):
    """Response model for legal answers."""
    question: str
    answer: str
    context_chunks: List[ContextChunk]
    model_info: Dict[str, Any]
    processing_time: Optional[float] = None
    error: Optional[str] = None


class SystemStatus(BaseModel):
    """System status model."""
    status: str
    model_loaded: bool
    rag_initialized: bool
    rag_stats: Dict[str, Any]
    version: str = "1.0.0"


class BuildIndexRequest(BaseModel):
    """Request model for building RAG index."""
    force_rebuild: bool = Field(False, description="Force rebuilding even if index exists")
    data_directory: str = Field("data/laws_raw", description="Directory containing legal documents")


# Global state
app_state = {
    "model_loaded": False,
    "rag_initialized": False
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting AI Lawyer API...")
    
    # Initialize RAG system
    try:
        logger.info("Initializing RAG system...")
        rag_success = build_index()
        app_state["rag_initialized"] = rag_success
        
        if rag_success:
            logger.info("RAG system initialized successfully")
        else:
            logger.warning("RAG system initialization failed")
    except Exception as e:
        logger.error(f"Error initializing RAG: {str(e)}")
    
    # Try to load model
    try:
        logger.info("Loading Saiga model...")
        model_success = load_lawyer_model()
        app_state["model_loaded"] = model_success
        
        if model_success:
            logger.info("Saiga model loaded successfully")
        else:
            logger.warning("Saiga model loading failed - API will work in limited mode")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
    
    yield
    
    logger.info("Shutting down AI Lawyer API...")


# Create FastAPI app
app = FastAPI(
    title="AI Lawyer API",
    description="Legal consultation API powered by Saiga-2 and RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "AI Lawyer API", 
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Health check endpoint."""
    try:
        rag_stats = get_rag_stats()
    except Exception:
        rag_stats = {"error": "RAG stats unavailable"}
    
    return SystemStatus(
        status="healthy" if app_state["rag_initialized"] else "degraded",
        model_loaded=app_state["model_loaded"],
        rag_initialized=app_state["rag_initialized"],
        rag_stats=rag_stats
    )


@app.post("/ask", response_model=LegalAnswer)
async def ask_legal_question(request: LegalQuestion):
    """
    Ask a legal question and get an AI-powered answer.
    
    Args:
        request: Legal question with optional parameters
        
    Returns:
        Legal answer with context and metadata
    """
    import time
    start_time = time.time()
    
    try:
        # Validate system state
        if not app_state["rag_initialized"]:
            raise HTTPException(
                status_code=503, 
                detail="RAG system not initialized. Please check system status."
            )
        
        logger.info(f"Processing question: {request.question[:100]}...")
        
        # Get context if requested
        context_chunks = []
        if request.use_context:
            try:
                search_results = search_law(request.question, k=request.max_context)
                context_chunks = search_results
                logger.info(f"Retrieved {len(context_chunks)} context chunks")
            except Exception as e:
                logger.error(f"Error retrieving context: {str(e)}")
                # Continue without context
        
        # Generate answer
        try:
            if app_state["model_loaded"]:
                result = generate_answer(request.question, context_chunks)
                answer_text = result['answer']
                error_msg = result.get('error')
            else:
                # Fallback response when model is not loaded
                answer_text = self._generate_fallback_answer(request.question, context_chunks)
                error_msg = "Model not loaded - using fallback response"
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer_text = f"Произошла ошибка при генерации ответа: {str(e)}"
            error_msg = str(e)
        
        # Format context chunks for response
        formatted_chunks = []
        for chunk in context_chunks:
            formatted_chunks.append(ContextChunk(
                content=chunk['chunk'],
                source=chunk['metadata']['source'],
                relevance_score=chunk['score'],
                rank=chunk['rank']
            ))
        
        # Prepare model info
        model_info = {
            "model_name": "Saiga-2 7B" if app_state["model_loaded"] else "Fallback",
            "model_loaded": app_state["model_loaded"],
            "context_chunks_used": len(formatted_chunks),
            "rag_enabled": request.use_context
        }
        
        processing_time = time.time() - start_time
        
        return LegalAnswer(
            question=request.question,
            answer=answer_text,
            context_chunks=formatted_chunks,
            model_info=model_info,
            processing_time=processing_time,
            error=error_msg
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def _generate_fallback_answer(question: str, context_chunks: List[Dict]) -> str:
    """
    Generate a fallback answer when the model is not available.
    
    Args:
        question: User question
        context_chunks: Retrieved context
        
    Returns:
        Fallback answer
    """
    if context_chunks:
        context_text = "\n\n".join([
            f"Из {chunk['metadata']['source']}:\n{chunk['chunk'][:300]}..."
            for chunk in context_chunks[:2]
        ])
        
        return f"""К сожалению, основная модель ИИ недоступна, но я нашел релевантную информацию в правовых документах:

{context_text}

Для получения полного юридического анализа рекомендую:
1. Обратиться к квалифицированному юристу
2. Изучить полные тексты упомянутых нормативных актов
3. Учесть специфику вашей ситуации

Внимание: данная информация носит справочный характер и не является юридической консультацией."""
    else:
        return f"""К сожалению, основная модель ИИ недоступна, и в базе данных не найдено релевантной информации по вашему вопросу: "{question}"

Рекомендации:
1. Обратитесь к квалифицированному юристу
2. Изучите актуальное законодательство Республики Казахстан
3. Проконсультируйтесь в юридической консультации

Внимание: для получения точной правовой помощи необходимо обращение к специалистам."""


@app.post("/search", response_model=List[ContextChunk])
async def search_legal_documents(
    query: str = Field(..., description="Search query"),
    limit: int = Field(5, description="Maximum number of results", ge=1, le=20)
):
    """
    Search legal documents without generating an answer.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of relevant document chunks
    """
    try:
        if not app_state["rag_initialized"]:
            raise HTTPException(
                status_code=503,
                detail="RAG system not initialized"
            )
        
        results = search_law(query, k=limit)
        
        formatted_results = []
        for result in results:
            formatted_results.append(ContextChunk(
                content=result['chunk'],
                source=result['metadata']['source'],
                relevance_score=result['score'],
                rank=result['rank']
            ))
        
        return formatted_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/build-index")
async def build_rag_index(request: BuildIndexRequest, background_tasks: BackgroundTasks):
    """
    Build or rebuild the RAG index.
    
    Args:
        request: Build index parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        Status message
    """
    def rebuild_index():
        try:
            logger.info("Starting index rebuild...")
            success = build_index(
                data_dir=request.data_directory,
                force_rebuild=request.force_rebuild
            )
            app_state["rag_initialized"] = success
            logger.info(f"Index rebuild completed. Success: {success}")
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}")
            app_state["rag_initialized"] = False
    
    background_tasks.add_task(rebuild_index)
    
    return {
        "message": "Index rebuild started in background",
        "force_rebuild": request.force_rebuild,
        "data_directory": request.data_directory
    }


@app.get("/stats", response_model=Dict[str, Any])
async def get_system_stats():
    """
    Get detailed system statistics.
    
    Returns:
        System statistics including RAG and model info
    """
    try:
        stats = {
            "api_version": "1.0.0",
            "model_status": {
                "loaded": app_state["model_loaded"],
                "name": "Saiga-2 7B" if app_state["model_loaded"] else "Not loaded"
            },
            "rag_status": {
                "initialized": app_state["rag_initialized"]
            }
        }
        
        if app_state["rag_initialized"]:
            try:
                rag_stats = get_rag_stats()
                stats["rag_status"]["details"] = rag_stats
            except Exception as e:
                stats["rag_status"]["error"] = str(e)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting AI Lawyer API on {host}:{port}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )