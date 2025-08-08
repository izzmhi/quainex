# main.py
from fastapi import FastAPI, Request, HTTPException, status, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv
import asyncio
from elevenlabs.client import ElevenLabs
from groq import Groq
from typing import Optional
import logging
from datetime import datetime

# ---------- Configuration ---------- #
load_dotenv()

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Keys
API_KEYS = {
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "together": os.getenv("TOGETHER_API_KEY"),
    "groq": os.getenv("GROQ_API_KEY"),
    "elevenlabs": os.getenv("ELEVENLABS_API_KEY"),
    "serper": os.getenv("SERPER_API_KEY")
}

# Initialize clients
clients = {
    "elevenlabs": ElevenLabs(api_key=API_KEYS["elevenlabs"]) if API_KEYS["elevenlabs"] else None,
    "groq": Groq(api_key=API_KEYS["groq"]) if API_KEYS["groq"] else None
}

# ---------- FastAPI App ---------- #
app = FastAPI(
    title="Quainex AI API",
    description="Premium AI Assistant Backend Service",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Configuration
origins = [
    "https://quainexai.onrender.com",
    "https://quainex.onrender.com",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Response-Time"]
)

# Middleware for logging and response time
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    response.headers["X-Response-Time"] = str(process_time)
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response

# ---------- Pydantic Models ---------- #
class ChatRequest(BaseModel):
    message: str
    provider: str = "openrouter"
    personality: str = "default"
    conversation_id: Optional[str] = None

class ToolRequest(BaseModel):
    tool: str
    content: str
    provider: str = "openrouter"
    options: Optional[dict] = None

class SearchRequest(BaseModel):
    query: str
    provider: str = "openrouter"
    num_results: int = 5

class TTSRequest(BaseModel):
    text: str
    voice: str = "Rachel"
    model: str = "eleven_monolingual_v2"

class ImageGenRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"
    quality: str = "standard"
    style: str = "vivid"

class VoiceRequest(BaseModel):
    language: str = "en-US"
    model: str = "whisper-1"

# ---------- Personality System ---------- #
PERSONALITIES = {
    "default": {
        "name": "Standard Assistant",
        "system_prompt": "You are Quainex, an intelligent AI assistant created by Bright SecureTech. Provide clear, concise, and helpful responses. Maintain a professional yet friendly tone."
    },
    "strict": {
        "name": "Technical Expert",
        "system_prompt": "You are Quainex, a highly technical AI assistant specialized in software development, cybersecurity, and system analysis. Provide precise, factual answers with minimal fluff."
    },
    "fun": {
        "name": "Entertaining Assistant",
        "system_prompt": "You're Quainex, a witty and entertaining AI. Use humor, emojis, and a casual tone while still being helpful. Keep responses engaging and fun!"
    },
    "creative": {
        "name": "Creative Writer",
        "system_prompt": "You are Quainex in creative mode. Provide imaginative, detailed responses. Use rich descriptions and creative analogies when appropriate."
    }
}

# ---------- Helper Functions ---------- #
def build_prompt_context(prompt: str, personality: str = "default", history: list = None):
    """Build the conversation context with personality and optional history"""
    base_prompt = PERSONALITIES.get(personality, PERSONALITIES["default"])["system_prompt"]
    messages = [{"role": "system", "content": base_prompt}]
    
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": prompt})
    return messages

# ---------- Robust fetch_ai_response (replace existing one) ----------
async def fetch_ai_response(provider: str, messages: list, timeout: int = 60):
    """Handle communication with different AI providers (robust parsing + logging)"""
    if provider not in API_KEYS or not API_KEYS[provider]:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"{provider.capitalize()} API key not configured"
        )

    endpoints = {
        "openrouter": {
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "model": "openai/gpt-4-turbo",
            "headers": {"Authorization": f"Bearer {API_KEYS['openrouter']}"}
        },
        "together": {
            "url": "https://api.together.xyz/v1/chat/completions",
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "headers": {"Authorization": f"Bearer {API_KEYS['together']}"}
        },
        "groq": {
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "model": "llama3-70b-8192",
            "headers": {"Authorization": f"Bearer {API_KEYS['groq']}"}
        }
    }

    config = endpoints.get(provider)
    if not config:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid provider specified")

    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(config["url"], headers=config["headers"], json=payload)
            # Raise for non-2xx
            resp.raise_for_status()

            raw_text = resp.text
            try:
                data = resp.json()
            except Exception:
                logger.error("Invalid JSON from provider: %s", raw_text)
                raise HTTPException(status_code=502, detail="Invalid JSON from AI provider")

            logger.info("%s raw response: %s", provider, data)

            # Defensive extraction: try a few common shapes
            content = None
            if isinstance(data, dict):
                # common openai-like shape
                choices = data.get("choices") or data.get("results") or []
                if choices and isinstance(choices, list):
                    first = choices[0]
                    if isinstance(first, dict):
                        content = (first.get("message") or {}).get("content") or first.get("text") or first.get("content")

                # fallback top-level keys
                content = content or data.get("response") or data.get("output_text") or data.get("text")

                # some providers return {"message": {"content": "..."}}
                if not content and isinstance(data.get("message"), dict):
                    content = data["message"].get("content")

            # Final sanity
            if not content or not str(content).strip():
                logger.error("No content found in provider response: %s", data)
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="AI provider returned empty response")

            return {
                "response": str(content).strip(),
                "model": data.get("model", config["model"]),
                "usage": data.get("usage", {})
            }

    except httpx.HTTPStatusError as e:
        # provider returned non-2xx
        body = e.response.text if e.response is not None else "no body"
        logger.error("API error from provider: %s %s", e.response.status_code if e.response else "?", body)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"AI service error: {body}")
    except HTTPException:
        # re-raise HTTPExceptions we deliberately raised above
        raise
    except Exception as e:
        logger.exception("Unexpected error fetching AI response")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get AI response: {str(e)}")
# --------------------------------------------------------------------


# ---------- API Routes ---------- #
@app.post("/api/chat", response_model=dict, tags=["AI Services"])
async def chat_handler(request: ChatRequest):
    """
    Handle chat conversations with different AI providers and personalities.
    
    - **message**: User's input message
    - **provider**: AI provider (openrouter|together|groq)
    - **personality**: Response style (default|strict|fun|creative)
    - **conversation_id**: Optional conversation ID for context
    """
    try:
        messages = build_prompt_context(
            request.message,
            request.personality
        )
        
        response = await fetch_ai_response(
            request.provider,
            messages
        )
        
        return {
            "success": True,
            "response": response["response"],
            "metadata": {
                "provider": request.provider,
                "model": response["model"],
                "personality": request.personality,
                "usage": response["usage"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during chat processing"
        )

@app.post("/api/tools/{tool}", response_model=dict, tags=["AI Services"])
async def tool_handler(tool: str, request: ToolRequest):
    """
    Handle various AI tools (summarize, translate, analyze, etc.)
    
    - **tool**: Tool to use (summarize|translate|analyze|search)
    - **content**: Content to process
    - **provider**: AI provider (openrouter|together|groq)
    - **options**: Additional tool-specific options
    """
    tool_prompts = {
        "summarize": "Provide a concise summary of the following content. Focus on key points and main ideas.",
        "translate": "Translate the following text to the specified language. Maintain original meaning and tone.",
        "analyze": "Analyze this content and provide key insights, patterns, and important information.",
        "search": "Process these search results and extract the most relevant information."
    }
    
    if tool not in tool_prompts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported tool specified"
        )
    
    try:
        messages = [
            {"role": "system", "content": tool_prompts[tool]},
            {"role": "user", "content": request.content}
        ]
        
        if request.options:
            messages[0]["content"] += f"\nOptions: {request.options}"
        
        response = await fetch_ai_response(
            request.provider,
            messages
        )
        
        return {
            "success": True,
            "result": response["response"],
            "tool": tool,
            "provider": request.provider
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tool error ({tool}): {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing {tool} request"
        )

@app.post("/api/search", response_model=dict, tags=["AI Services"])
async def search_handler(request: SearchRequest):
    """
    Perform web search and optionally process results with AI
    
    - **query**: Search query
    - **provider**: AI provider for processing (openrouter|together|groq)
    - **num_results**: Number of results to return (default: 5)
    """
    if not API_KEYS["serper"]:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Search functionality not configured"
        )
    
    try:
        headers = {"X-API-KEY": API_KEYS["serper"], "Content-Type": "application/json"}
        payload = {"q": request.query, "gl": "us", "hl": "en", "num": request.num_results}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://google.serper.dev/search",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            organic_results = data.get("organic", [])
            if not organic_results:
                return {
                    "success": True,
                    "results": [],
                    "message": "No results found"
                }
            
            # Process with AI if provider specified
            if request.provider:
                context = "\n".join([
                    f"Title: {r['title']}\nLink: {r['link']}\nSnippet: {r.get('snippet', '')}"
                    for r in organic_results[:5]
                ])
                
                ai_response = await fetch_ai_response(
                    request.provider,
                    [
                        {"role": "system", "content": "Summarize these search results concisely."},
                        {"role": "user", "content": context}
                    ]
                )
                
                return {
                    "success": True,
                    "results": organic_results,
                    "summary": ai_response["response"],
                    "processed": True
                }
            
            return {
                "success": True,
                "results": organic_results,
                "processed": False
            }
    except httpx.HTTPStatusError as e:
        logger.error(f"Search API error: {e.response.status_code} {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Search service error"
        )
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing search request"
        )

@app.post("/api/tts", tags=["AI Services"])
async def tts_handler(request: TTSRequest):
    """
    Convert text to speech
    
    - **text**: Text to convert
    - **voice**: Voice to use (Rachel, Bella, etc.)
    - **model**: TTS model to use
    """
    if not clients["elevenlabs"]:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="TTS service not configured"
        )
    
    try:
        audio = clients["elevenlabs"].generate(
            text=request.text,
            voice=request.voice,
            model=request.model
        )
        return Response(
            content=audio,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=quainex-tts.mp3"}
        )
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating speech"
        )

@app.post("/api/images", response_model=dict, tags=["AI Services"])
async def image_gen_handler(request: ImageGenRequest):
    """
    Generate images from text prompts
    
    - **prompt**: Description of the image to generate
    - **size**: Image dimensions (1024x1024, 512x512, etc.)
    - **quality**: Image quality (standard|hd)
    - **style**: Image style (vivid|natural)
    """
    if not clients["groq"]:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Image generation not configured"
        )
    
    try:
        response = clients["groq"].images.generate(
            model="dall-e-3",
            prompt=request.prompt,
            size=request.size,
            quality=request.quality,
            style=request.style,
            n=1
        )
        
        return {
            "success": True,
            "image_url": response.data[0].url,
            "revised_prompt": response.data[0].revised_prompt
        }
    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating image"
        )

@app.post("/api/voice", response_model=dict, tags=["AI Services"])
async def voice_handler(file: UploadFile = File(...), request: VoiceRequest = None):
    """
    Convert speech to text
    
    - **file**: Audio file to transcribe
    - **language**: Language of the audio (en-US, etc.)
    - **model**: Transcription model to use
    """
    # This is a placeholder implementation
    # In production, integrate with Whisper or similar service
    logger.info(f"Received voice file: {file.filename} ({file.content_type})")
    
    return {
        "success": True,
        "message": "Voice transcription service placeholder",
        "details": "In production, this would return transcribed text from the audio file."
    }

# ---------- Metadata Endpoints ---------- #
@app.get("/api/models", response_model=dict, tags=["Metadata"])
async def list_models():
    """List available AI models for each provider"""
    return {
        "openrouter": [
            {"id": "openai/gpt-4-turbo", "name": "GPT-4 Turbo"},
            {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus"}
        ],
        "together": [
            {"id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "name": "Mixtral 8x7B"},
            {"id": "meta-llama/Llama-3-70b-chat-hf", "name": "Llama 3 70B"}
        ],
        "groq": [
            {"id": "llama3-70b-8192", "name": "Llama 3 70B"},
            {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B"}
        ]
    }

@app.get("/api/personalities", response_model=dict, tags=["Metadata"])
async def list_personalities():
    """List available response personalities"""
    return {
        "personalities": [
            {"id": key, "name": value["name"]} 
            for key, value in PERSONALITIES.items()
        ]
    }

@app.get("/api/status", response_model=dict, tags=["Metadata"])
async def service_status():
    """Check service status and available features"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "features": {
            "chat": bool(API_KEYS["openrouter"] or API_KEYS["together"] or API_KEYS["groq"]),
            "tts": bool(API_KEYS["elevenlabs"]),
            "image_generation": bool(API_KEYS["groq"]),
            "search": bool(API_KEYS["serper"])
        },
        "uptime": str(datetime.now())
    }

# ---------- Error Handlers ---------- #
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "code": exc.status_code
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Server error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR
        }
    )