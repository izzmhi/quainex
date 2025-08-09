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
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import sqlite3
import json
from supabase import create_client

# ---------- Configuration ----------
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# API Keys (ensure these are in your .env)
API_KEYS = {
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "together": os.getenv("TOGETHER_API_KEY"),
    "groq": os.getenv("GROQ_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "elevenlabs": os.getenv("ELEVENLABS_API_KEY"),
    "serper": os.getenv("SERPER_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
    "supabaseurl": os.getenv("SUPABASE_URL"),
    "supabasekey": os.getenv("SUPABASE_KEY")
}

# Define FastAPI app first
app = FastAPI(
    title="Quainex AI API",
    description="Premium AI Assistant Backend Service",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)
# Now initialize Supabase client with actual keys
SUPABASE_URL = API_KEYS["supabaseurl"]
SUPABASE_KEY = API_KEYS["supabasekey"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Startup event to cleanup old history
@app.on_event("startup")
async def cleanup_old_history():
    thirty_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)
    res = supabase.table("chat_history")\
        .delete()\
        .lt("created_at", thirty_days_ago.isoformat())\
        .execute()
    logger.info(f"Supabase cleanup deleted {res.data} rows older than 30 days")

# Clients (third-party SDKs if available)
clients = {
    "elevenlabs": ElevenLabs(api_key=API_KEYS["elevenlabs"]) if API_KEYS.get("elevenlabs") else None,
    "groq": Groq(api_key=API_KEYS["groq"]) if API_KEYS.get("groq") else None
}

# Constants
DEFAULT_MAX_TOKENS = 500
PROVIDER_FALLBACK_ORDER = ["openrouter", "together", "gemini", "groq", "deepseek"]

# CORS Configuration — adjust origins as needed
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

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    response.headers["X-Response-Time"] = str(process_time)
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response
@app.get("/")
async def root():
    return {"message": "Hello from Quainex API!"}

    

@app.on_event("startup")
async def cleanup_old_history():
    thirty_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)
    res = supabase.table("chat_history")\
        .delete()\
        .lt("created_at", thirty_days_ago.isoformat())\
        .execute()
    logger.info(f"Supabase cleanup deleted {res.data} rows older than 30 days")
    return {"message": "Hello from Quainex API!"}

# ---------- Pydantic Models ----------
class ChatRequest(BaseModel):
    message: str
    provider: str = "openrouter"        # can be 'openrouter', 'together', 'gemini', 'groq', or 'ensemble'
    personality: str = "default"
    conversation_id: Optional[str] = None

class ToolRequest(BaseModel):
    tool: str
    content: str
    provider: str = "deepseek"
    options: Optional[dict] = None

class SearchRequest(BaseModel):
    query: str
    provider: str = "deepseek"
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

# ---------- Personalities ----------
PERSONALITIES = {
    "default": {
    "name": "Technical Expert (90% tech)",
    "system_prompt": (
        "You are Quainex, a world-class senior technical consultant created by Bright SecureTech. "
        "Your role is to assist with advanced software engineering, backend/frontend development, "
        "mobile apps, cybersecurity, penetration testing, DevOps, cloud computing, AI/ML, and data science. "
        "For every answer: "
        "1. Begin with 'Problem → Analysis → Solution → Example'. "
        "2. Include security considerations, edge cases, and performance tips. "
        "3. Provide real production-ready code snippets (prefer Python, JavaScript/Node.js, and Bash). "
        "4. Use bullet points for clarity and numbered steps for processes. "
        "5. End with a 'Deployment Checklist' containing practical action items to implement the solution in a live system. "
        "6. If applicable, include architecture diagrams in ASCII format. "
        "7. Avoid generic theory-only answers; all responses must be implementation-ready."
    )
},

    "strict": {
        "name": "Technical Expert (concise)",
        "system_prompt": (
            "You are Quainex, a concise technical expert. Provide precise, factual answers focused on correctness and brevity. Include commands and code snippets when relevant."
        )
    },
    "fun": {
        "name": "Entertaining Assistant",
        "system_prompt": "You're Quainex, a witty and entertaining AI. Use humor, emojis, and a casual tone while still being helpful."
    },
    "creative": {
        "name": "Creative Writer",
        "system_prompt": "You are Quainex in creative mode. Provide imaginative, detailed responses with rich descriptions."
    }
}

# ---------- Helper Utilities ----------
def build_prompt_context(prompt: str, personality: str = "default", history: list = None):
    base_prompt = PERSONALITIES.get(personality, PERSONALITIES["default"])["system_prompt"]
    messages = [{"role": "system", "content": base_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages

# SQLite logging
DB_PATH = "chat_logs.db"
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            provider_requested TEXT,
            final_provider TEXT,
            user_message TEXT,
            ai_response TEXT,
            response_candidates TEXT,
            personality TEXT
        )
    """)
    conn.commit()
    return conn

_db_conn = init_db()
_db_cursor = _db_conn.cursor()

def log_chat_db(user_message: str, final_provider: str, ai_response: str,
                provider_requested: str, candidates: List[Dict[str, Any]], personality: str):
    try:
        _db_cursor.execute(
            "INSERT INTO chats (timestamp, provider_requested, final_provider, user_message, ai_response, response_candidates, personality) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), provider_requested, final_provider, user_message, ai_response, json.dumps(candidates, ensure_ascii=False), personality)
        )
        _db_conn.commit()
    except Exception as e:
        logger.exception("Failed to write chat log to DB: %s", e)

def detect_credit_error(text: str) -> bool:
    if not text:
        return False
    s = text.lower()
    keywords = ["credit", "quota", "limit", "insufficient", "balance", "payment required", "billing", "quota exceeded", "credit limit", "insufficient credits"]
    return any(k in s for k in keywords)

# Basic scoring for ensemble selection
def response_score(candidate_text: str) -> float:
    if not candidate_text:
        return 0.0
    s = 0.1
    t = candidate_text.strip()
    l = len(t)
    if l < 20:
        s += 0.2
    elif l < 200:
        s += 0.6
    else:
        s += 0.5
    if any(w in t.lower() for w in ["error", "unauthorized", "invalid", "failed", "sorry"]):
        s -= 0.4
    if any(p in t for p in ".!?"):
        s += 0.05
    return s

# Robust response extraction (tries common shapes)
def extract_text_from_provider_response(data: Any) -> Optional[str]:
    try:
        if not data:
            return None
        # OpenAI-like: choices[0].message.content
        if isinstance(data, dict):
            choices = data.get("choices") or data.get("results")
            if choices and isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                if isinstance(first, dict):
                    # common shapes
                    m = (first.get("message") or {})
                    if isinstance(m, dict) and m.get("content"):
                        return m.get("content")
                    if first.get("text"):
                        return first.get("text")
                    if first.get("content"):
                        return first.get("content")
            # google-style: candidates -> content/parts
            if "candidates" in data:
                cand = data.get("candidates", [])
                if isinstance(cand, list) and len(cand) > 0:
                    c0 = cand[0]
                    # try several shapes
                    if isinstance(c0, str):
                        return c0
                    if isinstance(c0, dict):
                        # try nested content.parts[].text
                        cont = c0.get("content") or c0.get("message") or c0
                        if isinstance(cont, dict):
                            parts = cont.get("parts") or cont.get("text") or []
                            if isinstance(parts, list) and len(parts) > 0:
                                if isinstance(parts[0], dict) and parts[0].get("text"):
                                    return parts[0].get("text")
                                if isinstance(parts[0], str):
                                    return parts[0]
                        # fallback to string form fields
                        for k in ("text", "output", "content"):
                            if cont.get(k):
                                return cont.get(k)
            # fallback top-level
            for key in ("response","output_text","text"):
                if data.get(key):
                    return data.get(key)
    except Exception:
        pass
    return None

# ---------- Provider Query ----------
async def query_provider_once(provider_key: str, messages: list, timeout: int = 30, max_tokens: int = DEFAULT_MAX_TOKENS) -> Dict[str, Any]:
    """
    Query a single provider. Returns a dict:
    { provider, ok (bool), response (str), error (str), raw (any), status (int) }
    """
    endpoints = {
        "openrouter": {
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "model": "openai/gpt-4-turbo",
            "headers": {"Authorization": f"Bearer {API_KEYS.get('openrouter')}"}
        },
        "together": {
            "url": "https://api.together.xyz/v1/chat/completions",
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "headers": {"Authorization": f"Bearer {API_KEYS.get('together')}"}
        },
        "groq": {
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "model": "llama3-70b-8192",
            "headers": {"Authorization": f"Bearer {API_KEYS.get('groq')}"}
        },
        "gemini": {
            # Using Google Generative Language endpoint shape (key in URL). This may vary by API version.
            "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={API_KEYS.get('gemini')}",
            "model": "gemini-pro",
            "headers": {"Content-Type": "application/json"}
        },
            "deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-chat",
        "headers": {"Authorization": f"Bearer {API_KEYS.get('deepseek')}"}
    }
    }

    if provider_key not in endpoints:
        return {"provider": provider_key, "ok": False, "response": "", "error": "Provider not configured", "raw": None, "status": None}

    cfg = endpoints[provider_key]

    # Build payload per-provider
    if provider_key == "gemini":
        # flatten messages into a single text for Gemini's content API
        full_text = "\n".join([m.get("content", "") for m in messages])
        payload = {
            "contents": [{"parts": [{"text": full_text}]}],
            "temperature": 0.3 if provider_key == "deepseek" else 0.7,
        "maxOutputTokens": max_tokens
        }
    else:
        payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.3 if provider_key == "deepseek" else 0.7,
        "max_tokens": max_tokens
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(cfg["url"], headers=cfg.get("headers", {}), json=payload)
            text_body = resp.text
            status_code = resp.status_code

            # handle non-2xx
            if status_code >= 400:
                # detect credit/quota/billing problems
                is_credit = detect_credit_error(text_body) or status_code in (402, 429)
                err_msg = f"HTTP {status_code} - {text_body[:300]}"
                logger.warning("Provider %s returned error status %s", provider_key, status_code)
                return {"provider": provider_key, "ok": False, "response": "", "error": err_msg, "raw": text_body, "status": status_code, "is_credit": is_credit}

            # try to parse JSON
            try:
                data = resp.json()
            except Exception:
                # maybe plain text
                data = None

            # try extracting
            content = None
            if data:
                content = extract_text_from_provider_response(data)

            # fallback: if gemini and we didn't parse json well, attempt to parse heuristically
            if not content and provider_key == "gemini":
                # some Gemini responses might embed text in 'candidates'
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        cand = data.get("candidates") or []
                        if cand:
                            # attempt several shapes
                            c0 = cand[0]
                            if isinstance(c0, dict):
                                # nested 'content' -> 'parts' -> text
                                cont = c0.get("content") or {}
                                parts = cont.get("parts") or []
                                if parts and isinstance(parts[0], dict) and parts[0].get("text"):
                                    content = parts[0].get("text")
                except Exception:
                    pass

            # final fallback: raw text
            if not content and text_body:
                # sometimes providers return plain text
                content = text_body.strip()

            if not content:
                return {"provider": provider_key, "ok": False, "response": "", "error": "Empty content", "raw": data or text_body, "status": status_code}

            return {"provider": provider_key, "ok": True, "response": str(content).strip(), "error": None, "raw": data or text_body, "status": status_code}

    except Exception as e:
        logger.exception("Error querying provider %s: %s", provider_key, str(e))
        return {"provider": provider_key, "ok": False, "response": "", "error": str(e), "raw": None, "status": None}

# ---------- Aggregation / Fallback ----------
def select_best_response(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    # candidates is a list of {provider, ok, response, ...}
    valid = [c for c in candidates if c.get("ok") and c.get("response")]
    if not valid:
        # return first error candidate for debugging
        return {"ok": False, "candidates": candidates}
    # score and pick
    for c in valid:
        c["_score"] = response_score(c["response"])
    valid.sort(key=lambda x: x["_score"], reverse=True)
    best = valid[0]
    return {"ok": True, "provider": best["provider"], "response": best["response"], "candidates": candidates, "score": best.get("_score")}

async def fetch_ai_response(provider: str, messages: list, timeout: int = 60, max_tokens: int = DEFAULT_MAX_TOKENS):
    """
    If provider == 'ensemble', queries all available providers and chooses best.
    If provider is a single provider, tries it, and automatically falls back to others on errors (credit/quota/timeouts).
    Returns: { response, final_provider, candidates }
    """
    # build list of available providers (in fallback order) based on API keys
    available = [p for p in PROVIDER_FALLBACK_ORDER if API_KEYS.get(p)]
    if not available:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="No AI providers configured")

    # Ensemble mode
    if provider == "ensemble":
        tasks = [query_provider_once(p, messages, timeout, max_tokens) for p in available]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        # pick best
        best = select_best_response(results)
        if not best.get("ok"):
            logger.error("Ensemble produced no valid responses: %s", best.get("candidates"))
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Ensemble failed to produce content")
        return {"response": best["response"], "final_provider": best["provider"], "candidates": results}

    # Single-provider with fallback
    if provider not in available:
        # user asked for a provider that's not available/has no key
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Requested provider '{provider}' not configured or missing API key")

    tried = []
    order = [provider] + [p for p in available if p != provider]
    for p in order:
        res = await query_provider_once(p, messages, timeout, max_tokens)
        tried.append(res)
        # if successful, return it
        if res.get("ok"):
            return {"response": res["response"], "final_provider": p, "candidates": tried}
        # if error but not credit-related, we still continue to next
        # if credit-related or other errors -> continue to next provider automatically
        # loop continues until someone succeeds

    # none succeeded
    logger.error("All providers failed for request. Tried: %s", [t.get("provider") for t in tried])
    # Give client useful debug info (not full raw) but log raw to server logs
    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="All AI providers failed")

# ---------- API Routes ----------
@app.post("/api/chat", response_model=dict, tags=["AI Services"])
async def chat_handler(request: ChatRequest):
    try:
        messages = build_prompt_context(request.message, request.personality)
        result = await fetch_ai_response(request.provider, messages, timeout=60, max_tokens=DEFAULT_MAX_TOKENS)

        final_provider = result.get("final_provider")
        candidates = result.get("candidates", [])
        ai_text = result.get("response", "")

        # Log to SQLite DB (your existing code)
        try:
            log_chat_db(request.message, final_provider or request.provider, ai_text, request.provider, candidates, request.personality)
        except Exception:
            logger.exception("Failed to log chat to DB")

        # Save chat to Supabase
        user_id = "guest"  # Replace with actual user identifier if available
        try:
            supabase.table("chat_history").insert({
                "user_id": user_id,
                "role": "user",
                "message": request.message
            }).execute()
            supabase.table("chat_history").insert({
                "user_id": user_id,
                "role": "assistant",
                "message": ai_text
            }).execute()
        except Exception as e:
            logger.error(f"Supabase insert failed: {e}")

        return {
            "success": True,
            "response": ai_text,
            "final_provider": final_provider,
            "metadata": {
                "provider_requested": request.provider,
                "personality": request.personality
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred during chat processing")

@app.post("/api/tools/{tool}", response_model=dict, tags=["AI Services"])
async def tool_handler(tool: str, request: ToolRequest):
    tool_prompts = {
        "summarize": "Provide a concise summary of the following content. Focus on key points and main ideas.",
        "translate": "Translate the following text to the specified language. Maintain original meaning and tone.",
        "analyze": "Analyze this content and provide key insights, patterns, and important information.",
        "search": "Process these search results and extract the most relevant information."
    }
    if tool not in tool_prompts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported tool specified")
    try:
        messages = [
            {"role": "system", "content": tool_prompts[tool]},
            {"role": "user", "content": request.content}
        ]
        result = await fetch_ai_response(request.provider, messages)
        final_provider = result.get("final_provider")
        # log
        try:
            log_chat_db(request.content, final_provider or request.provider, result.get("response",""), request.provider, [], "tool-"+tool)
        except Exception:
            logger.exception("Failed to log tool call")
        return {"success": True, "result": result.get("response",""), "final_provider": final_provider}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Tool error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing {tool} request")

@app.post("/api/search", response_model=dict, tags=["AI Services"])
async def search_handler(request: SearchRequest):
    if not API_KEYS.get("serper"):
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Search functionality not configured")
    try:
        headers = {"X-API-KEY": API_KEYS["serper"], "Content-Type": "application/json"}
        payload = {"q": request.query, "gl": "us", "hl": "en", "num": request.num_results}
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post("https://google.serper.dev/search", headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()
        organic_results = data.get("organic", [])
        if request.provider:
            context = "\n".join([f"Title: {r.get('title')}\nLink: {r.get('link')}\nSnippet: {r.get('snippet','')}" for r in organic_results[:5]])
            ai_result = await fetch_ai_response(request.provider, [{"role":"system","content":"Summarize these search results concisely."},{"role":"user","content":context}])
            return {"success": True, "results": organic_results, "summary": ai_result.get("response",""), "final_provider": ai_result.get("final_provider")}
        return {"success": True, "results": organic_results, "processed": False}
    except httpx.HTTPStatusError as e:
        logger.error("Search API error: %s %s", e.response.status_code, e.response.text)
        raise HTTPException(status_code=e.response.status_code, detail="Search service error")
    except Exception as e:
        logger.exception("Search error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing search request")

@app.post("/api/tts", tags=["AI Services"])
async def tts_handler(request: TTSRequest):
    if not clients.get("elevenlabs"):
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="TTS service not configured")
    try:
        audio = clients["elevenlabs"].generate(text=request.text, voice=request.voice, model=request.model)
        return Response(content=audio, media_type="audio/mpeg", headers={"Content-Disposition":"inline; filename=quainex-tts.mp3"})
    except Exception as e:
        logger.exception("TTS error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error generating speech")

@app.post("/api/images", response_model=dict, tags=["AI Services"])
async def image_gen_handler(request: ImageGenRequest):
    if not clients.get("groq"):
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Image generation not configured")
    try:
        response = clients["groq"].images.generate(model="dall-e-3", prompt=request.prompt, size=request.size, quality=request.quality, style=request.style, n=1)
        return {"success": True, "image_url": response.data[0].url, "revised_prompt": response.data[0].revised_prompt}
    except Exception:
        logger.exception("Image generation error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error generating image")

@app.post("/api/voice", response_model=dict, tags=["AI Services"])
async def voice_handler(file: UploadFile = File(...), request: VoiceRequest = None):
    logger.info("Received voice file: %s (%s)", file.filename, file.content_type)
    return {"success": True, "message": "Voice transcription service placeholder", "details": "In production, this would return transcribed text."}

# ---------- Metadata / Utility Endpoints ----------
@app.get("/api/models", response_model=dict, tags=["Metadata"])
async def list_models():
    return {
        "openrouter": [{"id":"openai/gpt-4-turbo","name":"GPT-4 Turbo"}],
        "together": [{"id":"mistralai/Mixtral-8x7B-Instruct-v0.1","name":"Mixtral 8x7B"}],
        "groq": [{"id":"llama3-70b-8192","name":"Llama 3 70B"}],
        "gemini": [{"id":"gemini-pro","name":"Gemini Pro (Google)"}]
    }

@app.get("/api/personalities", response_model=dict, tags=["Metadata"])
async def list_personalities():
    return {"personalities":[{"id":k,"name":v["name"]} for k,v in PERSONALITIES.items()]}

@app.get("/api/status", response_model=dict, tags=["Metadata"])
async def service_status():
    return {
        "status": "operational",
        "version": "1.0.0",
        "features": {
            "chat": bool(API_KEYS.get("openrouter") or API_KEYS.get("together") or API_KEYS.get("groq") or API_KEYS.get("gemini")),
            "tts": bool(API_KEYS.get("elevenlabs")),
            "image_generation": bool(API_KEYS.get("groq")),
            "search": bool(API_KEYS.get("serper"))
        },
        "uptime": str(datetime.now())
    }

# ---------- Error Handlers ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "error": exc.detail, "code": exc.status_code})

@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception):
    logger.exception("Server error")
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"success": False, "error": "Internal server error", "code": status.HTTP_500_INTERNAL_SERVER_ERROR})

@app.on_event("startup")
async def cleanup_old_history():
    thirty_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)
    supabase.table("chat_history")\
        .delete()\
        .lt("created_at", thirty_days_ago.isoformat())\
        .execute()
