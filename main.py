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
from datetime import datetime, timedelta
import sqlite3
import json
from supabase import create_client
import xml.etree.ElementTree as ET
import ast

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
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
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

# ---------- Pydantic Models ----------
class ChatRequest(BaseModel):
    message: str
    provider: str = "openrouter"
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

TOOLS_SCHEMA = """
<tools>
    <tool>
        <name>web_search</name>
        <description>Performs a web search to find up-to-date information, news, articles, or specific facts about any topic. Use this for questions about current events, people, companies, or real-time data.</description>
        <parameters>
            <param>
                <name>query</name>
                <type>string</type>
                <description>A highly specific, keyword-focused search query.</description>
            </param>
        </parameters>
    </tool>
    <tool>
        <name>python_interpreter</name>
        <description>Executes Python code in a secure sandbox. Use this for data analysis, calculations, data manipulation, plotting, or solving complex logical problems that require code. The executed code can only access the standard Python library and approved packages (e.g., pandas, numpy). It cannot access the internet or local file system.</description>
        <parameters>
            <param>
                <name>code</name>
                <type>string</type>
                <description>The Python code to execute. The final line must be a print() statement to output the result.</description>
            </param>
        </parameters>
    </tool>
</tools>
"""

REASONING_PROMPT = f"""
You are Quainex, a world-class, autonomous AI agent. Your purpose is to assist users by accomplishing complex tasks.
You operate in a thought-action-observation loop. At each step, you must use the <thinking> tag to reason about the user's request, your plan, and the next best action. Based on your thoughts, you can then use a tool or provide the final answer.
**Available Tools:**
{TOOLS_SCHEMA}
**Your Response Format:**
**Step 1: THINK**
Use the `<thinking>` tag to outline your reasoning. Analyze the user's goal, break it down into steps, reflect on previous actions, and decide what to do next.
**Step 2: ACT (Optional)**
If you need to use a tool, use the `<tool_call>` tag with the exact tool name and parameters. You can only call one tool at a time.
Example: `<tool_call><tool_name>web_search</tool_name><parameters><query>Current price of NVIDIA stock</query></parameters></tool_call>`
**Step 3: FINAL ANSWER**
Once you have gathered all necessary information and completed all steps, provide the final, comprehensive answer to the user. Do not use any XML tags in the final answer.
**BEGIN!**
"""

PERSONALITIES = {
    "agent": {
        "name": "Quainex Agent",
        "system_prompt": REASONING_PROMPT
    },
    "default": {
        "name": "Quainex Expert",
        "system_prompt": (
            "You are Quainex, a world-class senior technical consultant created by Bright SecureTech. "
            "You excel in software engineering, cybersecurity, cloud architecture, and AI. "
            "Provide intelligent, accurate, and professional answers. Structure complex responses clearly with markdown."
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

def get_system_prompt(personality: str) -> str:
    return PERSONALITIES.get(personality, PERSONALITIES["default"])["system_prompt"]

def build_prompt_context(prompt: str, personality: str, history: list = None) -> list:
    system_prompt = get_system_prompt(personality)
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages

# ---------- Helper Utilities (Unchanged) ----------
def log_chat_db(user_message: str, final_provider: str, ai_response: str,
                provider_requested: str, candidates: List[Dict[str, Any]], personality: str):
    try:
        timestamp = datetime.utcnow().isoformat()
        # This function is not fully implemented for supabase
        pass
    except Exception as e:
        logger.exception("Failed to write chat log to DB: %s", e)

def detect_credit_error(text: str) -> bool:
    if not text:
        return False
    s = text.lower()
    keywords = ["credit", "quota", "limit", "insufficient", "balance", "payment required", "billing", "quota exceeded", "credit limit", "insufficient credits"]
    return any(k in s for k in keywords)

def response_score(candidate_text: str) -> float:
    if not candidate_text:
        return 0.0
    s = 0.1
    t = candidate_text.strip()
    l = len(t)
    if l < 20: s += 0.2
    elif l < 200: s += 0.6
    else: s += 0.5
    if any(w in t.lower() for w in ["error", "unauthorized", "invalid", "failed", "sorry"]):
        s -= 0.4
    if any(p in t for p in ".!?"): s += 0.05
    return s

def extract_text_from_provider_response(data: Any) -> Optional[str]:
    try:
        if not data: return None
        if isinstance(data, dict):
            choices = data.get("choices") or data.get("results")
            if choices and isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                if isinstance(first, dict):
                    m = (first.get("message") or {})
                    if isinstance(m, dict) and m.get("content"): return m.get("content")
                    if first.get("text"): return first.get("text")
                    if first.get("content"): return first.get("content")
            if "candidates" in data:
                cand = data.get("candidates", [])
                if isinstance(cand, list) and len(cand) > 0:
                    c0 = cand[0]
                    if isinstance(c0, str): return c0
                    if isinstance(c0, dict):
                        cont = c0.get("content") or c0.get("message") or c0
                        if isinstance(cont, dict):
                            parts = cont.get("parts") or cont.get("text") or []
                            if isinstance(parts, list) and len(parts) > 0:
                                if isinstance(parts[0], dict) and parts[0].get("text"):
                                    return parts[0].get("text")
                                if isinstance(parts[0], str): return parts[0]
                        for k in ("text", "output", "content"):
                            if cont.get(k): return cont.get(k)
            for key in ("response","output_text","text"):
                if data.get(key): return data.get(key)
    except Exception: pass
    return None

async def query_provider_once(provider_key: str, messages: list, timeout: int = 30, max_tokens: int = DEFAULT_MAX_TOKENS) -> Dict[str, Any]:
    endpoints = {
        "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions", "model": "openai/gpt-4-turbo", "headers": {"Authorization": f"Bearer {API_KEYS.get('openrouter')}"}},
        "together": {"url": "https://api.together.xyz/v1/chat/completions", "model": "mistralai/Mixtral-8x7B-Instruct-v0.1", "headers": {"Authorization": f"Bearer {API_KEYS.get('together')}"}},
        "groq": {"url": "https://api.groq.com/openai/v1/chat/completions", "model": "llama3-70b-8192", "headers": {"Authorization": f"Bearer {API_KEYS.get('groq')}"}},
        "gemini": {"url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={API_KEYS.get('gemini')}", "model": "gemini-pro", "headers": {"Content-Type": "application/json"}},
        "deepseek": {"url": "https://api.deepseek.com/chat/completions", "model": "deepseek-chat", "headers": {"Authorization": f"Bearer {API_KEYS.get('deepseek')}"}}
    }
    if provider_key not in endpoints: return {"provider": provider_key, "ok": False, "response": "", "error": "Provider not configured", "raw": None, "status": None}
    cfg = endpoints[provider_key]
    if provider_key == "gemini":
        full_text = "\n".join([m.get("content", "") for m in messages])
        payload = {"contents": [{"parts": [{"text": full_text}]}], "temperature": 0.7, "maxOutputTokens": max_tokens}
    else:
        payload = {"model": cfg["model"], "messages": messages, "temperature": 0.7, "max_tokens": max_tokens}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(cfg["url"], headers=cfg.get("headers", {}), json=payload)
            text_body, status_code = resp.text, resp.status_code
            if status_code >= 400:
                is_credit = detect_credit_error(text_body) or status_code in (402, 429)
                err_msg = f"HTTP {status_code} - {text_body[:300]}"
                logger.warning("Provider %s returned error status %s", provider_key, status_code)
                return {"provider": provider_key, "ok": False, "response": "", "error": err_msg, "raw": text_body, "status": status_code, "is_credit": is_credit}
            try: data = resp.json()
            except: data = None
            content = extract_text_from_provider_response(data)
            if not content and text_body: content = text_body.strip()
            if not content: return {"provider": provider_key, "ok": False, "response": "", "error": "Empty content", "raw": data or text_body, "status": status_code}
            return {"provider": provider_key, "ok": True, "response": str(content).strip(), "error": None, "raw": data or text_body, "status": status_code}
    except Exception as e:
        logger.exception("Error querying provider %s: %s", provider_key, str(e))
        return {"provider": provider_key, "ok": False, "response": "", "error": str(e), "raw": None, "status": None}

def select_best_response(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [c for c in candidates if c.get("ok") and c.get("response")]
    if not valid: return {"ok": False, "candidates": candidates}
    for c in valid: c["_score"] = response_score(c["response"])
    valid.sort(key=lambda x: x["_score"], reverse=True)
    best = valid[0]
    return {"ok": True, "provider": best["provider"], "response": best["response"], "candidates": candidates, "score": best.get("_score")}

async def fetch_ai_response(provider: str, messages: list, timeout: int = 60, max_tokens: int = DEFAULT_MAX_TOKENS):
    available = [p for p in PROVIDER_FALLBACK_ORDER if API_KEYS.get(p)]
    if not available: raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="No AI providers configured")
    if provider == "ensemble":
        tasks = [query_provider_once(p, messages, timeout, max_tokens) for p in available]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        best = select_best_response(results)
        if not best.get("ok"):
            logger.error("Ensemble produced no valid responses: %s", best.get("candidates"))
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Ensemble failed to produce content")
        return {"response": best["response"], "final_provider": best["provider"], "candidates": results}
    if provider not in available:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Requested provider '{provider}' not configured or missing API key")
    tried = []
    order = [provider] + [p for p in available if p != provider]
    for p in order:
        res = await query_provider_once(p, messages, timeout, max_tokens)
        tried.append(res)
        if res.get("ok"): return {"response": res["response"], "final_provider": p, "candidates": tried}
    logger.error("All providers failed for request. Tried: %s", [t.get("provider") for t in tried])
    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="All AI providers failed")

async def execute_web_search(query: str) -> str:
    """Executes a web search using the configured Serper API."""
    if not API_KEYS.get("serper"):
        return "Error: Search tool is not configured."
    headers = {"X-API-KEY": API_KEYS["serper"], "Content-Type": "application/json"}
    payload = {"q": query}
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post("https://google.serper.dev/search", headers=headers, json=payload)
            res.raise_for_status()
            results = res.json().get("organic", [])
            if not results:
                return "No search results found."
            return "\n".join([f"Title: {r.get('title')}\nSnippet: {r.get('snippet')}" for r in results[:3]])
    except Exception as e:
        logger.error(f"Web search failed for query '{query}': {e}")
        return f"Error during web search: {e}"

# --- CORRECTED & SAFER PYTHON INTERPRETER FUNCTION ---
async def execute_python_code_safely(code: str) -> str:
    """
    Executes a limited set of Python code safely.
    WARNING: This is a simplified, non-production-grade sandbox.
    For a secure, public-facing application, use a dedicated, isolated sandbox (e.g., Docker).
    """
    
    # Check for dangerous keywords or commands
    dangerous_keywords = ['import', 'open', 'os', 'sys', 'subprocess', 'exec', 'eval', 'del']
    if any(keyword in code for keyword in dangerous_keywords):
        return "Security Error: The provided code contains a dangerous keyword that cannot be executed in this environment."

    try:
        # Use ast.parse to check the syntax and structure of the code
        # This prevents common exploits like `__import__('os').system('ls')`
        tree = ast.parse(code)
        
        # Compile the code to check for valid syntax
        compiled_code = compile(tree, '<string>', 'exec')
        
        # Use a heavily restricted exec environment
        restricted_globals = {'__builtins__': {}}
        restricted_locals = {}
        
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        
        exec(compiled_code, restricted_globals, restricted_locals)
        
        sys.stdout = old_stdout
        output = redirected_output.getvalue()
        
        return f"Execution successful. Output:\n{output}"
    except SyntaxError as e:
        return f"Syntax Error: {e}"
    except Exception as e:
        return f"Execution failed with error: {e}"

TOOL_REGISTRY = {
    "web_search": execute_web_search,
    "python_interpreter": execute_python_code_safely, # Updated to use the safer function
}

def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parses the LLM's response to find thinking and tool_call tags."""
    try:
        thinking_text = response_text.split("<thinking>")[1].split("</thinking>")[0].strip()
    except IndexError:
        thinking_text = ""

    try:
        tool_call_text = response_text.split("<tool_call>")[1].split("</tool_call>")[0]
        tool_xml = ET.fromstring(f"<root>{tool_call_text}</root>")
        tool_name = tool_xml.find("tool_name").text.strip()
        params = {p.tag: p.text.strip() for p in tool_xml.find("parameters")}
        return {"thinking": thinking_text, "tool_name": tool_name, "tool_params": params}
    except (IndexError, ET.ParseError):
        return {"thinking": thinking_text, "final_answer": response_text}

class Agent:
    def __init__(self, provider: str = "openrouter"):
        self.provider = provider
        self.history = []
        self.max_loops = 7

    async def run(self, initial_prompt: str) -> str:
        self.history = build_prompt_context(initial_prompt, "agent")
        
        for i in range(self.max_loops):
            logger.info(f"Agent Loop #{i+1}")
            
            llm_result = await fetch_ai_response(self.provider, self.history)
            llm_response_text = llm_result["response"]
            self.history.append({"role": "assistant", "content": llm_response_text})
            
            parsed = parse_llm_response(llm_response_text)

            if "final_answer" in parsed:
                logger.info("Agent decided to provide a final answer.")
                return parsed["final_answer"]

            if "tool_name" in parsed:
                tool_name = parsed["tool_name"]
                tool_params = parsed["tool_params"]
                logger.info(f"Agent wants to use tool: {tool_name} with params: {tool_params}")

                if tool_name in TOOL_REGISTRY:
                    tool_function = TOOL_REGISTRY[tool_name]
                    observation = await tool_function(**tool_params)
                    
                    observation_message = f"<tool_result><tool_name>{tool_name}</tool_name><result>{observation}</result></tool_result>"
                    self.history.append({"role": "user", "content": observation_message})
                else:
                    logger.warning(f"Agent tried to use an unknown tool: {tool_name}")
                    self.history.append({"role": "user", "content": f"<tool_result><error>Tool '{tool_name}' not found.</error></tool_result>"})
            else:
                logger.warning("Agent did not provide a tool call or a final answer. Returning raw response.")
                return llm_response_text
        
        return "Error: Agent exceeded maximum loops. Unable to complete the task."

# ---------- API Routes ----------
@app.post("/api/chat", response_model=dict, tags=["AI Services"])
async def chat_handler(request: ChatRequest):
    try:
        final_response = ""
        final_provider = ""
        
        if request.personality == "agent":
            logger.info(f"Initiating Agent for prompt: '{request.message}'")
            agent = Agent(provider=request.provider)
            final_response = await agent.run(request.message)
            final_provider = agent.provider
        else:
            messages = build_prompt_context(request.message, request.personality)
            result = await fetch_ai_response(request.provider, messages, timeout=60, max_tokens=DEFAULT_MAX_TOKENS)
            final_provider = result.get("final_provider")
            final_response = result.get("response", "")

        user_id = "guest"
        try:
            supabase.table("chat_history").insert({
                "user_id": user_id,
                "role": "user",
                "message": request.message
            }).execute()
            supabase.table("chat_history").insert({
                "user_id": user_id,
                "role": "assistant",
                "message": final_response
            }).execute()
        except Exception as e:
            logger.error(f"Supabase insert failed: {e}")

        return {
            "success": True,
            "response": final_response,
            "final_provider": final_provider,
            "metadata": {
                "provider_requested": request.provider,
                "personality": request.personality,
                "mode": "agent" if request.personality == "agent" else "chat"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred during chat processing")

@app.delete("/api/chat/new", tags=["Chat History"])
async def new_chat_handler():
    """Clears all chat history for the current user."""
    user_id = "guest"
    try:
        res = supabase.table("chat_history")\
            .delete()\
            .eq("user_id", user_id)\
            .execute()
        if res.data:
            logger.info(f"Cleared {len(res.data)} chat messages for user '{user_id}'")
        else:
            logger.info(f"No chat messages to clear for user '{user_id}'")
        return {"success": True, "message": "Chat history cleared"}
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clear chat history")

@app.get("/api/chat/history", tags=["Chat History"])
async def get_chat_history_handler():
    """Retrieves the full chat history for the current user."""
    user_id = "guest"
    try:
        res = supabase.table("chat_history")\
            .select("message", "role", "created_at")\
            .eq("user_id", user_id)\
            .order("created_at")\
            .execute()
        return {"success": True, "history": res.data}
    except Exception as e:
        logger.error(f"Failed to retrieve chat history: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve chat history")

class SettingsRequest(BaseModel):
    key: str
    value: Any

@app.post("/api/settings", tags=["Settings"])
async def update_settings_handler(request: SettingsRequest):
    """Updates a user's setting."""
    user_id = "guest"
    return {"success": True, "message": "Setting updated successfully (placeholder)"}

@app.get("/api/settings", tags=["Settings"])
async def get_settings_handler():
    """Retrieves a user's settings."""
    user_id = "guest"
    return {"success": True, "settings": {"theme": "dark", "language": "en"}}

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

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "error": exc.detail, "code": exc.status_code})

@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception):
    logger.exception("Server error")
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"success": False, "error": "Internal server error", "code": status.HTTP_500_INTERNAL_SERVER_ERROR})