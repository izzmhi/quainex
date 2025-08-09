# main.py
from fastapi import FastAPI, Request, HTTPException, status, Response, UploadFile, File, Body
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
import re
from fastapi.encoders import jsonable_encoder

# ---------- Configuration ----------
load_dotenv()

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quainex.log')
    ]
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

# Initialize FastAPI app
app = FastAPI(
    title="Quainex AI API",
    description="Premium AI Assistant Backend Service",
    version="2.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    contact={
        "name": "API Support",
        "email": "support@quainex.ai"
    }
)

# Initialize Supabase client
SUPABASE_URL = API_KEYS["supabaseurl"]
SUPABASE_KEY = API_KEYS["supabasekey"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://quainexai.onrender.com",
        "https://quainex.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
    expose_headers=["*"],
    max_age=600
)

# Startup cleanup
@app.on_event("startup")
async def startup_cleanup():
    """Cleanup old chat history on startup"""
    try:
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        res = supabase.table("chat_history") \
            .delete() \
            .lt("created_at", thirty_days_ago.isoformat()) \
            .execute()
        logger.info(f"Cleaned up {len(res.data)} old chat records")
    except Exception as e:
        logger.error(f"Startup cleanup failed: {str(e)}")

# Clients initialization
clients = {
    "elevenlabs": ElevenLabs(api_key=API_KEYS["elevenlabs"]) if API_KEYS.get("elevenlabs") else None,
    "groq": Groq(api_key=API_KEYS["groq"]) if API_KEYS.get("groq") else None
}

# Constants
DEFAULT_MAX_TOKENS = 500
PROVIDER_FALLBACK_ORDER = ["openrouter", "together", "groq", "deepseek", "gemini"]
MAX_AGENT_LOOPS = 5

# Add root endpoint
@app.get("/")
async def root():
    return JSONResponse(
        content={
            "status": "running",
            "service": "Quainex AI Backend",
            "version": "2.1.0",
            "docs": "/api/docs",
            "available_endpoints": [
                "/api/chat",
                "/voice",
                "/api/providers",
                "/health"
            ]
        },
        headers={
            "Access-Control-Allow-Origin": "https://quainexai.onrender.com",
            "Access-Control-Allow-Credentials": "true"
        }
    )

# Explicit OPTIONS handler for preflight requests
@app.options("/api/chat")
async def options_chat():
    return Response(
        status_code=status.HTTP_200_OK,
        headers={
            "Access-Control-Allow-Origin": "https://quainexai.onrender.com",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Credentials": "true"
        }
    )

# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    response.headers["X-Response-Time"] = str(process_time)
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response

# ---------- Data Models ----------
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

class DeveloperAPIKey(BaseModel):
    email: str
    company: Optional[str] = None

# ---------- Core AI Components ----------
TOOLS_SCHEMA = """
<tools>
    <tool>
        <name>web_search</name>
        <description>Performs a web search to find up-to-date information</description>
        <parameters>
            <param>
                <name>query</name>
                <type>string</type>
                <description>Search query</description>
            </param>
        </parameters>
    </tool>
    <tool>
        <name>python_interpreter</name>
        <description>Executes Python code in a secure sandbox</description>
        <parameters>
            <param>
                <name>code</name>
                <type>string</type>
                <description>Python code to execute</description>
            </param>
        </parameters>
    </tool>
</tools>
"""

REASONING_PROMPT = f"""
You are Quainex, an advanced AI assistant. Follow this process:

1. THINK: Analyze the request using <thinking> tags
2. ACT: Use <tool_call> tags if you need to use a tool
3. ANSWER: Provide the final answer without XML tags

Available Tools:
{TOOLS_SCHEMA}

Example Tool Call:
<tool_call>
    <tool_name>web_search</tool_name>
    <parameters>
        <query>Current weather in New York</query>
    </parameters>
</tool_call>

Example Final Answer:
The current weather in New York is 72°F and sunny.

Now begin!
"""

PERSONALITIES = {
    "agent": {
        "name": "Quainex Agent",
        "system_prompt": REASONING_PROMPT
    },
    "default": {
        "name": "Standard Assistant",
        "system_prompt": "You are Quainex, a helpful AI assistant. Provide clear, accurate responses."
    },
    "technical": {
        "name": "Technical Expert",
        "system_prompt": "You are Quainex, a technical expert. Provide precise, detailed answers."
    }
}

# ---------- Helper Functions ----------
def get_system_prompt(personality: str) -> str:
    """Get the system prompt for the specified personality"""
    return PERSONALITIES.get(personality, PERSONALITIES["default"])["system_prompt"]

def build_prompt_context(prompt: str, personality: str, history: list = None) -> list:
    """Build the conversation context with personality and history"""
    system_prompt = get_system_prompt(personality)
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages

async def query_provider(provider: str, messages: list, timeout: int = 30) -> dict:
    """Query a specific AI provider"""
    endpoints = {
        "openrouter": {
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "headers": {
                "Authorization": f"Bearer {API_KEYS['openrouter']}",
                "HTTP-Referer": "https://quainexai.onrender.com",
                "X-Title": "Quainex AI"
            },
            "payload": {
                "model": "openai/gpt-4-turbo",
                "messages": messages,
                "temperature": 0.7
            }
        },
        "groq": {
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "headers": {
                "Authorization": f"Bearer {API_KEYS['groq']}"
            },
            "payload": {
                "model": "llama3-70b-8192",
                "messages": messages,
                "temperature": 0.7
            }
        },
        "deepseek": {
            "url": "https://api.deepseek.com/v1/chat/completions",
            "headers": {
                "Authorization": f"Bearer {API_KEYS['deepseek']}"
            },
            "payload": {
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.7
            }
        }
    }
    
    if provider not in endpoints:
        return {
            "ok": False,
            "error": f"Provider {provider} not configured",
            "provider": provider
        }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoints[provider]["url"],
                headers=endpoints[provider]["headers"],
                json=endpoints[provider]["payload"]
            )
            
            if response.status_code != 200:
                error_msg = f"Provider {provider} returned {response.status_code}"
                logger.error(f"{error_msg}: {response.text[:200]}")
                return {
                    "ok": False,
                    "error": error_msg,
                    "status_code": response.status_code,
                    "provider": provider
                }
            
            data = response.json()
            if not data.get("choices"):
                return {
                    "ok": False,
                    "error": "Invalid response format",
                    "provider": provider,
                    "raw": data
                }
            
            return {
                "ok": True,
                "response": data["choices"][0]["message"]["content"],
                "provider": provider,
                "raw": data
            }
            
    except Exception as e:
        logger.error(f"Error querying {provider}: {str(e)}")
        return {
            "ok": False,
            "error": str(e),
            "provider": provider
        }

async def fetch_ai_response(provider: str, messages: list, timeout: int = 60) -> dict:
    """Get AI response with fallback logic"""
    if provider not in PROVIDER_FALLBACK_ORDER:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider: {provider}. Available: {PROVIDER_FALLBACK_ORDER}"
        )
    
    # Try requested provider first
    result = await query_provider(provider, messages, timeout)
    if result["ok"]:
        return {
            "success": True,
            "response": result["response"],
            "provider": provider
        }
    
    # Fallback to other providers
    for fallback in [p for p in PROVIDER_FALLBACK_ORDER if p != provider]:
        if not API_KEYS.get(fallback):
            continue
            
        result = await query_provider(fallback, messages, timeout)
        if result["ok"]:
            return {
                "success": True,
                "response": result["response"],
                "provider": fallback,
                "fallback_used": True
            }
    
    # All providers failed
    raise HTTPException(
        status_code=502,
        detail="All AI providers failed to respond"
    )

# ---------- Tool Implementations ----------
async def execute_web_search(query: str) -> str:
    """Perform a web search using Serper API"""
    if not API_KEYS.get("serper"):
        return "Error: Search API not configured"
    
    try:
        headers = {"X-API-KEY": API_KEYS["serper"]}
        payload = {"q": query, "gl": "us", "hl": "en"}
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://google.serper.dev/search",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("organic", [])[:3]:
                results.append(
                    f"Title: {item.get('title', 'N/A')}\n"
                    f"Link: {item.get('link', 'N/A')}\n"
                    f"Snippet: {item.get('snippet', 'N/A')}"
                )
            
            return "\n\n".join(results) if results else "No results found"
            
    except Exception as e:
        logger.error(f"Web search failed: {str(e)}")
        return f"Search error: {str(e)}"

async def execute_python_code(code: str) -> str:
    """Safely execute Python code (sandboxed)"""
    # Security checks
    blocked_keywords = [
        'import', 'open', 'os', 'sys', 'subprocess', 
        'exec', 'eval', 'delete', 'write'
    ]
    
    if any(re.search(rf'\b{kw}\b', code) for kw in blocked_keywords):
        return "Error: Code contains restricted keywords"
    
    try:
        # Create restricted globals
        restricted_globals = {
            '__builtins__': {
                'range': range,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'sum': sum,
                'min': min,
                'max': max
            }
        }
        
        # Execute in isolated namespace
        local_vars = {}
        exec(code, restricted_globals, local_vars)
        
        # Get the result
        if '_result' in local_vars:
            return str(local_vars['_result'])
        return "Code executed (no result captured)"
        
    except Exception as e:
        return f"Execution error: {str(e)}"

TOOL_REGISTRY = {
    "web_search": execute_web_search,
    "python_interpreter": execute_python_code
}

# ---------- Agent Implementation ----------
class QuainexAgent:
    def __init__(self, provider: str = "openrouter"):
        self.provider = provider
        self.history = []
        self.max_loops = MAX_AGENT_LOOPS
    
    async def run(self, user_prompt: str) -> str:
        """Execute the agent loop"""
        self.history = build_prompt_context(user_prompt, "agent")
        
        for loop in range(self.max_loops):
            logger.info(f"Agent loop {loop + 1}/{self.max_loops}")
            
            # Get AI response
            try:
                response = await fetch_ai_response(self.provider, self.history)
                ai_response = response["response"]
                self.history.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                logger.error(f"Agent failed to get response: {str(e)}")
                return f"Agent error: {str(e)}"
            
            # Parse the response
            parsed = self.parse_response(ai_response)
            
            # Handle final answer
            if parsed["final_answer"]:
                logger.info("Agent completed successfully")
                return parsed["final_answer"]
            
            # Handle tool call
            if parsed["tool_name"]:
                tool_result = await self.execute_tool(
                    parsed["tool_name"],
                    parsed["tool_params"]
                )
                self.history.append({
                    "role": "user",
                    "content": f"<tool_result>\n{tool_result}\n</tool_result>"
                })
            else:
                logger.warning("Agent response missing both answer and tool call")
                return "Agent failed to provide a valid response"
        
        return "Agent reached maximum loops without completing the task"
    
    def parse_response(self, response_text: str) -> dict:
        """Parse the AI response for thinking, tools, and answers"""
        result = {
            "thinking": "",
            "tool_name": None,
            "tool_params": {},
            "final_answer": None
        }
        
        # Extract thinking
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response_text, re.DOTALL)
        if thinking_match:
            result["thinking"] = thinking_match.group(1).strip()
        
        # Extract tool call
        tool_match = re.search(r'<tool_call>(.*?)</tool_call>', response_text, re.DOTALL)
        if tool_match:
            try:
                tool_xml = ET.fromstring(f"<root>{tool_match.group(1)}</root>")
                result["tool_name"] = tool_xml.find("tool_name").text.strip()
                params = tool_xml.find("parameters")
                if params is not None:
                    for param in params:
                        result["tool_params"][param.tag] = param.text.strip() if param.text else ""
            except ET.ParseError as e:
                logger.error(f"Failed to parse tool call: {str(e)}")
        
        # If no tool call, treat as final answer
        if not result["tool_name"]:
            # Clean the response by removing any XML tags
            clean_text = re.sub(r'<[^>]+>', '', response_text).strip()
            result["final_answer"] = clean_text
        
        return result
    
    async def execute_tool(self, tool_name: str, params: dict) -> str:
        """Execute a tool and return the result"""
        if tool_name not in TOOL_REGISTRY:
            return f"Error: Tool '{tool_name}' not found"
        
        logger.info(f"Executing tool: {tool_name} with params: {params}")
        try:
            tool_func = TOOL_REGISTRY[tool_name]
            result = await tool_func(**params)
            return f"<tool_name>{tool_name}</tool_name>\n<result>{result}</result>"
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(error_msg)
            return f"<error>{error_msg}</error>"

# ---------- API Endpoints ----------
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest = Body(...)):
    """Main chat endpoint with enhanced CORS support"""
    try:
        start_time = datetime.now()
        logger.info(f"Received chat request: {request}")
        
        if request.personality == "agent":
            logger.info("Using agent personality")
            agent = QuainexAgent(request.provider)
            response_text = await agent.run(request.message)
            provider = request.provider
        else:
            logger.info(f"Using {request.personality} personality")
            messages = build_prompt_context(request.message, request.personality)
            result = await fetch_ai_response(request.provider, messages)
            response_text = result["response"]
            provider = result["provider"]
        
        # Log to database
        try:
            db_response = supabase.table("chat_history").insert([
                {
                    "user_id": "api_user", 
                    "role": "user", 
                    "message": request.message,
                    "provider": provider
                },
                {
                    "user_id": "api_user", 
                    "role": "assistant", 
                    "message": response_text,
                    "provider": provider
                }
            ]).execute()
            logger.debug(f"Database insert result: {db_response}")
        except Exception as e:
            logger.error(f"Database logging failed: {str(e)}")
        
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Request completed in {response_time:.2f}ms")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=jsonable_encoder({
                "success": True,
                "response": response_text,
                "provider": provider,
                "time_ms": response_time
            }),
            headers={
                "Access-Control-Allow-Origin": "https://quainexai.onrender.com",
                "Access-Control-Allow-Credentials": "true"
            }
        )
        
    except HTTPException as he:
        logger.error(f"HTTPException in chat endpoint: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request"
        )

@app.post("/voice")
async def voice_endpoint(file: UploadFile = File(...)):
    """Voice transcription endpoint with proper CORS handling"""
    try:
        logger.info("Received voice transcription request")
        # In a real implementation, process the audio file here
        # For now, return a mock response
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "response": "This is a mock response from voice transcription"
            },
            headers={
                "Access-Control-Allow-Origin": "https://quainexai.onrender.com",
                "Access-Control-Allow-Credentials": "true"
            }
        )
    except Exception as e:
        logger.error(f"Error in voice endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Voice processing failed"
        )

# Developer API endpoints
@app.post("/api/developers/register")
async def register_developer(data: DeveloperAPIKey):
    """Register a new developer and issue API key"""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "api_key": f"quainex_dev_{hash(data.email)}",
            "rate_limit": "100/day"
        },
        headers={
            "Access-Control-Allow-Origin": "https://quainexai.onrender.com",
            "Access-Control-Allow-Credentials": "true"
        }
    )

@app.get("/api/providers")
async def list_providers():
    """List available AI providers"""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "providers": [
                {
                    "id": provider,
                    "enabled": bool(API_KEYS.get(provider))
                }
                for provider in PROVIDER_FALLBACK_ORDER
            ]
        },
        headers={
            "Access-Control-Allow-Origin": "https://quainexai.onrender.com",
            "Access-Control-Allow-Credentials": "true"
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """System health check"""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "providers": {
                provider: "active" if API_KEYS.get(provider) else "inactive"
                for provider in PROVIDER_FALLBACK_ORDER
            }
        },
        headers={
            "Access-Control-Allow-Origin": "https://quainexai.onrender.com",
            "Access-Control-Allow-Credentials": "true"
        }
    )

# ---------- Error Handling ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "code": exc.status_code
        },
        headers={
            "Access-Control-Allow-Origin": "https://quainexai.onrender.com",
            "Access-Control-Allow-Credentials": "true"
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "code": 500
        },
        headers={
            "Access-Control-Allow-Origin": "https://quainexai.onrender.com",
            "Access-Control-Allow-Credentials": "true"
        }
    )

# ---------- Main Entry ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)