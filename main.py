from fastapi import FastAPI, Request, HTTPException, status, Response, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
import re
from fastapi.encoders import jsonable_encoder
from quainexmemory import MemoryManager
import requests
import io
import uvicorn # <-- Missing import added

# Initialize MemoryManager
memory = MemoryManager(limit=10)

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
    "supabasekey": os.getenv("SUPABASE_KEY"),
    "newsapi": os.getenv("NEWS_API_KEY") # <-- API Key loaded from .env
}

NEWS_API_URL = "https://newsapi.org/v2/top-headlines"

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
        "https://quainex.space", # <-- YOUR NEW DOMAIN
        "https://quainexai.onrender.com",
        "https://quainex.onrender.com",
        "http://localhost:3000" # For local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600
)

# ---------------- NEWS FUNCTIONS ----------------
def get_latest_world_news():
    if not API_KEYS.get("newsapi"):
        return "Error: News API key not configured."
    params = {
        "language": "en",
        "pageSize": 5,
        "apiKey": API_KEYS["newsapi"]
    }
    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "ok":
            return "Error fetching news: " + data.get("message", "Unknown error")
        
        news_list = []
        for article in data["articles"]:
            title = article.get("title", "No title")
            description = article.get("description", "No description")
            source = article["source"].get("name", "Unknown source")
            url = article.get("url", "")
            news_item = f"**{title}** ({source})\n{description}\nRead more: {url}"
            news_list.append(news_item)
        return "\n\n".join(news_list) if news_list else "No news found."
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return f"Error: Could not retrieve news. {e}"

@app.get("/api/news/world")
async def fetch_world_news():
    news = get_latest_world_news()
    return {"news": news}

@app.get("/api/news")
async def fetch_news(country: Optional[str] = None):
    if not API_KEYS.get("newsapi"):
        raise HTTPException(status_code=500, detail="News API key not configured.")
    params = {
        "language": "en",
        "pageSize": 5,
        "apiKey": API_KEYS["newsapi"]
    }
    if country:
        params["country"] = country.lower()

    try:
        response = requests.get(NEWS_API_URL, params=params)
        data = response.json()
        if data.get("status") != "ok":
            raise HTTPException(status_code=500, detail=data.get("message", "Error fetching news"))

        news_list = [
            {
                "title": article.get("title"),
                "description": article.get("description"),
                "source": article["source"].get("name"),
                "url": article.get("url")
            }
            for article in data["articles"]
        ]
        return {"news": news_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
# ---------- Tools Schema ----------
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
    <tool>
        <name>world_time</name>
        <description>Get current time for any city or country</description>
        <parameters>
            <param>
                <name>location</name>
                <type>string</type>
                <description>The city or country name (e.g., 'Paris' or 'Japan')</description>
            </param>
        </parameters>
    </tool>
    <tool>
        <name>latest_news</name>
        <description>Get the latest world news headlines</description>
        <parameters></parameters>
    </tool>
    <tool>
        <name>generate_image</name>
        <description>Generates an image based on a descriptive text prompt.</description>
        <parameters>
            <param>
                <name>prompt</name>
                <type>string</type>
                <description>A detailed description of the image to generate.</description>
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
        "system_prompt": "You are Quainex, a helpful AI assistant. Provide clear, accurate responses. You were built by Bright SecureTech, and the founder is called Bright Quainoo."
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

DEFAULT_PROVIDER = "openrouter"  # Backend provider for Brilux
DEFAULT_MODEL_NAME = "brilux"    # Friendly AI model name

async def fetch_ai_response(provider: str, messages: list, timeout: int = 60) -> dict:
    """Get AI response with fallback logic"""

    if not provider or provider not in PROVIDER_FALLBACK_ORDER:
        logger.warning(f"Defaulting to Brilux ({DEFAULT_PROVIDER})")
        provider = DEFAULT_PROVIDER

    result = await query_provider(provider, messages, timeout)
    if result["ok"]:
        return {
            "success": True,
            "response": result["response"],
            "provider": DEFAULT_MODEL_NAME if provider == DEFAULT_PROVIDER else provider
        }

    for fallback in [p for p in PROVIDER_FALLBACK_ORDER if p != provider]:
        if not API_KEYS.get(fallback):
            continue
        result = await query_provider(fallback, messages, timeout)
        if result["ok"]:
            logger.warning(f"Fell back to provider: {fallback}")
            return {
                "success": True,
                "response": result["response"],
                "provider": fallback,
                "fallback_used": True
            }

    raise HTTPException(
        status_code=502,
        detail="All AI providers failed to respond."
    )

# ---------- Tool Implementations ----------
async def execute_web_search(query: str) -> str:
    """Perform a web search using Serper API"""
    if not API_KEYS.get("serper"):
        return "Error: Search API not configured"
    
    try:
        headers = {"X-API-KEY": API_KEYS["serper"], "Content-Type": "application/json"}
        payload = json.dumps({"q": query, "gl": "us", "hl": "en"})
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://google.serper.dev/search",
                headers=headers,
                content=payload
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
    """
    Safely execute Python code.
    NOTE: This sandbox is basic. For production, use a more secure solution 
    like Docker containers or a dedicated sandboxing library.
    """
    blocked_keywords = [
        'import', 'open', 'os', 'sys', 'subprocess', 
        'exec', 'eval', 'delete', 'write', 'input'
    ]
    
    if any(re.search(rf'\b{kw}\b', code) for kw in blocked_keywords):
        return "Error: Code contains restricted keywords for security reasons."
    
    try:
        output_capture = io.StringIO()
        
        # Create a safe execution environment
        safe_globals = {
            '__builtins__': {
                'print': lambda *args, **kwargs: print(*args, file=output_capture, **kwargs),
                'range': range, 'len': len, 'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'tuple': tuple, 'set': set, 'sum': sum,
                'min': min, 'max': max, 'abs': abs, 'round': round, 'pow': pow
            }
        }
        
        exec(code, safe_globals)
        
        result = output_capture.getvalue()
        return result if result else "Code executed successfully with no printed output."
        
    except Exception as e:
        return f"Execution error: {str(e)}"

async def execute_image_generation(prompt: str) -> str:
    """Generates an image using OpenRouter's Stable Diffusion model."""
    if not API_KEYS.get("openrouter"):
        return "Error: OpenRouter API key not configured."
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {API_KEYS['openrouter']}",
                    "HTTP-Referer": "https://quainexai.onrender.com",
                    "X-Title": "Quainex AI"
                },
                json={
                    "model": "stabilityai/stable-diffusion-3",
                    "prompt": prompt,
                    "n": 1,
                    "size": "1024x1024"
                }
            )
            response.raise_for_status()
            data = response.json()
            image_url = data['data'][0]['url']
            return f"Image generated successfully. You can view it here: {image_url}"
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")
        return f"Sorry, I was unable to generate the image. Error: {str(e)}"

async def execute_world_time(location: str) -> str:
    """Gets the current time for a specified location using WorldTimeAPI."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # First, try to find a timezone for the location
            response = await client.get(f"http://worldtimeapi.org/api/timezone/{location}")
            
            # If that fails, it might be an area, try to auto-detect
            if response.status_code != 200:
                 response = await client.get(f"http://worldtimeapi.org/api/ip") # Fallback to IP
                 if response.status_code != 200:
                    return f"Error: Could not find a valid timezone for '{location}'."

            data = response.json()
            datetime_str = data.get('datetime')
            timezone = data.get('timezone')
            
            if not datetime_str or not timezone:
                return "Error: Invalid response from time API."

            # Parse the datetime string and format it nicely
            dt_object = datetime.fromisoformat(datetime_str)
            formatted_time = dt_object.strftime('%A, %B %d, %Y, %I:%M:%S %p')
            
            return f"The current time in {timezone} is {formatted_time}."

    except Exception as e:
        logger.error(f"World time tool failed for '{location}': {str(e)}")
        return f"Error: Could not retrieve the time for '{location}'. {str(e)}"

TOOL_REGISTRY = {
    "web_search": execute_web_search,
    "python_interpreter": execute_python_code,
    "latest_news": get_latest_world_news,
    "generate_image": execute_image_generation,
    "world_time": execute_world_time # <-- Implemented tool added to registry
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
            
            try:
                response = await fetch_ai_response(self.provider, self.history)
                ai_response = response["response"]
                self.history.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                logger.error(f"Agent failed to get response: {str(e)}")
                return f"Agent error: {str(e)}"
            
            parsed = self.parse_response(ai_response)
            
            if parsed["final_answer"]:
                logger.info("Agent completed successfully")
                return parsed["final_answer"]
            
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
                logger.warning("Agent response missing both answer and tool call; treating as final answer.")
                return ai_response
        
        return "Agent reached maximum loops without completing the task. Please try rephrasing your request."
    
    def parse_response(self, response_text: str) -> dict:
        """Parse the AI response for thinking, tools, and answers"""
        result = {
            "thinking": "",
            "tool_name": None,
            "tool_params": {},
            "final_answer": None
        }
        
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response_text, re.DOTALL)
        if thinking_match:
            result["thinking"] = thinking_match.group(1).strip()
        
        tool_match = re.search(r'<tool_call>(.*?)</tool_call>', response_text, re.DOTALL)
        if tool_match:
            try:
                tool_xml_string = tool_match.group(1).strip()
                tool_xml_string = tool_xml_string.replace('&', '&amp;')
                
                tool_name_match = re.search(r'<tool_name>([^<]+)</tool_name>', tool_xml_string)
                if tool_name_match:
                    result["tool_name"] = tool_name_match.group(1).strip()
                else: 
                    return result 

                # Using regex for simpler parsing of parameters
                params_match = re.search(r'<parameters>(.*?)</parameters>', tool_xml_string, re.DOTALL)
                if params_match:
                    params_str = params_match.group(1)
                    param_tags = re.findall(r'<([^>]+)>([^<]*)</\1>', params_str)
                    for tag, value in param_tags:
                        result["tool_params"][tag.strip()] = value.strip()

            except Exception as e:
                logger.error(f"Failed to parse tool call: {str(e)}")
                logger.error(f"Malformed content was: {tool_match.group(1)}")

        if not result["tool_name"]:
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
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**params)
            else:
                result = tool_func(**params)
            return f"<tool_name>{tool_name}</tool_name>\n<result>{result}</result>"
        except Exception as e:
            error_msg = f"Tool '{tool_name}' execution failed: {str(e)}"
            logger.error(error_msg)
            return f"<error>{error_msg}</error>"

# ---------- API Endpoints ----------
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest = Body(...)):
    """Main chat endpoint with memory support"""
    try:
        start_time = datetime.now()
        logger.info(f"Received chat request: {request.model_dump_json()}")

        conv_id = request.conversation_id or "default"
        memory.add_message(conv_id, "user", request.message)
        provider = request.provider

        if request.personality == "agent":
            logger.info("Using agent personality")
            agent = QuainexAgent(provider)
            response_text = await agent.run(request.message)
        else:
            logger.info(f"Using {request.personality} personality")
            history = memory.get_history(conv_id)
            messages = build_prompt_context(request.message, request.personality, history)
            result = await fetch_ai_response(provider, messages)
            response_text = result["response"]
            provider = result.get("provider", provider)

        memory.add_message(conv_id, "assistant", response_text)

        try:
            db_response = supabase.table("chat_history").insert([
                { "user_id": "api_user", "role": "user", "message": request.message, "provider": provider },
                { "user_id": "api_user", "role": "assistant", "message": response_text, "provider": provider }
            ]).execute()
            if db_response.data:
                 logger.debug(f"Logged {len(db_response.data)} records to database.")
            else:
                 logger.warning(f"Database logging failed. Response: {db_response.status_code} - {db_response.error}")

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
            })
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
    """Transcribes audio using Groq and gets a chat response."""
    if not clients.get("groq"):
        raise HTTPException(status_code=500, detail="Groq client not configured")

    logger.info("Received voice transcription request")
    try:
        files = { "file": (file.filename, await file.read(), file.content_type) }
        payload = { "model": "whisper-large-v3" }
        headers = { "Authorization": f"Bearer {API_KEYS['groq']}" }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                files=files,
                data=payload,
                headers=headers
            )
            response.raise_for_status()
            transcription_data = response.json()
            transcript = transcription_data.get("text")

        if not transcript:
            raise HTTPException(status_code=500, detail="Transcription failed to return text.")

        logger.info(f"Transcript: {transcript}")

        chat_req = ChatRequest(message=transcript, provider="groq")
        chat_response = await chat_endpoint(chat_req)
        
        response_content = json.loads(chat_response.body.decode())
        response_content["transcript"] = transcript

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_content
        )

    except Exception as e:
        logger.error(f"Error in voice endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice processing failed: {str(e)}"
        )


@app.post("/api/tts")
async def tts_endpoint(request: TTSRequest):
    """Generates audio from text using ElevenLabs."""
    if not clients.get("elevenlabs"):
        raise HTTPException(status_code=500, detail="ElevenLabs client not configured.")

    try:
        audio_stream = clients["elevenlabs"].generate(
            text=request.text,
            voice=request.voice,
            model=request.model,
            stream=True
        )
        return StreamingResponse(audio_stream, media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate audio.")

@app.post("/api/developers/register")
async def register_developer(data: DeveloperAPIKey):
    """Register a new developer and issue API key"""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "api_key": f"quainex_dev_{hash(data.email)}",
            "rate_limit": "100/day"
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
        }
    )

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
        }
    )

# ---------- Main Entry ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)