# main.py
from fastapi import FastAPI, Request, HTTPException, status, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv
import asyncio
from elevenlabs.client import ElevenLabs
from groq import Groq

# ---------- Load Environment Variables ---------- #
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Initialize ElevenLabs and Groq clients
elevenlabs_client = None
if ELEVENLABS_API_KEY:
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# ---------- FastAPI App ---------- #
app = FastAPI()

origins = [
    "https://quainexai.onrender.com",
    "https://quainex.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Set-Cookie"]
)

# ---------- Pydantic Models ---------- #
class ChatData(BaseModel):
    message: str
    provider: str = "openrouter"
    personality: str = "default"

class ToolData(BaseModel):
    tool: str
    content: str
    provider: str = "openrouter"

class SearchData(BaseModel):
    query: str

class TextToSpeechData(BaseModel):
    text: str

class ImageGenerationData(BaseModel):
    prompt: str

# ------------------ Personality System ------------------ #
PERSONALITIES = {
    "default": "You are Quainex, an intelligent AI assistant created by Bright SecureTech. You help users clearly and kindly. You were built by Bright Quainoo.",
    "strict": "You are Quainex, a very serious and efficient AI built to provide accurate answers only and also a hacker and a software specialist.",
    "fun": "You're Quainex, a witty and entertaining assistant who helps users with a fun attitude."
}

# ------------------ Helpers ------------------ #
def build_contextual_prompt(current_prompt: str, personality: str = "default"):
    messages = [{"role": "system", "content": PERSONALITIES.get(personality, PERSONALITIES["default"])}]
    messages.append({"role": "user", "content": current_prompt})
    return messages

async def get_chat_response(provider: str, messages: list):
    headers = {}
    data = {"messages": messages}
    url = ""
    model = ""
    timeout = httpx.Timeout(60.0)

    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            return {"response": "⚠️ OpenRouter API key is not configured."}
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        model = "openai/gpt-3.5-turbo"
        url = "https://openrouter.ai/api/v1/chat/completions"
    elif provider == "together":
        if not TOGETHER_API_KEY:
            return {"response": "⚠️ Together AI key is not configured."}
        headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
        model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        url = "https://api.together.xyz/v1/chat/completions"
    elif provider == "groq":
        if not GROQ_API_KEY:
            return {"response": "⚠️ Groq API key is not configured."}
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        model = "llama3-70b-8192"
        url = "https://api.groq.com/openai/v1/chat/completions"
    else:
        return {"response": "⚠️ Invalid provider."}
    
    data["model"] = model

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            res = response.json()
            return {"response": res["choices"][0]["message"]["content"]}
    except httpx.HTTPStatusError as e:
        print(f"API error: {e.response.status_code} {e.response.text}")
        return {"response": f"⚠️ API error: {e.response.status_code} {e.response.text}"}
    except Exception as e:
        print(f"Failed to get response from AI: {str(e)}")
        return {"response": f"⚠️ Failed to get response from AI: {str(e)}"}


# ------------------ Routes ------------------ #
@app.post("/chat")
async def chat(data: ChatData):
    prompt = data.message
    provider = data.provider
    personality = data.personality

    if not prompt:
        raise HTTPException(status_code=400, detail="⚠️ Prompt is required.")

    messages_context = build_contextual_prompt(prompt, personality) 
    ai_response = await get_chat_response(provider, messages_context)
    
    return ai_response

@app.post("/tool")
async def run_tool(data: ToolData):
    tool = data.tool
    content = data.content
    provider = data.provider

    tool_prompts = {
        "summarize": "Summarize this content clearly and concisely.",
        "translate": "Translate this text into the language they specify.",
        "analyze": "Analyze the following content and provide key insights."
    }

    if tool not in tool_prompts:
        raise HTTPException(status_code=400, detail="⚠️ Unknown tool selected.")

    messages = [
        {"role": "system", "content": tool_prompts[tool]},
        {"role": "user", "content": content}
    ]

    response = await get_chat_response(provider, messages)
    
    return response

@app.post("/search")
async def search_web(data: SearchData):
    query = data.query

    if not query:
        raise HTTPException(status_code=400, detail="⚠️ Query is required.")
    
    if not SERPER_API_KEY:
        raise HTTPException(status_code=501, detail="Serper API key not configured")

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "gl": "us", "hl": "en"}
    timeout = httpx.Timeout(30.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post("https://google.serper.dev/search", headers=headers, json=payload)
            response.raise_for_status()
            search_data = response.json()
            results = search_data.get("organic", [])

            if not results:
                return {"response": "❌ No search results found."}

            summary = "\n".join([f"- {r['title']}\n{r['link']}" for r in results[:5]])
            
            return {"response": summary}
    except httpx.HTTPStatusError as e:
        print(f"Serper API error: {e.response.status_code} {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"⚠️ Search API error: {e.response.text}")
    except Exception as e:
        print(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"⚠️ Search error: {str(e)}")

@app.post("/ask-and-search")
async def ask_and_search(data: SearchData):
    query = data.query
    provider = "openrouter"

    if not query:
        raise HTTPException(status_code=400, detail="⚠️ Query is required.")
    
    if not SERPER_API_KEY:
        raise HTTPException(status_code=501, detail="Serper API key not configured")

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "gl": "us", "hl": "en"}
    timeout = httpx.Timeout(30.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            search_response = await client.post("https://google.serper.dev/search", headers=headers, json=payload)
            search_response.raise_for_status()
            search_data = search_response.json()
            results = search_data.get("organic", [])

            if not results:
                return {"response": "❌ No search results found."}

            combined_text = "\n".join([f"{r['title']} - {r['snippet']}" for r in results[:5]])

            messages = [
                {"role": "system", "content": "Summarize and explain this web search result:"},
                {"role": "user", "content": combined_text}
            ]

            ai_response = await get_chat_response(provider, messages)
            
            return {"response": ai_response["response"]}
    except httpx.HTTPStatusError as e:
        print(f"Serper API error: {e.response.status_code} {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"⚠️ Combined tool API error: {e.response.text}")
    except Exception as e:
        print(f"Combined tool error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"⚠️ Combined tool error: {str(e)}")

@app.post("/tts")
async def text_to_speech(data: TextToSpeechData):
    if not elevenlabs_client:
        raise HTTPException(status_code=501, detail="ElevenLabs API key not configured")
    try:
        audio = elevenlabs_client.generate(text=data.text, voice="Rachel")
        return Response(content=audio, media_type="audio/mpeg")
    except Exception as e:
        print(f"ElevenLabs TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail="⚠️ Failed to generate speech.")

@app.post("/image-generation")
async def image_generation(data: ImageGenerationData):
    if not groq_client:
        raise HTTPException(status_code=501, detail="Groq API key not configured")
    try:
        response = groq_client.images.generate(
            model="dall-e",
            prompt=data.prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        image_url = response.data[0].url
        return {"image_url": image_url}
    except Exception as e:
        print(f"Groq Image Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="⚠️ Failed to generate image.")

@app.post("/voice")
async def voice_transcription(file: UploadFile = File(...)):
    # Placeholder for voice transcription.
    # In a real application, you would send 'file.file' (BytesIO object) to a Speech-to-Text API
    # (e.g., Google Cloud Speech-to-Text, OpenAI Whisper, AssemblyAI).
    # For now, it returns a dummy response.
    print(f"Received voice file: {file.filename}, content type: {file.content_type}")
    return {"response": "Voice transcription is a placeholder. Integrate a Speech-to-Text API here!"}

@app.get("/models")
def list_models():
    return {
        "openrouter": ["openai/gpt-3.5-turbo"],
        "together": ["mistralai/Mixtral-8x7B-Instruct-v0.1"],
        "groq": ["llama3-70b-8192"]
    }

@app.get("/personality")
def list_personalities():
    return {"available": list(PERSONALITIES.keys())}