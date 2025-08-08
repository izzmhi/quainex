# main.py
from fastapi import FastAPI, Request, Depends, HTTPException, status, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import httpx
import json
from datetime import datetime, timedelta
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

# CRITICAL: Get SECRET_KEY and raise an error if it's not set
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable not set. Please set it in your .env file.")
    
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize ElevenLabs and Groq clients
elevenlabs_client = None
if ELEVENLABS_API_KEY:
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# ---------- FastAPI App ---------- #
app = FastAPI()

# IMPORTANT: Update this list with your actual frontend domains for production
# For local development, you might use ["http://localhost:3000"] or similar.
# For Render deployment, replace with your actual frontend URL(s), e.g., ["https://your-quainex-app-frontend.onrender.com"]
origins = [
    "http://127.0.0.1:5500",
    "https://quainex.onrender.com",
    "https://your-frontend-domain.com"  # Add your actual frontend domain here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Add your production frontend URL(s) here when you deploy to Render:
# Example: production_origins.append("https://your-quainex-app-frontend.onrender.com")


# ---------- Database Setup ---------- #
# Use DATABASE_URL for Render (PostgreSQL) and fallback to SQLite for local development
# Corrected: os.getenv takes the ENV VAR name as first arg, not the URL itself.
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./users.db")

# For PostgreSQL, ensure you have 'psycopg2-binary' installed in requirements.txt
# For SQLite, 'sqlite:///./users.db' will create a file in your project directory
engine = create_engine(SQLALCHEMY_DATABASE_URL) 
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    # Relationship to ChatMessages
    chat_messages = relationship("ChatMessage", back_populates="user_rel", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    user_message = Column(String)
    bot_response = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user_rel = relationship("User", back_populates="chat_messages")

# Create database tables (will create for SQLite or connect to existing for PostgreSQL)
Base.metadata.create_all(bind=engine)

# ---------- Auth & Security ---------- #
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

def cookie_auth_scheme(request: Request):
    return request.cookies.get("access_token")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(cookie_auth_scheme),
    request: Request = None # Ensure Request is available for oauth2_scheme
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # If token not found in cookie, try checking Authorization header (for fallback/API clients)
    if not token:
        try:
            token = await oauth2_scheme(request)
        except HTTPException:
            raise credentials_exception

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(db, username)
    if user is None:
        raise credentials_exception
    return user

# ---------- Pydantic Models ---------- #
class SignupData(BaseModel):
    username: str
    password: str

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
    
class MessageData(BaseModel):
    message: str
    response: str

# ------------------ Personality System ------------------ #
PERSONALITIES = {
 "default": "You are Quainex, an intelligent AI assistant created by Bright SecureTech. You help users clearly and kindly. You were built by Bright Quainoo.",
    "strict": "You are Quainex, a very serious and efficient AI built to provide accurate answers only and also a hacker and a software specialist.",
    "fun": "You're Quainex, a witty and entertaining assistant who helps users with a fun attitude."
}

# ------------------ Helpers ------------------ #
def build_contextual_prompt(user_id: int, current_prompt: str, personality: str = "default", db: Session = Depends(get_db)):
    # Fetch last 10 messages from DB for context
    # Ensure this is called with a 'db' session
    history_messages = db.query(ChatMessage).filter(ChatMessage.user_id == user_id).order_by(ChatMessage.timestamp.desc()).limit(10).all()
    history_messages.reverse() # Reverse to get chronological order

    messages = [{"role": "system", "content": PERSONALITIES.get(personality, PERSONALITIES["default"])}]
    for item in history_messages:
        messages.append({"role": "user", "content": item.user_message})
        messages.append({"role": "assistant", "content": item.bot_response})
    messages.append({"role": "user", "content": current_prompt})
    return messages

async def get_chat_response(provider: str, messages: list):
    headers = {}
    data = {"messages": messages}
    url = ""
    model = ""
    timeout = httpx.Timeout(60.0) # Increased timeout for AI responses

    if provider == "openrouter":
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        model = "openai/gpt-3.5-turbo"
        url = "https://openrouter.ai/api/v1/chat/completions"
    elif provider == "together":
        headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
        model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        url = "https://api.together.xyz/v1/chat/completions"
    elif provider == "groq":
        if not groq_client:
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
@app.post("/signup")
def signup(data: SignupData, db: Session = Depends(get_db)):
    if get_user(db, data.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(data.password)
    new_user = User(username=data.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}

@app.post("/token")
def login_for_access_token(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/token-cookie")
def login_with_cookie(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    
    response = Response(content="Login successful", media_type="text/plain")
   response.set_cookie(
    key="access_token",
    value=access_token,
    httponly=True,
    samesite="Lax",
    secure=True,
    max_age=1800 
)
    return response

@app.post("/logout")
def logout(response: Response):
    response.delete_cookie(key="access_token")
    return {"message": "Logged out successfully"}

@app.get("/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username, "id": current_user.id}

@app.get("/history")
def get_chat_history_from_db(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    messages = db.query(ChatMessage).filter(ChatMessage.user_id == current_user.id).order_by(ChatMessage.timestamp).all()
    return {"messages": [{"user": m.user_message, "bot": m.bot_response} for m in messages]}

# This endpoint is no longer strictly needed if chat, tool, search, ask-and-search save directly
# but kept for consistency with frontend's explicit save call if it exists.
@app.post("/history")
def save_chat_history_to_db(data: MessageData, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_message = ChatMessage(user_id=current_user.id, user_message=data.message, bot_response=data.response)
    db.add(new_message)
    db.commit()
    db.refresh(new_message)
    return {"message": "History saved successfully"}

# ------------------ Protected Routes ------------------ #
@app.post("/chat")
async def chat(data: ChatData, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user_id = current_user.id
    prompt = data.message
    provider = data.provider
    personality = data.personality

    if not prompt:
        raise HTTPException(status_code=400, detail="⚠️ Prompt is required.")

    # Pass db to build_contextual_prompt
    messages_context = build_contextual_prompt(user_id, prompt, personality, db) 
    ai_response = await get_chat_response(provider, messages_context)
    
    # Save to database
    new_message = ChatMessage(user_id=user_id, user_message=prompt, bot_response=ai_response["response"])
    db.add(new_message)
    db.commit()
    db.refresh(new_message)

    return ai_response

@app.post("/tool")
async def run_tool(data: ToolData, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user_id = current_user.id
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
    
    # Save to database
    new_message = ChatMessage(user_id=user_id, user_message=f"[{tool.upper()}] {content}", bot_response=response["response"])
    db.add(new_message)
    db.commit()
    db.refresh(new_message)

    return response

@app.post("/search")
async def search_web(data: SearchData, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user_id = current_user.id
    query = data.query

    if not query:
        raise HTTPException(status_code=400, detail="⚠️ Query is required.")

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
            
            # Save to database
            new_message = ChatMessage(user_id=user_id, user_message=f"[SEARCH] {query}", bot_response=summary)
            db.add(new_message)
            db.commit()
            db.refresh(new_message)

            return {"response": summary}
    except httpx.HTTPStatusError as e:
        print(f"Serper API error: {e.response.status_code} {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"⚠️ Search API error: {e.response.text}")
    except Exception as e:
        print(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"⚠️ Search error: {str(e)}")

@app.post("/ask-and-search")
async def ask_and_search(data: SearchData, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user_id = current_user.id
    query = data.query
    provider = "openrouter" # Default provider for this route

    if not query:
        raise HTTPException(status_code=400, detail="⚠️ Query is required.")

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
            
            # Save to database
            new_message = ChatMessage(user_id=user_id, user_message=f"[SEARCH+ASK] {query}", bot_response=ai_response["response"])
            db.add(new_message)
            db.commit()
            db.refresh(new_message)

            return {"response": ai_response["response"]}
    except httpx.HTTPStatusError as e:
        print(f"Serper API error: {e.response.status_code} {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"⚠️ Combined tool API error: {e.response.text}")
    except Exception as e:
        print(f"Combined tool error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"⚠️ Combined tool error: {str(e)}")

@app.post("/tts")
async def text_to_speech(data: TextToSpeechData, current_user: User = Depends(get_current_user)):
    if not elevenlabs_client:
        raise HTTPException(status_code=501, detail="ElevenLabs API key not configured")
    try:
        audio = elevenlabs_client.generate(text=data.text, voice="Rachel")
        return Response(content=audio, media_type="audio/mpeg")
    except Exception as e:
        print(f"ElevenLabs TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail="⚠️ Failed to generate speech.")

@app.post("/image-generation")
async def image_generation(data: ImageGenerationData, current_user: User = Depends(get_current_user)):
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
async def voice_transcription(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
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