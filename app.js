// app.js
// ---------- DOM References ----------
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");
const toolBtns = document.querySelectorAll(".tool-btn");
const voiceBtn = document.getElementById("voice-btn");
const providerSelect = document.getElementById("provider-select");
const ttsBtn = document.getElementById("tts-btn");
const imgGenBtn = document.getElementById("img-gen-btn");
const mobileMenuBtn = document.getElementById("mobile-menu-btn");
const sidebar = document.getElementById("sidebar");
const userAvatar = document.getElementById("user-avatar");
const userAvatarSidebar = document.getElementById("user-avatar-sidebar");

// ---------- Global Variables ----------
let currentTool = "chat";
let currentUser = "guest";
let mediaRecorder,
  audioChunks = [];
let audio = new Audio();

// IMPORTANT: Get the backend URL from the global variable set in index.html
const backendBaseUrl = window.BACKEND_URL;

// ---------- Init ----------
function init() {
  const provider = localStorage.getItem("provider") || "openrouter";
  providerSelect.value = provider;
  updateUserAvatar(currentUser);
  
  // Add welcome message
  setTimeout(() => {
    appendMessage(
      "Quainex AI", 
      "Hello! I'm Quainex AI, your intelligent assistant. How can I help you today?", 
      "bot"
    );
  }, 500);
}

function updateUserAvatar(username) {
  if (userAvatar) {
    userAvatar.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(
      username
    )}&background=22c55e&color=fff`;
  }
  if (userAvatarSidebar) {
    userAvatarSidebar.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(
      username
    )}&background=22c55e&color=fff`;
  }
}

// ---------- Tool Switching ----------
toolBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    // Remove active class from all buttons
    toolBtns.forEach(b => b.classList.remove("bg-primary-500/20", "text-primary-400"));
    
    // Add active class to clicked button
    btn.classList.add("bg-primary-500/20", "text-primary-400");
    
    currentTool = btn.dataset.tool;
    userInput.placeholder = `Ask me to ${currentTool}...`;
    
    // Show tool switch notification
    showCustomMessage(`Switched to ${currentTool} mode`);
  });
});

// ---------- Chat Message Submission ----------
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = userInput.value.trim();
  if (!message) return;

  appendMessage("You", message, "user");
  userInput.value = "";
  const loader = appendMessage("Quainex", "...", "bot", true);
  showTyping(loader.querySelector(".typing"));

  try {
    const body = {
      message: message,
      provider: providerSelect.value,
      personality: "default",
    };

    const res = await fetch(`${backendBaseUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await res.json();
    loader.remove();
    appendMessage("Quainex", data.response, "bot");
  } catch (error) {
    console.error("Chat error:", error);
    loader.remove();
    appendMessage("Quainex", "⚠️ Sorry, I encountered an error. Please try again.", "bot");
  }
});

// ---------- Typing Dots Animation ----------
function showTyping(el) {
  let dots = 0;
  const interval = setInterval(() => {
    if (!el || !el.isConnected) {
      clearInterval(interval);
      return;
    }
    el.textContent = ".".repeat((dots++ % 4));
  }, 300);
}

// ---------- Append Message to Chat ----------
function appendMessage(
  sender,
  text,
  type = "bot",
  loading = false,
  isImage = false
) {
  const wrapper = document.createElement("div");
  wrapper.className = `flex ${type === "user" ? "justify-end" : "justify-start"} fade-in`;
  
  const bubble = document.createElement("div");
  bubble.className = `px-5 py-3 rounded-2xl shadow max-w-[90%] md:max-w-[80%] whitespace-pre-wrap ${
    type === "user"
      ? "bg-primary-600 text-white rounded-br-none user-bubble"
      : "bg-dark-700 text-white rounded-bl-none bot-bubble"
  }`;
  
  const senderElement = document.createElement("div");
  senderElement.className = `text-xs font-semibold mb-1 ${
    type === "user" ? "text-primary-200" : "text-gray-400"
  }`;
  senderElement.textContent = sender;
  
  bubble.appendChild(senderElement);

  if (loading) {
    const typing = document.createElement("span");
    typing.classList.add("typing");
    typing.textContent = "...";
    bubble.appendChild(typing);
  } else if (isImage) {
    const imgElement = document.createElement("img");
    imgElement.src = text;
    imgElement.alt = "Generated Image";
    imgElement.className = "rounded-lg shadow-md mt-2 w-full max-w-xs md:max-w-sm lg:max-w-md";
    bubble.appendChild(imgElement);
  } else {
    const contentElement = document.createElement("div");
    contentElement.className = "text-gray-100";
    contentElement.textContent = text;
    bubble.appendChild(contentElement);
  }

  wrapper.appendChild(bubble);
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;
  return wrapper;
}

// ---------- Text-to-Speech ----------
ttsBtn.addEventListener("click", async () => {
  const message = userInput.value.trim();
  if (!message) {
    showCustomMessage("Please enter text to speak.");
    return;
  }

  appendMessage("You", `[TTS Request]\n${message}`, "user");
  userInput.value = "";
  const loader = appendMessage("Quainex", "Generating audio...", "bot", true);
  showTyping(loader.querySelector(".typing"));

  try {
    const res = await fetch(`${backendBaseUrl}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: message }),
    });

    const data = await res.blob();
    loader.remove();

    const audioUrl = URL.createObjectURL(data);
    audio.src = audioUrl;
    audio.play();
    appendMessage("Quainex", "🔊 Playing audio...", "bot");
  } catch (error) {
    console.error("TTS error:", error);
    loader.remove();
    appendMessage("Quainex", "⚠️ Couldn't generate audio. Please try again.", "bot");
  }
});

// ---------- Image Generation ----------
imgGenBtn.addEventListener("click", async () => {
  const message = userInput.value.trim();
  if (!message) {
    showCustomMessage("Please enter a prompt to generate an image.");
    return;
  }

  appendMessage("You", `[Image Generation]\n${message}`, "user");
  userInput.value = "";
  const loader = appendMessage("Quainex", "Generating image...", "bot", true);
  showTyping(loader.querySelector(".typing"));

  try {
    const res = await fetch(`${backendBaseUrl}/image-generation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: message }),
    });

    const data = await res.json();
    loader.remove();

    if (data.image_url) {
      appendMessage("Quainex", data.image_url, "bot", false, true);
    } else {
      appendMessage("Quainex", data.output, "bot");
    }
  } catch (error) {
    console.error("Image generation error:", error);
    loader.remove();
    appendMessage("Quainex", "⚠️ Image generation failed. Please try again.", "bot");
  }
});

// ---------- Voice Input (Recording) ----------
voiceBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    voiceBtn.classList.remove("text-red-500");
    voiceBtn.classList.add("text-gray-400");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
      const formData = new FormData();
      formData.append("file", audioBlob);

      const loader = appendMessage("Quainex", "Transcribing voice...", "bot", true);
      try {
        const res = await fetch(`${backendBaseUrl}/voice`, {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        loader.remove();
        appendMessage("Quainex", data.response, "bot");
      } catch (error) {
        console.error("Voice transcription error:", error);
        loader.remove();
        appendMessage("Quainex", "⚠️ Voice transcription failed. Please try again.", "bot");
      }
    };

    mediaRecorder.start();
    voiceBtn.classList.remove("text-gray-400");
    voiceBtn.classList.add("text-red-500");
    showCustomMessage("Recording... Click again to stop");
  } catch (error) {
    console.error("Error accessing microphone:", error);
    showCustomMessage("Microphone access denied: " + error.message);
  }
});

// ---------- Mobile Menu Toggle ----------
mobileMenuBtn.addEventListener("click", () => {
  sidebar.classList.toggle("hidden");
  sidebar.classList.toggle("flex");
});

// ---------- Provider Change ----------
providerSelect.addEventListener("change", () => {
  localStorage.setItem("provider", providerSelect.value);
  showCustomMessage(`Model provider set to ${providerSelect.options[providerSelect.selectedIndex].text}`);
});

// ---------- Custom Message Display ----------
function showCustomMessage(message) {
  const messageDiv = document.createElement("div");
  messageDiv.className =
    "fixed top-4 right-4 bg-primary-600 text-white px-4 py-2 rounded-lg shadow-lg z-[1000] flex items-center space-x-2 fade-in";
  
  const icon = document.createElement("i");
  icon.className = "fas fa-info-circle";
  messageDiv.appendChild(icon);
  
  const text = document.createElement("span");
  text.textContent = message;
  messageDiv.appendChild(text);
  
  document.body.appendChild(messageDiv);

  setTimeout(() => {
    messageDiv.classList.add("opacity-0", "translate-y-2");
    setTimeout(() => {
      messageDiv.remove();
    }, 300);
  }, 3000);
}

// ---------- Run Init ----------
init();