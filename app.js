// app.js
// ---------- DOM References ----------
const toggleDark = document.getElementById("toggle-dark");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");
const toolBtns = document.querySelectorAll(".tool-btn");
const toolStatus = document.getElementById("tool-status");
const voiceBtn = document.getElementById("voice-btn");
const providerSelect = document.getElementById("provider-select");
const ttsBtn = document.getElementById("tts-btn");
const imgGenBtn = document.getElementById("img-gen-btn");
const mobileMenuBtn = document.getElementById("mobile-menu-btn");
const sidebar = document.querySelector("aside");
const mainSection = document.querySelector("section.flex-1");
const userAvatar = document.getElementById("user-avatar");

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
}

function updateUserAvatar(username) {
  if (userAvatar) {
    userAvatar.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(
      username
    )}&background=random&color=fff`;
  }
}

// ---------- Tool Switching ----------
toolBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    currentTool = btn.dataset.tool;
    const toolStatusElement = document.getElementById("tool-status");
    if (toolStatusElement) {
      toolStatusElement.textContent = `Tool: ${currentTool}`;
    }
    userInput.placeholder = `Ask me to ${currentTool}...`;
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
    appendMessage("Quainex", "⚠️ Chat error. Please check.", "bot");
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

// ---------- Append Message to Chat (updated for XSS security) ----------
function appendMessage(
  sender,
  text,
  type = "bot",
  loading = false,
  isImage = false
) {
  const wrapper = document.createElement("div");
  wrapper.className = `flex ${type === "user" ? "justify-end" : "justify-start"}`;

  const bubble = document.createElement("div");
  bubble.className = `px-4 py-2 rounded-2xl shadow max-w-[80%] whitespace-pre-wrap ${
    type === "user"
      ? "bg-green-600 text-white rounded-br-none"
      : "bg-gray-700 text-white rounded-bl-none"
  }`;

  if (loading) {
    const typing = document.createElement("span");
    typing.classList.add("typing");
    typing.textContent = "...";
    bubble.appendChild(typing);
  } else if (isImage) {
    const imgElement = document.createElement("img");
    imgElement.src = text;
    imgElement.alt = "Generated Image";
    imgElement.className =
      "rounded-lg shadow-md mt-2 w-full max-w-xs md:max-w-sm lg:max-w-md";
    bubble.appendChild(imgElement);
  } else {
    bubble.textContent = text;
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

  appendMessage("You", `[TTS]\n${message}`, "user");
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
    appendMessage("Quainex", "▶️ Playing audio...", "bot");
  } catch (error) {
    console.error("TTS error:", error);
    loader.remove();
    appendMessage("Quainex", "⚠️ TTS error..", "bot");
  }
});

// ---------- Image Generation ----------
imgGenBtn.addEventListener("click", async () => {
  const message = userInput.value.trim();
  if (!message) {
    showCustomMessage("Please enter a prompt to generate an image.");
    return;
  }

  appendMessage("You", `[Image Gen]\n${message}`, "user");
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
    appendMessage("Quainex", "⚠️ Image generation error.", "bot");
  }
});

// ---------- Voice Input (Recording) ----------
voiceBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    voiceBtn.classList.remove("text-red-500");
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
        appendMessage("Quainex", "⚠️ Voice transcription error.", "bot");
      }
    };

    mediaRecorder.start();
    voiceBtn.classList.add("text-red-500");
  } catch (error) {
    console.error("Error accessing microphone:", error);
    showCustomMessage("Error accessing microphone: " + error.message);
  }
});

// ---------- Mobile Menu Toggle ----------
mobileMenuBtn.addEventListener("click", () => {
  sidebar.classList.toggle("hidden");
  sidebar.classList.toggle("flex");
  if (sidebar.classList.contains("hidden")) {
    mainSection.classList.remove("md:w-auto");
    mainSection.classList.add("w-full");
  } else {
    mainSection.classList.remove("w-full");
    mainSection.classList.add("md:w-auto");
  }
});

// ---------- Theme Toggle ----------
toggleDark.addEventListener("click", () => {
  document.documentElement.classList.toggle("dark");
});

// ---------- Provider Change ----------
providerSelect.addEventListener("change", () => {
  localStorage.setItem("provider", providerSelect.value);
});

// ---------- Custom Message Display (instead of alert) ----------
function showCustomMessage(message) {
  const messageDiv = document.createElement("div");
  messageDiv.className =
    "fixed top-4 right-4 bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg z-[1000]";
  messageDiv.textContent = message;
  document.body.appendChild(messageDiv);

  setTimeout(() => {
    messageDiv.remove();
  }, 3000);
}

// ---------- Run Init ----------
init();