// ---------- DOM References ----------
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");
const voiceBtn = document.getElementById("voice-btn");
const providerSelect = document.getElementById("provider-select");
const mobileMenuBtn = document.getElementById("mobile-menu-btn");
const sidebar = document.getElementById("sidebar");
const userAvatar = document.getElementById("user-avatar");
const userAvatarSidebar = document.getElementById("user-avatar-sidebar");
const welcomeScreen = document.getElementById("welcome-screen");
const overlay = document.getElementById("overlay");

// New DOM references for the buttons
const newChatBtn = document.querySelector("#sidebar button:first-of-type");
const searchBtn = document.querySelector("#sidebar button:nth-of-type(2)");
const settingsBtn = document.querySelector("#sidebar .mt-auto button:first-of-type");

// ---------- Global Variables ----------
let currentUser = "guest";
let mediaRecorder,
  audioChunks = [];
let audio = new Audio();

// IMPORTANT: Set backend URL to your Render backend
const backendBaseUrl = "https://quainex.onrender.com";

// ---------- Init ----------
function init() {
  // Remove any existing listeners to prevent duplicates
  console.log("handleFormSubmit fired!");
  console.log("init() starting...");
  chatForm.removeEventListener("submit", handleFormSubmit);
  voiceBtn.removeEventListener("click", handleVoiceInput);
  mobileMenuBtn.removeEventListener("click", toggleSidebar);
  overlay.removeEventListener("click", toggleSidebar);
  providerSelect.removeEventListener("change", handleProviderChange);

  // Initialize provider from localStorage
  const provider = localStorage.getItem("provider") || "openrouter";
  providerSelect.value = provider;
  updateUserAvatar(currentUser);
  
  // Set up event listeners
  setupEventListeners();
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

// ---------- Event Listeners Setup ----------
function setupEventListeners() {
    console.log("setupEventListeners called, attaching submit handler");

    document.getElementById("chat-form")
        .addEventListener("submit", handleFormSubmit);
  // Voice button
  voiceBtn.addEventListener("click", handleVoiceInput);
  
  // Sidebar buttons
  newChatBtn.addEventListener("click", handleNewChat);
  searchBtn.addEventListener("click", handleSearch);
  settingsBtn.addEventListener("click", handleSettings);
  
  // Sidebar toggle
  mobileMenuBtn.addEventListener("click", toggleSidebar);
  overlay.addEventListener("click", toggleSidebar);
  
  // Provider change
  providerSelect.addEventListener("change", handleProviderChange);
  
  // Auto-resize textarea
  userInput.addEventListener("input", () => {
    userInput.style.height = 'auto';
    userInput.style.height = userInput.scrollHeight + 'px';
  });
}

// ---------- Form Submission Handler ----------
async function handleFormSubmit(e) {
  e.preventDefault();
  e.stopPropagation();
  e.stopImmediatePropagation();

  const message = userInput.value.trim();
  if (!message) return;

  // disable submit button while request is in-flight
  const submitBtn = chatForm.querySelector('button[type="submit"]');
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.classList.add('opacity-50', 'cursor-not-allowed');
  }

  console.log("Preparing to send message:", message);

  // Hide welcome and show chat when first message sent
  if (welcomeScreen.style.display !== "none") {
    welcomeScreen.style.display = "none";
    chatBox.classList.remove("hidden");
  }

  appendMessage("You", message, "user");
  userInput.value = "";
  userInput.style.height = "auto";

  const loader = appendMessage("Quainex", "...", "bot", true);
  showTyping(loader.querySelector(".typing"));

  try {
    console.log("Sending request to backend...");
    
    // <-- IMPORTANT: await and assign response
    const response = await fetch(`${backendBaseUrl}/api/chat`, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify({
        message: message,
        provider: providerSelect.value,
        personality: "default"
      }),
      credentials: "include"
    });

    console.log("Received response status:", response.status);

    // handle non-JSON or error bodies gracefully
    if (!response.ok) {
      let errorData = {};
      try {
        errorData = await response.json();
      } catch (_) {
        // not JSON
      }
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("Response data:", data);

    loader.remove();

    if (data && data.success && data.response) {
      appendMessage("Quainex", data.response, "bot");
    } else {
      throw new Error("Invalid response format from server");
    }
  } catch (error) {
    console.error("Chat error:", error);
    try { loader.remove(); } catch (e) {}
    appendMessage("Quainex", `⚠️ Error: ${error.message || "Please try again later"}`, "bot");
  } finally {
    // re-enable submit button
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    }
  }
}


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
function appendMessage(sender, text, type = "bot", loading = false) {
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

// ---------- Voice Input Handler ----------
async function handleVoiceInput() {
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
          credentials: "include"
        });
        
        if (!res.ok) {
          throw new Error(`Voice transcription failed: ${res.status}`);
        }
        
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
}

// ---------- Sidebar Handlers ----------
function handleNewChat() {
  chatBox.innerHTML = "";
  chatBox.classList.add("hidden");
  welcomeScreen.style.display = "block";
  sidebar.classList.remove("active");
  overlay.classList.remove("active");
  showCustomMessage("Started a new chat!");
}

function handleSearch() {
  showCustomMessage("Search functionality not implemented yet.");
  sidebar.classList.remove("active");
  overlay.classList.remove("active");
}

function handleSettings() {
  showCustomMessage("Settings page not implemented yet.");
  sidebar.classList.remove("active");
  overlay.classList.remove("active");
}

function toggleSidebar() {
  sidebar.classList.toggle("active");
  overlay.classList.toggle("active");
}

// ---------- Provider Change Handler ----------
function handleProviderChange() {
  localStorage.setItem("provider", providerSelect.value);
  showCustomMessage(`Model provider set to ${providerSelect.options[providerSelect.selectedIndex].text}`);
}

// ---------- Custom Message Display ----------
function showCustomMessage(message) {
  const messageDiv = document.createElement("div");
  messageDiv.className = "fixed top-4 right-4 bg-primary-600 text-white px-4 py-2 rounded-lg shadow-lg z-[1000] flex items-center space-x-2 fade-in";
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

document.addEventListener("DOMContentLoaded", function() {
    console.log("init() starting...");
    setupEventListeners();
});
