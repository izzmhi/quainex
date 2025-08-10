// app.js (FULL - matches your index.html)

// ---------- Globals & placeholders ----------
let chatForm,
  chatInput,
  chatBox,
  welcomeScreen,
  providerSelect,
  voiceBtn,
  mobileMenuBtn,
  sidebar,
  overlay,
  userAvatar,
  userAvatarSidebar,
  newChatBtn,
  searchBtn,
  settingsBtn;

let mediaRecorder;
let audioChunks = [];
let audio = new Audio();

const BACKEND_URL = window.BACKEND_URL || "https://quainex.onrender.com";

// ---------- Initialization ----------
function init() {
  console.log("[init] Starting app...");

  // Grab elements (safe guards)
  chatForm = document.getElementById("chat-form");
  chatInput = document.getElementById("chat-input");
  chatBox = document.getElementById("chat-box");
  welcomeScreen = document.getElementById("welcome-screen");
  providerSelect = document.getElementById("provider-select");
  voiceBtn = document.getElementById("voice-btn");
  mobileMenuBtn = document.getElementById("mobile-menu-btn");
  sidebar = document.getElementById("sidebar");
  overlay = document.getElementById("overlay");
  userAvatar = document.getElementById("user-avatar");
  userAvatarSidebar = document.getElementById("user-avatar-sidebar");

  // Sidebar buttons (may return null if structure changes)
  newChatBtn = document.querySelector("#sidebar button:first-of-type");
  searchBtn = document.querySelector("#sidebar button:nth-of-type(2)");
  settingsBtn = document.querySelector("#sidebar .mt-auto button:first-of-type");

  // Ensure providerSelect default stored
  if (providerSelect) {
    const provider = localStorage.getItem("provider") || providerSelect.value || "deepseek";
    providerSelect.value = provider;
  }

  // Remove any existing listeners first (defensive)
  safeRemoveListener(chatForm, "submit", handleFormSubmit);
  safeRemoveListener(chatInput, "keydown", handleKeyPress);
  safeRemoveListener(voiceBtn, "click", handleVoiceInput);
  safeRemoveListener(mobileMenuBtn, "click", toggleSidebar);
  safeRemoveListener(overlay, "click", toggleSidebar);
  safeRemoveListener(providerSelect, "change", handleProviderChange);
  safeRemoveListener(newChatBtn, "click", handleNewChat);
  safeRemoveListener(searchBtn, "click", handleSearch);
  safeRemoveListener(settingsBtn, "click", handleSettings);

  // Attach fresh listeners
  setupEventListeners();
  console.log("[init] Ready.");
}

// safe helper to remove listener only if element exists
function safeRemoveListener(el, evt, fn) {
  if (!el || !fn) return;
  try {
    el.removeEventListener(evt, fn);
  } catch (e) {
    /* ignore */
  }
}

// ---------- Event listeners binding ----------
function setupEventListeners() {
  if (chatForm) chatForm.addEventListener("submit", handleFormSubmit);
  if (chatInput) chatInput.addEventListener("keydown", handleKeyPress);
  if (voiceBtn) voiceBtn.addEventListener("click", handleVoiceInput);
  if (mobileMenuBtn) mobileMenuBtn.addEventListener("click", toggleSidebar);
  if (overlay) overlay.addEventListener("click", toggleSidebar);
  if (providerSelect) providerSelect.addEventListener("change", handleProviderChange);
  if (newChatBtn) newChatBtn.addEventListener("click", handleNewChat);
  if (searchBtn) searchBtn.addEventListener("click", handleSearch);
  if (settingsBtn) settingsBtn.addEventListener("click", handleSettings);

  // Auto-resize textarea
  if (chatInput) {
    chatInput.addEventListener("input", () => {
      chatInput.style.height = "auto";
      chatInput.style.height = chatInput.scrollHeight + "px";
    });
  }
}

// ---------- Submit handler ----------
async function handleFormSubmit(e) {
  if (e && e.preventDefault) e.preventDefault();
  if (e && e.stopPropagation) e.stopPropagation();

  if (!chatInput) return;
  const message = chatInput.value.trim();
  if (!message) return;

  console.log("[handleFormSubmit] message:", message);

  // Disable submit button if present
  const submitBtn = chatForm ? chatForm.querySelector('button[type="submit"]') : null;
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.classList.add("opacity-50", "cursor-not-allowed");
  }

  // Show chat area if it was hidden
  if (welcomeScreen && chatBox) {
    welcomeScreen.style.display = "none";
    chatBox.classList.remove("hidden");
  }

  appendMessage("You", message, "user");
  chatInput.value = "";
  chatInput.style.height = "auto";

  // Add loader message
  const loader = appendMessage("Quainex", "...", "bot", true);
  showTyping(loader.querySelector(".typing"));

  try {
    const payload = {
      message,
      provider: providerSelect ? providerSelect.value : undefined,
      personality: "default"
    };

    console.log("[handleFormSubmit] POST ->", `${BACKEND_URL}/api/chat`, payload);

    const response = await fetch(`${BACKEND_URL}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json"
      },
      credentials: "include",
      body: JSON.stringify(payload)
    });

    console.log("[handleFormSubmit] status:", response.status);

    // try to parse JSON (if possible)
    let data = null;
    try {
      data = await response.json();
    } catch (err) {
      console.warn("[handleFormSubmit] response not JSON");
    }

    loader.remove();

    if (!response.ok) {
      const errMsg = (data && (data.error || data.message)) || `HTTP ${response.status}`;
      throw new Error(errMsg);
    }

    // Support multiple possible reply fields
    const reply =
      (data && (data.response || data.reply || data.answer || data.data || data.text)) ||
      "No reply from server.";

    appendMessage("Quainex", reply, "bot");
    console.log("[handleFormSubmit] reply:", reply);
  } catch (err) {
    console.error("[handleFormSubmit] error:", err);
    try { loader.remove(); } catch (e) {}
    appendMessage("Quainex", `⚠️ Error: ${err.message || "Please try again later"}`, "bot");
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.classList.remove("opacity-50", "cursor-not-allowed");
    }
  }
}

// ---------- Keyboard: Enter to submit (Shift+Enter -> newline) ----------
function handleKeyPress(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    // prefer requestSubmit when available
    if (chatForm && typeof chatForm.requestSubmit === "function") {
      chatForm.requestSubmit();
    } else if (chatForm) {
      chatForm.dispatchEvent(new Event("submit", { cancelable: true }));
    } else {
      // fallback direct call
      handleFormSubmit();
    }
  }
}

// ---------- Typing dots for loader ----------
function showTyping(el) {
  if (!el) return;
  let dots = 0;
  const iv = setInterval(() => {
    if (!el || !el.isConnected) {
      clearInterval(iv);
      return;
    }
    el.textContent = ".".repeat(dots++ % 4);
  }, 300);
}

// ---------- Append message bubble ----------
function appendMessage(sender, text, type = "bot", loading = false) {
  if (!chatBox) return null;
  const wrapper = document.createElement("div");
  wrapper.className = `flex ${type === "user" ? "justify-end" : "justify-start"} fade-in`;

  const bubble = document.createElement("div");
  bubble.className =
    "px-5 py-3 rounded-2xl shadow max-w-[90%] md:max-w-[80%] whitespace-pre-wrap message-bubble " +
    (type === "user" ? "bg-primary-600 text-white rounded-br-none user-bubble" : "bg-dark-700 text-white rounded-bl-none bot-bubble");

  const senderEl = document.createElement("div");
  senderEl.className = `text-xs font-semibold mb-1 ${type === "user" ? "text-primary-200" : "text-gray-400"}`;
  senderEl.textContent = sender;
  bubble.appendChild(senderEl);

  if (loading) {
    const typing = document.createElement("span");
    typing.classList.add("typing");
    typing.textContent = "...";
    bubble.appendChild(typing);
  } else {
    const content = document.createElement("div");
    content.className = "text-gray-100";
    // preserve newlines
    content.textContent = text;
    bubble.appendChild(content);
  }

  wrapper.appendChild(bubble);
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;
  return wrapper;
}

// ---------- Voice (microphone) handling ----------
async function handleVoiceInput() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showCustomMessage("Microphone not supported in this browser.");
    return;
  }

  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    if (voiceBtn) {
      voiceBtn.classList.remove("text-red-500");
      voiceBtn.classList.add("text-gray-400");
    }
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
      formData.append("file", audioBlob, "recording.webm");

      const loader = appendMessage("Quainex", "Transcribing voice...", "bot", true);
      showTyping(loader.querySelector(".typing"));

      try {
        const res = await fetch(`${BACKEND_URL}/voice`, {
          method: "POST",
          body: formData,
          credentials: "include"
        });

        let data = null;
        try { data = await res.json(); } catch (e) {}

        loader.remove();
        if (res.ok && data && (data.response || data.transcript || data.text)) {
          appendMessage("Quainex", data.response || data.transcript || data.text, "bot");
        } else {
          appendMessage("Quainex", "⚠️ Voice transcription failed.", "bot");
        }
      } catch (err) {
        console.error("[voice] error:", err);
        try { loader.remove(); } catch (e) {}
        appendMessage("Quainex", "⚠️ Voice transcription failed.", "bot");
      }
    };

    mediaRecorder.start();
    if (voiceBtn) {
      voiceBtn.classList.remove("text-gray-400");
      voiceBtn.classList.add("text-red-500");
    }
    showCustomMessage("Recording... click the mic again to stop");
  } catch (err) {
    console.error("[voice] mic access error:", err);
    showCustomMessage("Microphone access denied: " + (err.message || ""));
  }
}

// ---------- Sidebar handlers ----------
function handleNewChat() {
  if (chatBox) chatBox.innerHTML = "";
  if (chatBox) chatBox.classList.add("hidden");
  if (welcomeScreen) welcomeScreen.style.display = "block";
  if (sidebar) sidebar.classList.remove("active");
  if (overlay) overlay.classList.remove("active");
  showCustomMessage("Started a new chat");
}

function handleSearch() {
  showCustomMessage("Search not implemented yet.");
  if (sidebar) sidebar.classList.remove("active");
  if (overlay) overlay.classList.remove("active");
}

function handleSettings() {
  showCustomMessage("Settings not implemented yet.");
  if (sidebar) sidebar.classList.remove("active");
  if (overlay) overlay.classList.remove("active");
}

function toggleSidebar() {
  if (!sidebar || !overlay) return;
  sidebar.classList.toggle("active");
  overlay.classList.toggle("active");
}

// ---------- Provider change ----------
function handleProviderChange() {
  if (!providerSelect) return;
  localStorage.setItem("provider", providerSelect.value);
  const label = providerSelect.options[providerSelect.selectedIndex]?.text || providerSelect.value;
  showCustomMessage(`Provider set to ${label}`);
}

// ---------- Tiny toast message ----------
function showCustomMessage(message) {
  const messageDiv = document.createElement("div");
  messageDiv.className = "fixed top-4 right-4 bg-primary-600 text-white px-4 py-2 rounded-lg shadow-lg z-[1000] flex items-center space-x-2 fade-in";
  const icon = document.createElement("i");
  icon.className = "fas fa-info-circle mr-2";
  messageDiv.appendChild(icon);
  const text = document.createElement("span");
  text.textContent = message;
  messageDiv.appendChild(text);
  document.body.appendChild(messageDiv);
  setTimeout(() => {
    messageDiv.classList.add("opacity-0", "translate-y-2");
    setTimeout(() => messageDiv.remove(), 300);
  }, 2500);
}

// ---------- Start ----------
document.addEventListener("DOMContentLoaded", init);
