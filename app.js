// app.js
// ---------- DOM References ----------
const loginScreen = document.getElementById("login-screen");
const chatScreen = document.getElementById("chat-screen");
const loginForm = document.getElementById("login-form");
const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");
const signupBtn = document.getElementById("signup-btn");
const logoutBtn = document.getElementById("logout-btn");
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
const mobileMenuBtn = document.getElementById("mobile-menu-btn"); // New mobile menu button
const sidebar = document.querySelector("aside"); // Reference to the sidebar
const mainSection = document.querySelector("section.flex-1"); // Reference to the main content section
const userAvatar = document.getElementById("user-avatar"); // Reference to the user avatar image

// ---------- Global Variables ----------
let currentTool = "chat";
let currentUser = null; // Stored for display purposes, not for authentication
let mediaRecorder,
  audioChunks = [];
let audio = new Audio();

// IMPORTANT: Get the backend URL from the global variable set in index.html
const backendBaseUrl = window.BACKEND_URL;

// ---------- Init ----------
function init() {
  currentUser = localStorage.getItem("quainex_user"); // Still use localStorage for username display
  const provider = localStorage.getItem("provider") || "openrouter";
  providerSelect.value = provider;

  // Check if a user is "known" from previous session (username in localStorage)
  // Then verify session with backend via cookie
  if (currentUser) {
    showChat();
    updateUserAvatar(currentUser); // Update avatar immediately if username is known
    fetchMe(); // Verify the session via backend cookie
  } else {
    loginScreen.classList.remove("hidden");
  }
}

function updateUserAvatar(username) {
  if (userAvatar) {
    userAvatar.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(
      username
    )}&background=random&color=fff`;
  }
}

// Fetches user details to verify session via cookie
async function fetchMe() {
  try {
    const res = await fetch(`${backendBaseUrl}/me`, {
      credentials: "include",
    });
    if (!res.ok) {
      // If session is expired or invalid, handle it
      console.error("Session verification failed:", res.status);
      handleTokenExpiry();
      return false;
    }
    const user = await res.json();
    currentUser = user.username; // Update currentUser from backend response
    localStorage.setItem("quainex_user", currentUser); // Keep username in localStorage for convenience
    updateUserAvatar(currentUser); // Update avatar after successful verification
    console.log("Session verified for user:", currentUser);
    return true;
  } catch (error) {
    console.error("Error fetching user details:", error);
    handleTokenExpiry();
    return false;
  }
}

// ---------- Login ----------
loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = usernameInput.value.trim();
  const password = passwordInput.value.trim();
  if (!username || !password) {
    showCustomMessage("Please enter username and password.");
    return;
  }

  try {
    // Call the /token-cookie endpoint to get an HttpOnly cookie
    const res = await fetch(`${backendBaseUrl}/token-cookie`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: `username=${encodeURIComponent(
        username
      )}&password=${encodeURIComponent(password)}`,
    });

    if (res.ok) {
      currentUser = username;
      localStorage.setItem("quainex_user", currentUser); // Store username for display
      updateUserAvatar(currentUser); // Update avatar on successful login
      showChat();
      showCustomMessage("Login successful!");
    } else {
      const data = await res.json();
      showCustomMessage("Login failed: " + (data.detail || "Invalid credentials"));
    }
  } catch (error) {
    console.error("Login error:", error);
    showCustomMessage("Login error. Please check backend connection.");
  }
});

// ---------- Signup ----------
signupBtn.addEventListener("click", async () => {
  const username = usernameInput.value.trim();
  const password = passwordInput.value.trim();
  if (!username || !password) {
    showCustomMessage("Please enter username and password.");
    return;
  }

  try {
    const res = await fetch(`${backendBaseUrl}/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    if (res.ok) {
      showCustomMessage("Signup successful! Please log in.");
    } else {
      const data = await res.json();
      showCustomMessage(data.detail || "Signup failed.");
    }
  } catch (error) {
    console.error("Signup error:", error);
    showCustomMessage("Signup error..");
  }
});

// ---------- Logout ----------
logoutBtn.addEventListener("click", async () => {
  try {
    // Call the /logout endpoint to clear the HttpOnly cookie
    await fetch(`${backendBaseUrl}/logout`, {
      method: "POST",
      credentials: "include", // ADD THIS LINE for logout
    });
  } catch (error) {
    console.error("Logout error:", error);
  }
  localStorage.clear(); // Clear any local storage data
  location.reload(); // Reload the page to show login screen
});

// ---------- Show Chat Screen ----------
function showChat() {
  loginScreen.classList.add("hidden");
  chatScreen.classList.remove("hidden");
  logoutBtn.classList.remove("hidden");
  loadHistoryOnce();
}

// ---------- Tool Switching ----------
toolBtns.forEach((btn) => {
  btn.addEventListener("click", async () => {
    currentTool = btn.dataset.tool;
    // Assuming toolStatus exists in index.html
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
      credentials: "include", // ADD THIS LINE
      body: JSON.stringify(body),
    });

    if (res.status === 401) {
      handleTokenExpiry();
      return;
    }
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
      // Check if element is still in DOM
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
    // Safely create and append an image element
    const imgElement = document.createElement("img");
    imgElement.src = text; // The 'text' here is actually the image URL
    imgElement.alt = "Generated Image";
    imgElement.className =
      "rounded-lg shadow-md mt-2 w-full max-w-xs md:max-w-sm lg:max-w-md";
    bubble.appendChild(imgElement);
  } else {
    // Use textContent for plain text to prevent XSS
    bubble.textContent = text;
  }

  wrapper.appendChild(bubble);
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;
  return wrapper;
}

// ---------- Load History Once ----------
let historyLoaded = false;
async function loadHistoryOnce() {
  if (historyLoaded) return;
  historyLoaded = true;

  try {
    const res = await fetch(`${backendBaseUrl}/history`, {
      credentials: "include", //
    });
    if (res.status === 401) {
      handleTokenExpiry();
      return;
    }
    const data = await res.json();
    data.messages.forEach((m) => {
      appendMessage("You", m.user, "user");
      appendMessage("Quainex", m.bot, "bot");
    });
  } catch (error) {
    console.error("Failed to load history:", error);
    appendMessage("Quainex", "⚠️ Failed to load history.", "bot");
  }
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
      credentials: "include", // ADD THIS LINE
      body: JSON.stringify({ text: message }),
    });

    if (res.status === 401) {
      handleTokenExpiry();
      return;
    }
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
      credentials: "include",
      body: JSON.stringify({ prompt: message }),
    });

    if (res.status === 401) {
      handleTokenExpiry();
      return;
    }
    const data = await res.json();
    loader.remove();

    if (data.image_url) {
      // Safely append the image without using innerHTML
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
          credentials: "include", // ADD THIS LINE
          body: formData, // No Content-Type header needed for FormData
        });
        if (res.status === 401) {
          handleTokenExpiry();
          return;
        }
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
  sidebar.classList.toggle("flex"); // Add flex for mobile view
  // Adjust main section width based on sidebar visibility
  if (sidebar.classList.contains("hidden")) {
    mainSection.classList.remove("md:w-auto"); // Remove fixed width
    mainSection.classList.add("w-full"); // Take full width
  } else {
    mainSection.classList.remove("w-full");
    mainSection.classList.add("md:w-auto"); // Re-add fixed width for larger screens
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

// ---------- Token Expiry (Custom Message) ----------
function handleTokenExpiry() {
  showCustomMessage("🔒 Session expired. Please log in again.");
  localStorage.clear();
  // Small delay before reloading to allow user to read message
  setTimeout(() => {
    location.reload();
  }, 1500);
}

// ---------- Custom Message Display (instead of alert) ----------
function showCustomMessage(message) {
  // For simplicity
  // In a real app, you'd implement a proper modal or toast notification system.
  const messageDiv = document.createElement("div");
  messageDiv.className =
    "fixed top-4 right-4 bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg z-[1000]";
  messageDiv.textContent = message;
  document.body.appendChild(messageDiv);

  setTimeout(() => {
    messageDiv.remove();
  }, 3000); // Message disappears after 3 seconds
}

// ---------- Run Init ----------
init();