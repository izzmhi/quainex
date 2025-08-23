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
  settingsBtn,
  logoutBtn;

// --- Profile Modal Elements ---
let profileModal,
  profileCloseBtn,
  changeAvatarBtn,
  avatarFileInput,
  profileAvatarPreview,
  profileNameInput,
  profileSaveBtn;


let mediaRecorder;
let audioChunks = [];
let audio = new Audio();

const BACKEND_URL = window.BACKEND_URL || "https://quainex.onrender.com";

// ---------- App State ----------
const APP = {
  prefsKey: "quainex_prefs_v1",
  historyKey: "quainex_history_v1",
  userKey: "quainex_user_v1",
  currentChatId: null,
  chats: [], // array of {id, title, messages: [{role, text, ts}], createdAt}
  prefs: {
    theme: "dark",
    fontSize: 16,
    provider: "deepseek"
  },
  user: {
    name: 'Guest User',
    avatar: ''
  } // {name, avatar}
};

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
  logoutBtn = document.getElementById("logout-btn");

  // --- Sidebar buttons ---
  // Note: Using more specific IDs or classes would be more robust.
  const sidebarButtons = document.querySelectorAll("#sidebar button");
  newChatBtn = sidebarButtons[0];
  searchBtn = sidebarButtons[1];
  settingsBtn = document.querySelector("#sidebar .mt-auto button");


  // --- Profile Modal elements ---
  profileModal = document.getElementById('profile-modal');
  profileCloseBtn = document.getElementById('profile-close-btn');
  changeAvatarBtn = document.getElementById('change-avatar-btn');
  avatarFileInput = document.getElementById('avatar-file-input');
  profileAvatarPreview = document.getElementById('profile-avatar-preview');
  profileNameInput = document.getElementById('profile-name-input');
  profileSaveBtn = document.getElementById('profile-save-btn');


  // Ensure providerSelect default stored
  if (providerSelect) {
    const provider = (loadPrefs().provider) || providerSelect.value || "deepseek";
    providerSelect.value = provider;
    APP.prefs.provider = provider;
  }

  // Load preferences, user and history
  loadPrefs();
  loadUser();
  loadHistory();
  renderChatHistoryList();
  applyPreferences();
  renderUserUI();


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
    // --- Remove any existing listeners first (defensive) ---
    safeRemoveListener(chatForm, "submit", handleFormSubmit);
    safeRemoveListener(chatInput, "keydown", handleKeyPress);
    safeRemoveListener(voiceBtn, "click", handleVoiceInput);
    safeRemoveListener(mobileMenuBtn, "click", toggleSidebar);
    safeRemoveListener(overlay, "click", toggleSidebar);
    safeRemoveListener(providerSelect, "change", handleProviderChange);
    safeRemoveListener(newChatBtn, "click", handleNewChat);
    safeRemoveListener(searchBtn, "click", handleSearch);
    safeRemoveListener(settingsBtn, "click", handleSettings);
    safeRemoveListener(userAvatar, "click", openProfileModal);
    safeRemoveListener(userAvatarSidebar, "click", openProfileModal);
    safeRemoveListener(logoutBtn, "click", handleLogout);
    // Profile Modal Listeners
    safeRemoveListener(profileCloseBtn, 'click', closeProfileModal);
    safeRemoveListener(profileSaveBtn, 'click', handleProfileSave);
    safeRemoveListener(changeAvatarBtn, 'click', () => avatarFileInput.click());
    safeRemoveListener(avatarFileInput, 'change', handleAvatarFileChange);


    // --- Attach fresh listeners ---
    if (chatForm) chatForm.addEventListener("submit", handleFormSubmit);
    if (chatInput) chatInput.addEventListener("keydown", handleKeyPress);
    if (voiceBtn) voiceBtn.addEventListener("click", handleVoiceInput);
    if (mobileMenuBtn) mobileMenuBtn.addEventListener("click", toggleSidebar);
    if (overlay) overlay.addEventListener("click", toggleSidebar);
    if (providerSelect) providerSelect.addEventListener("change", handleProviderChange);
    if (newChatBtn) newChatBtn.addEventListener("click", handleNewChat);
    if (searchBtn) searchBtn.addEventListener("click", handleSearch);
    if (settingsBtn) settingsBtn.addEventListener("click", handleSettings);
    if (userAvatar) userAvatar.addEventListener("click", openProfileModal);
    if (userAvatarSidebar) userAvatarSidebar.addEventListener("click", openProfileModal);
    if (logoutBtn) logoutBtn.addEventListener("click", handleLogout);

    // --- Profile Modal ---
    if (profileCloseBtn) profileCloseBtn.addEventListener('click', closeProfileModal);
    if (profileSaveBtn) profileSaveBtn.addEventListener('click', handleProfileSave);
    if (changeAvatarBtn) changeAvatarBtn.addEventListener('click', () => avatarFileInput.click());
    if (avatarFileInput) avatarFileInput.addEventListener('change', handleAvatarFileChange);


  // Auto-resize textarea
  if (chatInput) {
    chatInput.addEventListener("input", () => {
      chatInput.style.height = "auto";
      chatInput.style.height = chatInput.scrollHeight + "px";
    });
  }
}

// ---------- Preferences (localStorage) ----------
function savePrefs() {
  try {
    localStorage.setItem(APP.prefsKey, JSON.stringify(APP.prefs));
  } catch (e) {
    console.warn("[prefs] save failed", e);
  }
}
function loadPrefs() {
  try {
    const raw = localStorage.getItem(APP.prefsKey);
    if (raw) {
      APP.prefs = Object.assign(APP.prefs, JSON.parse(raw));
    }
  } catch (e) {
    console.warn("[prefs] load failed", e);
  }
  return APP.prefs;
}

function applyPreferences() {
  // Theme
  document.documentElement.dataset.theme = APP.prefs.theme || "dark";
  if (APP.prefs.theme === "light") {
    document.body.classList.remove("bg-dark-900");
    document.body.classList.add("bg-white");
    document.body.style.color = "#111";
  } else {
    document.body.classList.remove("bg-white");
    document.body.classList.add("bg-dark-900");
    document.body.style.color = "";
  }

  // Font size
  if (chatBox) chatBox.style.fontSize = (APP.prefs.fontSize || 16) + "px";

  // Provider
  if (providerSelect) providerSelect.value = APP.prefs.provider || providerSelect.value;
}

// ---------- User (profile) ----------
function saveUser() {
  try {
    localStorage.setItem(APP.userKey, JSON.stringify(APP.user));
  } catch (e) {
    console.warn("[user] save failed", e);
  }
}
function loadUser() {
  try {
    const raw = localStorage.getItem(APP.userKey);
    if (raw) {
      APP.user = JSON.parse(raw);
    }
  } catch (e) {
    console.warn("[user] load failed", e);
  }
  return APP.user;
}

function renderUserUI() {
    if (!APP.user) {
        APP.user = { name: 'Guest User', avatar: '' };
    }
    const isLogged = APP.user && APP.user.name && APP.user.name !== 'Guest User';
    const avatarSrc = APP.user.avatar || defaultAvatarFor(APP.user.name);

    if (userAvatar) userAvatar.src = avatarSrc;
    if (userAvatarSidebar) userAvatarSidebar.src = avatarSrc;

    const sidebarNameEl = document.querySelector('#sidebar .text-sm.font-medium.text-white');
    if (sidebarNameEl) sidebarNameEl.textContent = APP.user.name;

    if (logoutBtn) logoutBtn.classList.toggle('hidden', !isLogged);
}


function defaultAvatarFor(name) {
  const initial = name && name.length ? name[0].toUpperCase() : 'G';
  return `https://ui-avatars.com/api/?name=${initial}&background=22c55e&color=fff`;
}

function handleLogout() {
  APP.user = { name: 'Guest User', avatar: '' };
  saveUser();
  renderUserUI();
  showCustomMessage('Logged out');
}

// --- NEW/UPDATED --- Profile Modal Functions
function openProfileModal() {
    if (!profileModal) return;
    // Populate modal with current user data
    profileNameInput.value = APP.user.name === 'Guest User' ? '' : APP.user.name;
    profileAvatarPreview.src = APP.user.avatar || defaultAvatarFor(APP.user.name);
    // Show the modal
    profileModal.classList.remove('hidden');
    profileModal.classList.add('flex');
}

function closeProfileModal() {
    if (!profileModal) return;
    profileModal.classList.add('hidden');
    profileModal.classList.remove('flex');
}

function handleAvatarFileChange(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            profileAvatarPreview.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
}

function handleProfileSave() {
    const name = profileNameInput.value.trim();
    if (!name) {
        showCustomMessage('Please enter a display name.');
        return;
    }
    APP.user.name = name;
    // The avatar is a data URL from the file reader or existing URL
    APP.user.avatar = profileAvatarPreview.src;
    saveUser();
    renderUserUI();
    closeProfileModal();
    showCustomMessage('Profile saved successfully!');
}


// ---------- History (localStorage + UI) ----------
function loadHistory() {
  try {
    const raw = localStorage.getItem(APP.historyKey);
    if (raw) {
      APP.chats = JSON.parse(raw);
    } else {
      APP.chats = [];
    }
  } catch (e) {
    console.warn('[history] load failed', e);
    APP.chats = [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(APP.historyKey, JSON.stringify(APP.chats));
  } catch (e) {
    console.warn('[history] save failed', e);
  }
}

function createNewChat(initialTitle) {
  const id = 'chat_' + Date.now();
  const chat = { id, title: initialTitle || 'New Chat', messages: [], createdAt: Date.now() };
  APP.chats.unshift(chat);
  APP.currentChatId = id;
  saveHistory();
  renderChatHistoryList();
  // Set the new chat as active but don't clear the view yet
  return chat;
}


function saveMessageToCurrentChat(role, text) {
  let chat = APP.chats.find(c => c.id === APP.currentChatId);
  if (!chat) {
    chat = createNewChat(text.substring(0, 40));
  }
  chat.messages.push({ role, text, ts: Date.now() });
  // update title to first user message
  const firstUser = chat.messages.find(m => m.role === 'user');
  if (firstUser && chat.title === 'New Chat') {
      chat.title = firstUser.text.substring(0, 60);
  }
  saveHistory();
  renderChatHistoryList();
}


function renderChatHistoryList() {
  const container = document.getElementById('chat-history');
  if (!container) return;
  container.innerHTML = '';
  APP.chats.slice(0, 50).forEach(chat => {
    const btn = document.createElement('button');
    btn.className = 'w-full text-left p-2.5 rounded-lg hover:bg-dark-700 text-gray-300 hover:text-white transition-colors flex items-center space-x-3';
    btn.textContent = chat.title || 'Chat';
    btn.addEventListener('click', () => {
        loadChat(chat.id);
        if (sidebar && overlay) {
            sidebar.classList.remove('active');
            overlay.classList.remove('active');
        }
    });
    container.appendChild(btn);
  });
}

function loadChat(chatId) {
  const chat = APP.chats.find(c => c.id === chatId);
  if (!chat) return;
  APP.currentChatId = chatId;
  if (chatBox) chatBox.innerHTML = '';
  chat.messages.forEach(m => appendMessage(m.role === 'user' ? (APP.user.name || 'You') : 'Quainex', m.text, m.role === 'user' ? 'user' : 'bot'));
  if (welcomeScreen) welcomeScreen.style.display = 'none';
  if (chatBox) chatBox.classList.remove('hidden');
}

// ---------- Submit handler (enhanced) ----------
async function handleFormSubmit(e) {
  if (e && e.preventDefault) e.preventDefault();
  if (e && e.stopPropagation) e.stopPropagation();

  if (!chatInput) return;
  const message = chatInput.value.trim();
  if (!message) return;

  console.log('[handleFormSubmit] message:', message);

  // Disable submit button if present
  const submitBtn = chatForm ? chatForm.querySelector('button[type="submit"]') : null;
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.classList.add('opacity-50', 'cursor-not-allowed');
  }

  // Show chat area if it was hidden
  if (welcomeScreen && chatBox) {
    welcomeScreen.style.display = 'none';
    chatBox.classList.remove('hidden');
  }

  // If this is the first message of a new chat, create it.
  if (!APP.currentChatId || APP.chats.find(c => c.id === APP.currentChatId).messages.length === 0) {
      if (!APP.chats.find(c => c.id === APP.currentChatId)) {
          createNewChat(message.substring(0, 40));
      }
  }


  appendMessage(APP.user.name || 'You', message, 'user');
  saveMessageToCurrentChat('user', message);
  chatInput.value = '';
  chatInput.style.height = 'auto';

  // Add loader message
  const loader = appendMessage('Quainex', '...', 'bot', true);
  showTyping(loader.querySelector('.typing'));

  try {
    const payload = {
      message,
      provider: providerSelect ? providerSelect.value : undefined,
      personality: 'default'
    };

    console.log('[handleFormSubmit] POST ->', `${BACKEND_URL}/api/chat`, payload);

    const response = await fetch(`${BACKEND_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json'
      },
      credentials: 'include',
      body: JSON.stringify(payload)
    });

    console.log('[handleFormSubmit] status:', response.status);

    // try to parse JSON (if possible)
    let data = null;
    try {
      data = await response.json();
    } catch (err) {
      console.warn('[handleFormSubmit] response not JSON');
    }

    loader.remove();

    if (!response.ok) {
      const errMsg = (data && (data.error || data.message)) || `HTTP ${response.status}`;
      throw new Error(errMsg);
    }

    // Support multiple possible reply fields
    const reply =
      (data && (data.response || data.reply || data.answer || data.data || data.text)) ||
      'No reply from server.';

    appendMessage('Quainex', reply, 'bot');
    saveMessageToCurrentChat('bot', reply);
    console.log('[handleFormSubmit] reply:', reply);
  } catch (err) {
    console.error('[handleFormSubmit] error:', err);
    try { loader.remove(); } catch (e) {}
    const errText = `⚠️ Error: ${err.message || 'Please try again later'}`;
    appendMessage('Quainex', errText, 'bot');
    saveMessageToCurrentChat('bot', errText);
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    }
  }
}

// ---------- Keyboard: Enter to submit (Shift+Enter -> newline) ----------
function handleKeyPress(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    // prefer requestSubmit when available
    if (chatForm && typeof chatForm.requestSubmit === 'function') {
      chatForm.requestSubmit();
    } else if (chatForm) {
      chatForm.dispatchEvent(new Event('submit', { cancelable: true }));
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
    el.textContent = '.'.repeat(dots++ % 4);
  }, 300);
}

// ---------- Append message bubble ----------
function appendMessage(sender, text, type = 'bot', loading = false) {
  if (!chatBox) return null;
  const wrapper = document.createElement('div');
  wrapper.className = `flex ${type === 'user' ? 'justify-end' : 'justify-start'} fade-in`;

  const bubble = document.createElement('div');
  bubble.className =
    'px-5 py-3 rounded-2xl shadow max-w-[90%] md:max-w-[80%] whitespace-pre-wrap message-bubble ' +
    (type === 'user' ? 'bg-primary-600 text-white rounded-br-none user-bubble' : 'bg-dark-700 text-white rounded-bl-none bot-bubble');

  const senderEl = document.createElement('div');
  senderEl.className = `text-xs font-semibold mb-1 ${type === 'user' ? 'text-primary-200' : 'text-gray-400'}`;
  senderEl.textContent = sender;
  bubble.appendChild(senderEl);

  if (loading) {
    const typing = document.createElement('span');
    typing.classList.add('typing');
    typing.textContent = '...';
    bubble.appendChild(typing);
  } else {
    const content = document.createElement('div');
    content.className = 'text-gray-100';
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
    showCustomMessage('Microphone not supported in this browser.');
    return;
  }

  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    if (voiceBtn) {
      voiceBtn.classList.remove('text-red-500');
      voiceBtn.classList.add('text-gray-400');
    }
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');

      // Append user's prompt immediately
      const userMessage = "🎤 Voice Input";
      appendMessage(APP.user.name || 'You', userMessage, 'user');
      saveMessageToCurrentChat('user', userMessage);

      const loader = appendMessage('Quainex', 'Transcribing & thinking...', 'bot', true);
      showTyping(loader.querySelector('.typing'));

      try {
        // This assumes your backend `/voice` endpoint transcribes and then calls the chat API.
        const res = await fetch(`${BACKEND_URL}/voice`, {
          method: 'POST',
          body: formData,
          credentials: 'include'
        });

        let data = null;
        try { data = await res.json(); } catch (e) {}

        loader.remove();
        if (res.ok && data && (data.response || data.transcript || data.text)) {
          const reply = data.response || data.transcript || data.text;
          appendMessage('Quainex', reply, 'bot');
          saveMessageToCurrentChat('bot', reply);
        } else {
            const errorMsg = '⚠️ Voice transcription failed.';
            appendMessage('Quainex', errorMsg, 'bot');
            saveMessageToCurrentChat('bot', errorMsg);
        }
      } catch (err) {
        console.error('[voice] error:', err);
        try { loader.remove(); } catch (e) {}
        const errorMsg = '⚠️ Voice request failed.';
        appendMessage('Quainex', errorMsg, 'bot');
        saveMessageToCurrentChat('bot', errorMsg);
      }
    };

    mediaRecorder.start();
    if (voiceBtn) {
      voiceBtn.classList.remove('text-gray-400');
      voiceBtn.classList.add('text-red-500');
    }
    showCustomMessage('Recording... click the mic again to stop');
  } catch (err) {
    console.error('[voice] mic access error:', err);
    showCustomMessage('Microphone access denied: ' + (err.message || ''));
  }
}

// ---------- Sidebar handlers ----------
function handleNewChat() {
  if (chatBox) chatBox.innerHTML = '';
  if (chatBox) chatBox.classList.add('hidden');
  if (welcomeScreen) welcomeScreen.style.display = 'flex'; // Use flex for centering
  APP.currentChatId = null; // Unset current chat ID
  createNewChat(); // This will create a new chat object and set it as current
  showCustomMessage('Started a new chat');
  toggleSidebar(true); // Force close sidebar
}


function handleSearch() {
  openSearchModal();
  toggleSidebar(true); // Force close sidebar
}

function handleSettings() {
  openSettingsModal();
  toggleSidebar(true); // Force close sidebar
}

// ========== THIS IS THE CORRECTED FUNCTION ==========
function toggleSidebar(forceClose = false) {
    if (!sidebar || !overlay) return;

    const isActive = sidebar.classList.contains('active');

    if (forceClose || isActive) {
        // If we need to force it close, or if it's currently active, DEACTIVATE
        sidebar.classList.remove('active');
        overlay.classList.remove('active');
    } else {
        // Otherwise (it's inactive and we are not forcing it close), ACTIVATE
        sidebar.classList.add('active');
        overlay.classList.add('active');
    }
}


// ---------- Provider change ----------
function handleProviderChange() {
  if (!providerSelect) return;
  APP.prefs.provider = providerSelect.value;
  savePrefs();
  const label = providerSelect.options[providerSelect.selectedIndex]?.text || providerSelect.value;
  showCustomMessage(`Provider set to ${label}`);
}

// ---------- Tiny toast message ----------
function showCustomMessage(message) {
  // Remove any existing toasts
  document.querySelectorAll('.custom-toast').forEach(t => t.remove());

  const messageDiv = document.createElement('div');
  messageDiv.className = 'custom-toast fixed top-4 right-4 bg-primary-600 text-white px-4 py-2 rounded-lg shadow-lg z-[1000] flex items-center space-x-2 fade-in';

  const icon = document.createElement('i');
  icon.className = 'fas fa-info-circle mr-2';
  messageDiv.appendChild(icon);

  const text = document.createElement('span');
  text.textContent = message;
  messageDiv.appendChild(text);

  document.body.appendChild(messageDiv);
  setTimeout(() => {
    messageDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    messageDiv.style.opacity = '0';
    messageDiv.style.transform = 'translateY(-20px)';
    setTimeout(() => messageDiv.remove(), 300);
  }, 2500);
}


// ---------- Settings Modal ----------
function openSettingsModal() {
  const modalRoot = document.getElementById('settings-modal');
  if (!modalRoot) return;
  // Check if content already exists to prevent re-creating
  if (!document.getElementById('settings-modal-content')) {
      modalRoot.innerHTML = `
        <div id="settings-modal-content" class="bg-dark-800 rounded-2xl w-full max-w-2xl p-6 border border-dark-700">
          <div class="flex justify-between items-center mb-4">
            <h3 class="text-lg font-semibold">Settings</h3>
            <button id="settings-close" class="px-2 py-1 rounded bg-dark-700">&times;</button>
          </div>
          <div class="space-y-4">
            <div>
              <label class="block text-sm mb-1">Theme</label>
              <select id="settings-theme" class="w-full rounded px-3 py-2 bg-dark-700 border border-dark-600">
                <option value="dark">Dark</option>
                <option value="light">Light</option>
              </select>
            </div>
            <div>
              <label class="block text-sm mb-1">Chat font size</label>
              <input id="settings-fontsize" type="number" min="12" max="24" class="w-full rounded px-3 py-2 bg-dark-700 border border-dark-600" />
            </div>
            <div>
              <label class="block text-sm mb-1">Default provider</label>
              <select id="settings-provider" class="w-full rounded px-3 py-2 bg-dark-700 border border-dark-600">
                <option value="deepseek">BRILUX</option>
                <option value="ensemble">IZZMHI</option>
              </select>
            </div>
            <div class="flex justify-end mt-4">
              <button id="settings-save" class="px-3 py-1 rounded bg-primary-600 text-white">Save & Close</button>
            </div>
          </div>
        </div>
      `;
  }

  // Always show the modal
  modalRoot.classList.remove('hidden');


  // Populate values every time it opens
  document.getElementById('settings-theme').value = APP.prefs.theme || 'dark';
  document.getElementById('settings-fontsize').value = APP.prefs.fontSize || 16;
  document.getElementById('settings-provider').value = APP.prefs.provider || 'deepseek';

  // Attach event listeners (idempotent due to check above)
  const closeBtn = document.getElementById('settings-close');
  const saveBtn = document.getElementById('settings-save');

  // To avoid adding multiple listeners, we remove the old one if it exists.
  const newSaveBtn = saveBtn.cloneNode(true);
  saveBtn.parentNode.replaceChild(newSaveBtn, saveBtn);
  newSaveBtn.addEventListener('click', () => {
    APP.prefs.theme = document.getElementById('settings-theme').value;
    APP.prefs.fontSize = parseInt(document.getElementById('settings-fontsize').value, 10) || 16;
    APP.prefs.provider = document.getElementById('settings-provider').value;
    savePrefs();
    applyPreferences();
    modalRoot.classList.add('hidden');
    showCustomMessage('Settings saved');
  });

  const newCloseBtn = closeBtn.cloneNode(true);
  closeBtn.parentNode.replaceChild(newCloseBtn, closeBtn);
  newCloseBtn.addEventListener('click', () => modalRoot.classList.add('hidden'));
}


// ---------- Search Modal ----------
function openSearchModal() {
  let modal = document.getElementById('search-modal');
  if (!modal) {
      modal = document.createElement('div');
      modal.id = 'search-modal';
      modal.className = 'fixed inset-0 bg-black bg-opacity-50 z-50 hidden items-center justify-center p-4';
      modal.innerHTML = `
        <div class="bg-dark-800 rounded-2xl w-full max-w-2xl p-4 border border-dark-700 flex flex-col">
          <div class="flex items-center gap-3 mb-3 flex-shrink-0">
            <input id="search-input" placeholder="Search chats by title or content..." class="flex-1 rounded px-3 py-2 bg-dark-700 border border-dark-600" />
            <button id="search-close" class="px-3 py-2 rounded bg-dark-700">Close</button>
          </div>
          <div id="search-results" class="space-y-2 max-h-[70vh] overflow-y-auto"></div>
        </div>
      `;
      document.body.appendChild(modal);

      // Add listeners only once
      const input = document.getElementById('search-input');
      const results = document.getElementById('search-results');
      const closeBtn = document.getElementById('search-close');

      closeBtn.addEventListener('click', () => modal.classList.add('hidden'));
      input.addEventListener('input', (e) => {
        const q = e.target.value.trim().toLowerCase();
        results.innerHTML = '';
        if (!q) return;

        const matchingChats = APP.chats.filter(chat => {
            const inTitle = chat.title && chat.title.toLowerCase().includes(q);
            const inMessages = chat.messages.some(m => m.text && m.text.toLowerCase().includes(q));
            return inTitle || inMessages;
        });

        if (matchingChats.length === 0) {
            results.innerHTML = `<div class="text-center text-gray-400 p-4">No results found.</div>`;
        } else {
            matchingChats.forEach(chat => {
                const el = document.createElement('div');
                el.className = 'p-3 rounded-lg hover:bg-dark-700 cursor-pointer';
                el.innerHTML = `<div class="font-semibold text-white">${escapeHtml(chat.title)}</div><div class="text-xs text-gray-400 mt-1">${new Date(chat.createdAt).toLocaleString()}</div>`;
                el.addEventListener('click', () => {
                    loadChat(chat.id);
                    modal.classList.add('hidden');
                    toggleSidebar(true);
                });
                results.appendChild(el);
            });
        }
      });
  }
  // Show the modal
  modal.classList.remove('hidden');
  modal.classList.add('flex');
  document.getElementById('search-input').focus();
  // Clear previous search results
  document.getElementById('search-input').value = '';
  document.getElementById('search-results').innerHTML = '';
}


// ---------- Utilities ----------
function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/[&<>"']/g, (s) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"})[s]);
}

// ---------- Start ----------
document.addEventListener('DOMContentLoaded', init);