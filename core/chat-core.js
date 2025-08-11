// core/chat-core.js
import { ChatHistory } from '../modules/history.js';
import { FileHandler } from '../modules/files.js';
import { SettingsManager } from '../modules/settings.js';
import { renderMarkdown } from '../modules/markdown.js';
import { showToast } from '../modules/utils.js';

// Export initialized modules
export const chatHistory = new ChatHistory();
export const fileHandler = new FileHandler(window.BACKEND_URL);
export const settingsManager = new SettingsManager();

// Export modified functions
export function appendMessage(sender, text, type, loading = false) {
  if (!window.chatBox) return null;
  
  const wrapper = document.createElement("div");
  wrapper.className = `flex ${type === "user" ? "justify-end" : "justify-start"} fade-in`;

  const bubble = document.createElement("div");
  bubble.className = `px-5 py-3 rounded-2xl shadow max-w-[90%] md:max-w-[80%] whitespace-pre-wrap message-bubble ${
    type === "user" 
      ? "bg-primary-600 text-white rounded-br-none user-bubble" 
      : "bg-dark-700 text-white rounded-bl-none bot-bubble"
  }`;

  const senderEl = document.createElement("div");
  senderEl.className = `text-xs font-semibold mb-1 ${
    type === "user" ? "text-primary-200" : "text-gray-400"
  }`;
  senderEl.textContent = sender;
  bubble.appendChild(senderEl);

  if (loading) {
    const typing = document.createElement("span");
    typing.classList.add("typing");
    typing.textContent = "...";
    bubble.appendChild(typing);
  } else {
    const content = document.createElement("div");
    content.className = "text-gray-100 markdown-content";
    content.innerHTML = renderMarkdown(text);
    bubble.appendChild(content);
  }

  wrapper.appendChild(bubble);
  window.chatBox.appendChild(wrapper);
  window.chatBox.scrollTop = window.chatBox.scrollHeight;
  
  chatHistory.currentConversation.messages.push({
    sender,
    text,
    type,
    timestamp: new Date().toISOString()
  });
  chatHistory.saveConversation();
  
  return wrapper;
}

export function handleSettings() {
  settingsManager.open();
  window.toggleSidebar(); // Reference global function
}
// Add this at the bottom of chat-core.js
export function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('overlay');
  if (!sidebar || !overlay) return;
  
  sidebar.classList.toggle('active');
  overlay.classList.toggle('active');
}