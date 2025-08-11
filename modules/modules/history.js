export class ChatHistory {
  constructor() {
    this.currentConversation = this.createNewConversation();
  }

  createNewConversation() {
    return {
      id: Date.now(),
      title: "New Chat",
      messages: [],
      createdAt: new Date().toISOString()
    };
  }

  saveConversation() {
    const conversations = this.getConversations();
    const existingIndex = conversations.findIndex(c => c.id === this.currentConversation.id);
    
    if (existingIndex >= 0) {
      conversations[existingIndex] = this.currentConversation;
    } else {
      conversations.push(this.currentConversation);
    }
    
    localStorage.setItem('conversations', JSON.stringify(conversations));
    this.renderChatHistory();
  }

  getConversations() {
    return JSON.parse(localStorage.getItem('conversations') || []);
  }

  renderChatHistory() {
    const historyContainer = document.getElementById('chat-history');
    if (!historyContainer) return;
    
    historyContainer.innerHTML = this.getConversations().map(conv => `
      <div class="conversation-item" data-id="${conv.id}">
        <i class="fas fa-comment-alt"></i>
        <span class="conversation-title">${conv.title}</span>
      </div>
    `).join('');
  }
}