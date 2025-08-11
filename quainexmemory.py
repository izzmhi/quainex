# quainex_memory.py

class MemoryManager:
    def __init__(self, limit=10):
        self.limit = limit
        self.memory = {}  # {conversation_id: [{"role": "user/assistant", "content": "..."}]}

    def add_message(self, conversation_id: str, role: str, content: str):
        """Add a message to the conversation history."""
        if conversation_id not in self.memory:
            self.memory[conversation_id] = []
        self.memory[conversation_id].append({"role": role, "content": content})
        self.memory[conversation_id] = self.memory[conversation_id][-self.limit:]  # keep last N

    def get_history(self, conversation_id: str):
        """Get stored history for a conversation."""
        return self.memory.get(conversation_id, [])
