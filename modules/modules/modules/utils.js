// Shared utility functions
export function safeRemoveListener(el, evt, fn) {
  if (!el || !fn) return;
  try {
    el.removeEventListener(evt, fn);
  } catch (e) {
    /* ignore */
  }
}

export function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}