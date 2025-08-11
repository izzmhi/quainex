import { marked } from 'https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js';
import DOMPurify from 'https://cdn.jsdelivr.net/npm/dompurify@3.0.6/dist/purify.es.min.js';

export function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text));
}

export function initMarkdown() {
  document.addEventListener('DOMContentLoaded', () => {
    const style = document.createElement('link');
    style.rel = 'stylesheet';
    style.href = 'styles/markdown.css';
    document.head.appendChild(style);
  });
}