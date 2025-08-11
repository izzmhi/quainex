export class SettingsManager {
  constructor() {
    this.modal = document.getElementById('settings-modal');
    this.initSettings();
  }

  initSettings() {
    // Close button
    document.getElementById('close-settings').addEventListener('click', () => {
      this.modal.classList.add('hidden');
    });

    // Theme switcher
    document.getElementById('theme-select').addEventListener('change', (e) => {
      this.setTheme(e.target.value);
    });
  }

  setTheme(theme) {
    document.documentElement.classList.remove('dark', 'light');
    document.documentElement.classList.add(theme);
    localStorage.setItem('theme', theme);
  }

  openSettings() {
    this.modal.classList.remove('hidden');
  }
}