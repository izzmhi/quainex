export class FileHandler {
  constructor(backendUrl) {
    this.backendUrl = backendUrl;
    this.initUploadButton();
  }

  initUploadButton() {
    const uploadBtn = document.getElementById('upload-btn');
    if (!uploadBtn) return;

    uploadBtn.addEventListener('click', () => {
      document.getElementById('file-upload').click();
    });

    document.getElementById('file-upload').addEventListener('change', (e) => this.handleFileUpload(e));
  }

  async handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch(`${this.backendUrl}/upload`, {
        method: 'POST',
        body: formData
      });
      
      return await response.json();
    } catch (err) {
      console.error('File upload error:', err);
      throw err;
    }
  }
}