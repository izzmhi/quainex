// Auth state management
function updateAuthUI() {
    const isLoggedIn = localStorage.getItem('authToken');
    const username = localStorage.getItem('username') || 'Guest';
    
    document.getElementById('auth-status').textContent = username;
    
    if (isLoggedIn) {
        document.getElementById('user-section').classList.remove('hidden');
        document.getElementById('login-btn').classList.add('hidden');
        document.getElementById('signup-btn').classList.add('hidden');
        document.getElementById('username-display').textContent = username;
    } else {
        document.getElementById('user-section').classList.add('hidden');
        document.getElementById('login-btn').classList.remove('hidden');
        document.getElementById('signup-btn').classList.remove('hidden');
    }
}

// Initialize auth functionality
function initAuth() {
    // Auth dropdown toggle
    document.getElementById('auth-dropdown-btn')?.addEventListener('click', () => {
        document.getElementById('auth-dropdown').classList.toggle('hidden');
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('#auth-dropdown') && !e.target.closest('#auth-dropdown-btn')) {
            document.getElementById('auth-dropdown').classList.add('hidden');
        }
    });

    // Auth modal handlers
    document.getElementById('login-btn')?.addEventListener('click', (e) => {
        e.preventDefault();
        showAuthModal('login');
    });

    document.getElementById('signup-btn')?.addEventListener('click', (e) => {
        e.preventDefault();
        showAuthModal('signup');
    });

    document.getElementById('close-auth-modal')?.addEventListener('click', () => {
        document.getElementById('auth-modal').classList.add('hidden');
    });

    // Form submission
    document.getElementById('auth-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        handleAuthSubmit();
    });

    updateAuthUI();
}

function showAuthModal(type) {
    const title = type === 'login' ? 'Log In' : 'Sign Up';
    document.getElementById('auth-modal-title').textContent = title;
    
    const formContent = `
        <div>
            <label for="auth-email" class="block text-sm font-medium mb-1">Email</label>
            <input type="email" id="auth-email" required
                class="w-full bg-dark-700 border border-dark-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500">
        </div>
        <div>
            <label for="auth-password" class="block text-sm font-medium mb-1">Password</label>
            <input type="password" id="auth-password" required
                class="w-full bg-dark-700 border border-dark-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500">
        </div>
        ${type === 'signup' ? `
        <div>
            <label for="auth-username" class="block text-sm font-medium mb-1">Username</label>
            <input type="text" id="auth-username" required
                class="w-full bg-dark-700 border border-dark-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500">
        </div>
        ` : ''}
    `;
    
    document.getElementById('auth-form-content').innerHTML = formContent;
    document.getElementById('auth-modal').classList.remove('hidden');
    document.getElementById('auth-dropdown').classList.add('hidden');
}

function handleAuthSubmit() {
    // Replace with actual API call
    localStorage.setItem('authToken', 'demo-token');
    const email = document.getElementById('auth-email').value;
    localStorage.setItem('username', email.split('@')[0]);
    
    updateAuthUI();
    document.getElementById('auth-modal').classList.add('hidden');
    showToast(`Successfully ${document.getElementById('auth-modal-title').textContent.toLowerCase()}`);
}

function showToast(message) {
    const toast = document.createElement('div');
    toast.className = "fixed top-4 right-4 bg-primary-600 text-white px-4 py-2 rounded-lg shadow-lg z-[1000] flex items-center space-x-2 fade-in";
    toast.innerHTML = `<i class="fas fa-check-circle mr-2"></i> ${message}`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initAuth);