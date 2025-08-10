// script.js
document.addEventListener('DOMContentLoaded', function() {
    // Get elements
    const loginBtn = document.getElementById('login-btn');
    const signupBtn = document.getElementById('signup-btn');
    const guestLink = document.getElementById('guest-link');
    
    // Login button click
    loginBtn.addEventListener('click', function() {
        // Redirect to login page or show login modal
        window.location.href = '/login'; // or showModal('login');
    });
    
    // Signup button click
    signupBtn.addEventListener('click', function() {
        // Redirect to signup page or show signup modal
        window.location.href = '/signup'; // or showModal('signup');
    });
    
    // Guest option click
    guestLink.addEventListener('click', function() {
        // Redirect directly to chat interface
        window.location.href = '/chat';
    });
});

// Optional: Function to show modal
function showModal(type) {
    // Implement modal display logic for login/signup
    console.log(`Show ${type} modal`);
}