// guest-auth.js

// Function to generate a unique ID for a guest user
function generateGuestId() {
  // A simple way to generate a unique ID
  return 'guest_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Function to get the guest ID from local storage or create a new one
function getGuestId() {
  let guestId = localStorage.getItem('quainex-guest-id');
  if (!guestId) {
    guestId = generateGuestId();
    localStorage.setItem('quainex-guest-id', guestId);
  }
  return guestId;
}

// Call this function to initialize the guest user on page load
function initializeGuestUser() {
  const guestId = getGuestId();
  console.log(`[Guest Auth] Initialized guest user with ID: ${guestId}`);
  // You can set this ID in a global variable or use it to populate the user avatar/name
  window.currentUser = { id: guestId, name: "Guest User" };
}

// Export the function so it can be called from app.js
export { initializeGuestUser, getGuestId };