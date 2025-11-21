// static/js/auth.js

// Helper: get CSRF token from cookie (if you use Django CSRF)
function getCookie(name) {
  const v = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
  return v ? v.pop() : '';
}

// Show messages
function show(msg) {
  const el = document.getElementById('msg');
  if (el) el.innerText = msg;
}

// Login button
document.getElementById('loginBtn').addEventListener('click', async () => {
  const username = document.getElementById('username').value.trim();
  const password = document.getElementById('password').value;

  const csrftoken = getCookie('csrftoken'); // if you use CSRF cookie

  const resp = await fetch('/auth/login/', {
    method: 'POST',
    credentials: 'include', // <--- very important: include cookies in requests
    headers: {
      'Content-Type': 'application/json',
      // 'X-CSRFToken': csrftoken, // uncomment if you enforce CSRF
    },
    body: JSON.stringify({ username, password })
  });

  const data = await resp.json();
  if (resp.ok) {
    show('Login success! Redirecting to profile...');
    window.location.href = '/profile/';
  } else {
    show('Login failed: ' + (data.message || data.error || JSON.stringify(data)));
  }
});

// Generic fetch wrapper that automatically attempts refresh on access_expired
async function fetchWithAuth(url, options = {}, retry = true) {
  options.credentials = 'include'; // ensure cookies are sent
  options.headers = options.headers || {};
  // include CSRF if needed:
  const csrftoken = getCookie('csrftoken');
  if (csrftoken) options.headers['X-CSRFToken'] = csrftoken;

  const resp = await fetch(url, options);
  if (resp.status === 401) {
    // attempt to read body for access_expired
    let body = {};
    try { body = await resp.json(); } catch(e) {}
    if (body.error === 'access_expired' && retry) {
      // call refresh endpoint
      const r = await fetch('/auth/refresh/', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' }
      });
      if (r.ok) {
        // retry original request once
        return fetchWithAuth(url, options, false);
      } else {
        // refresh failed -> force logout / redirect to login
        window.location.href = '/login/';
        return resp;
      }
    } else {
      // other 401
      return resp;
    }
  }
  return resp;
}

// Example usage of fetchWithAuth:
async function loadProfile() {
  const resp = await fetchWithAuth('/profile/', { method: 'GET' });
  if (resp.ok) {
    const data = await resp.json();
    console.log('profile', data);
  } else {
    const body = await resp.json().catch(() => ({}));
    console.warn('profile fetch failed', resp.status, body);
  }
}

// Optionally call loadProfile() when on profile page
