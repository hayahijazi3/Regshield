const BACKEND_URL = "http://127.0.0.1:5001"; 
const REDIRECT_AFTER_LOGIN = "index.html";

function getRedirectPath() {
  const u = new URL(window.location.href);
  const from = u.searchParams.get("from");
  return from || "/case";
}

const $ = (sel) => document.querySelector(sel);

const form = $("#loginForm");
const email = $("#email");
const password = $("#password");
const remember = $("#remember");
const submitBtn = $("#submitBtn");
const errBox = $("#errorBox");
const togglePassBtn = document.querySelector(".toggle-pass");

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.innerHTML = isLoading
    ? '<span class="spinner" aria-hidden="true"></span> Logging in…'
    : '<span class="btn-text">Log In</span>';
}

function showError(msg) {
  errBox.textContent = msg;
  errBox.setAttribute("data-show", "true");
}
function clearError() {
  errBox.textContent = "";
  errBox.setAttribute("data-show", "false");
}

togglePassBtn.addEventListener("click", () => {
  const isHidden = password.type === "password";
  password.type = isHidden ? "text" : "password";
  togglePassBtn.textContent = isHidden ? "Hide" : "Show";
  togglePassBtn.setAttribute("aria-label", isHidden ? "Hide password" : "Show password");
  password.focus();
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();

  if (!email.value.trim() || !password.value.trim()) {
    showError("Please enter both email and password.");
    return;
  }

  setLoading(true);
  try {
    const res = await fetch(`${BACKEND_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: email.value.trim(), password: password.value })
    });

    let data = {};
    try { data = await res.json(); } catch (_) {}

    if (!res.ok) {
      const msg = (data && (data.msg || data.error)) || "Login failed. Please check your credentials.";
      throw new Error(msg);
    }

    const token = data.access_token || data.token || "";
    if (!token) throw new Error("Login succeeded but no token was returned.");

    if (remember.checked) {
      localStorage.setItem("token", token);
      sessionStorage.removeItem("token");
    } else {
      sessionStorage.setItem("token", token);
      localStorage.removeItem("token");
    }

  window.location.assign("index.html");
  } catch (err) {
    showError(err.message || "Something went wrong. Please try again.");
  } finally {
    setLoading(false);
  }
});
