const BACKEND_URL = "http://127.0.0.1:5001";
const REDIRECT_AFTER_LOGIN = "index.html";

(function keepFromParamOnLoginLink(){
  const a = document.getElementById("loginLink");
  if (!a) return;
  const u = new URL(window.location.href);
  const from = u.searchParams.get("from");
  if (from) a.href = `login.html?from=${encodeURIComponent(from)}`;
})();

function getRedirectPath() {
  const u = new URL(window.location.href);
  const from = u.searchParams.get("from");
  return from || REDIRECT_AFTER_LOGIN;
}

const $ = (sel) => document.querySelector(sel);

const form = $("#signupForm");
const username = $("#username");
const email = $("#email");
const password = $("#password");
const confirmPw = $("#confirm");
const remember = $("#remember");
const submitBtn = $("#submitBtn");
const errBox = $("#errorBox");
const togglePassBtn = document.querySelector(".toggle-pass");
const togglePassBtn2 = document.querySelector(".toggle-pass-2");

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.innerHTML = isLoading
    ? '<span class="spinner" aria-hidden="true"></span> Creating account…'
    : '<span class="btn-text">Create account</span>';
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
togglePassBtn2.addEventListener("click", () => {
  const isHidden = confirmPw.type === "password";
  confirmPw.type = isHidden ? "text" : "password";
  togglePassBtn2.textContent = isHidden ? "Hide" : "Show";
  togglePassBtn2.setAttribute("aria-label", isHidden ? "Hide password" : "Show password");
  confirmPw.focus();
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();

  const u = username.value.trim();
  const em = email.value.trim();
  const pw = password.value;
  const pw2 = confirmPw.value;

  if (!u || !em || !pw || !pw2) {
    showError("Please fill in all fields.");
    return;
  }
  if (pw.length < 8) {
    showError("Password must be at least 8 characters.");
    return;
  }
  if (pw !== pw2) {
    showError("Passwords do not match.");
    return;
  }

  setLoading(true);
  try {
    const res = await fetch(`${BACKEND_URL}/auth/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: u, email: em, password: pw })
    });

    let data = {};
    try { data = await res.json(); } catch (_) {}

    if (!res.ok) {
      const msg = (data && (data.msg || data.error)) || "Signup failed. Please try again.";
      throw new Error(msg);
    }

    const res2 = await fetch(`${BACKEND_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: em, password: pw })
    });
    let data2 = {};
    try { data2 = await res2.json(); } catch (_) {}

    if (!res2.ok) {
      const msg = (data2 && (data2.msg || data2.error)) || "Account created, but login failed.";
      throw new Error(msg);
    }

    const token = data2.access_token || data2.token || "";
    if (!token) throw new Error("Login succeeded but no token was returned.");

    if (remember.checked) {
      localStorage.setItem("token", token);
      sessionStorage.removeItem("token");
    } else {
      sessionStorage.setItem("token", token);
      localStorage.removeItem("token");
    }

    window.location.assign(getRedirectPath());
  } catch (err) {
    showError(err.message || "Something went wrong. Please try again.");
  } finally {
    setLoading(false);
  }
});
