// ── Configuration ────────────────────────────────────────────
const API_URL = "http://localhost:8000/chat";

// ── DOM Elements ─────────────────────────────────────────────
const toggle = document.getElementById("chat-toggle");
const chatWindow = document.getElementById("chat-window");
const closeBtn = document.getElementById("chat-close");
const iconOpen = document.getElementById("chat-icon-open");
const iconClose = document.getElementById("chat-icon-close");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("chat-send");
const messagesEl = document.getElementById("chat-messages");

// ── Toggle Chat Window ───────────────────────────────────────
function openChat() {
  chatWindow.classList.remove("hidden");
  iconOpen.classList.add("hidden");
  iconClose.classList.remove("hidden");
  input.focus();
}

function closeChat() {
  chatWindow.classList.add("hidden");
  iconOpen.classList.remove("hidden");
  iconClose.classList.add("hidden");
}

toggle.addEventListener("click", () => {
  chatWindow.classList.contains("hidden") ? openChat() : closeChat();
});

closeBtn.addEventListener("click", closeChat);

// ── Message Helpers ──────────────────────────────────────────
function appendMessage(text, role, sources) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;

  const p = document.createElement("p");
  p.textContent = text;
  div.appendChild(p);

  // Show sources for bot messages
  if (role === "bot" && sources && sources.length > 0) {
    const srcDiv = document.createElement("div");
    srcDiv.className = "sources";
    srcDiv.innerHTML =
      "📎 Sources: " +
      sources
        .map(
          (s) =>
            `<a href="${s.url}" target="_blank" rel="noopener">${s.title || s.url}</a>`
        )
        .join(", ");
    div.appendChild(srcDiv);
  }

  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showTyping() {
  const el = document.createElement("div");
  el.className = "typing-indicator";
  el.id = "typing";
  el.innerHTML = "<span></span><span></span><span></span>";
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById("typing");
  if (el) el.remove();
}

// ── Send Message ─────────────────────────────────────────────
async function sendMessage(question) {
  appendMessage(question, "user");
  input.value = "";
  sendBtn.disabled = true;
  showTyping();

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      throw new Error(`Server error (${res.status})`);
    }

    const data = await res.json();
    removeTyping();
    appendMessage(data.answer, "bot", data.sources);
  } catch (err) {
    removeTyping();
    appendMessage(
      "Sorry, I couldn't reach the server. Please try again later.",
      "bot"
    );
    console.error("Chat error:", err);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

// ── Form Submit ──────────────────────────────────────────────
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (q) sendMessage(q);
});
