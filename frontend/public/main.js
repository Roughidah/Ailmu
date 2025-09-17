const API_BASE = 'http://127.0.0.1:5000';

let currentChatId = null;

const els = {
  level: document.getElementById('level'),
  subject: document.getElementById('subject'),
  newChat: document.getElementById('newChat'),
  userId: document.getElementById('userId'),
  chatList: document.getElementById('chatList'),
  chatHeader: document.getElementById('chatHeader'),
  messages: document.getElementById('messages'),
  question: document.getElementById('question'),
  send: document.getElementById('send'),
};

function msgEl(role, content) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = content;
  return div;
}

async function listChats() {
  const uid = els.userId.value.trim();
  const res = await fetch(`${API_BASE}/chats?user_external_id=${encodeURIComponent(uid)}`);
  const data = await res.json();
  els.chatList.innerHTML = '';
  if (!data.ok) return;
  for (const c of data.chats) {
    const li = document.createElement('li');
    li.textContent = `${c.title} · ${new Date(c.created_at).toLocaleString()}`;
    li.onclick = () => openChat(c.id, c.title, c.level, c.subject);
    els.chatList.appendChild(li);
  }
}

async function openChat(id, title, level, subject) {
  currentChatId = id;
  els.chatHeader.textContent = `${title} — ${level} · ${subject}`;
  els.messages.innerHTML = '';
  const res = await fetch(`${API_BASE}/history?chat_id=${id}`);
  const data = await res.json();
  if (!data.ok) return;
  for (const m of data.messages) {
    if (m.role === 'system') continue;
    els.messages.appendChild(msgEl(m.role, m.content));
  }
  els.messages.scrollTop = els.messages.scrollHeight;
}

els.newChat.onclick = async () => {
  const uid = els.userId.value.trim() || 'guest';
  const level = els.level.value;
  const subject = els.subject.value;
  // start a chat by sending a first dummy user message "Hello" or wait until they click send?
  // We'll simply set header and wait for the first send.
  currentChatId = null;
  els.chatHeader.textContent = `${subject} (${level})`;
  els.messages.innerHTML = '';
};

els.send.onclick = async () => {
  const q = els.question.value.trim();
  if (!q) return;
  const uid = els.userId.value.trim() || 'guest';
  const level = els.level.value;
  const subject = els.subject.value;

  els.messages.appendChild(msgEl('user', q));
  els.question.value = '';
  els.send.disabled = true;
  const body = { user_external_id: uid, level, subject, question: q, chat_id: currentChatId };
  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.ok) {
      currentChatId = data.chat_id;
      els.messages.appendChild(msgEl('assistant', data.answer));
      await listChats();
      els.messages.scrollTop = els.messages.scrollHeight;
    } else {
      els.messages.appendChild(msgEl('assistant', 'Error: ' + (data.error || 'Unknown')));
    }
  } catch (e) {
    els.messages.appendChild(msgEl('assistant', 'Network or server error.'));
  } finally {
    els.send.disabled = false;
  }
};

// Initial load
listChats();
