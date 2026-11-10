import { Client } from "./vendor/gradio-client.js";
import { SPACE } from "./config.js";

const chat = document.getElementById("chat");
const empty = document.getElementById("empty");
const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("server-status");
const examples = [...document.querySelectorAll(".example")];

const SCOPE_NOTE = "This assistant only covers RA 9003, RA 6969, DAO 2013-22 and two NSWMC materials. " +
  "For other laws or local rules, check the official source or your LGU.";

let clientPromise = null;
let busy = false;

function setStatus(state, text) {
  statusEl.dataset.state = state;
  statusEl.textContent = text;
}

function onStatus(s) {
  // Hugging Face Space states: running, sleeping, building, stopped, error ...
  if (!s) return;
  if (s.status === "running") setStatus("ready", "Model server: ready");
  else if (s.status === "sleeping" || s.status === "building" || s.status === "starting")
    setStatus("waking", "Model server: waking up (1 to 2 minutes)");
  else if (s.status === "error" || s.status === "stopped")
    setStatus("error", "Model server: unavailable");
}

function getClient() {
  if (!clientPromise) {
    setStatus("waking", "Model server: connecting");
    clientPromise = Client.connect(SPACE, { status_callback: onStatus, events: ["data", "status"] })
      .then((c) => { setStatus("ready", "Model server: ready"); return c; })
      .catch((err) => { clientPromise = null; setStatus("error", "Model server: unavailable"); throw err; });
  }
  return clientPromise;
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

const LEAF = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1F6B4A" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 19c8 0 14-6 14-14C11 5 5 11 5 19z"/><path d="M5 19l7-7"/></svg>';
const SEARCH = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#7A4A00" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/><path d="M9 11h4"/></svg>';

function botShell(warn) {
  const wrap = el("article", "msg-bot");
  const avatar = el("div", "avatar" + (warn ? " warn" : ""));
  avatar.innerHTML = warn ? SEARCH : LEAF;   // static icons only, never model text
  const body = el("div", "bot-body");
  wrap.append(avatar, body);
  return { wrap, body, avatar };
}

function addUser(question) {
  empty.hidden = true;
  const row = el("div", "msg-user");
  row.append(el("p", null, question));
  chat.append(row);
}

function scrollDown() { chat.scrollTop = chat.scrollHeight; }

function renderSources(body, sources, refused) {
  if (!sources || !sources.length) return;
  body.append(el("div", "meta", refused ? "Retrieved passages (the answer was refused)" : "Sources (open one to read the passage)"));
  const list = el("div", "sources");
  for (const s of sources) {
    const d = el("details");
    const sum = el("summary");
    sum.append(el("b", null, s.document), el("span", null, `· page ${s.page} · similarity ${Number(s.score).toFixed(2)}`));
    d.append(sum, el("blockquote", null, s.passage || ""));
    list.append(d);
  }
  body.append(list);
}

function setBusy(on) {
  busy = on;
  sendBtn.disabled = on;
  examples.forEach((b) => (b.disabled = on));
}

async function ask(question) {
  question = (question || "").trim();
  if (!question || busy) return;
  setBusy(true);
  addUser(question);
  const { wrap, body, avatar } = botShell(false);
  const thinking = el("div", "thinking");
  const dots = el("span", "dots"); dots.innerHTML = "<i></i><i></i><i></i>";
  const label = el("span", null, "Searching the documents...");
  thinking.append(dots, label);
  body.append(thinking);
  chat.append(wrap);
  scrollDown();
  if (window.matchMedia("(max-width: 760px)").matches) {
    wrap.previousElementSibling.scrollIntoView({ behavior: "smooth", block: "start" });   // phones: show the new turn
  }

  const started = Date.now();
  const timer = setInterval(() => {
    const s = Math.round((Date.now() - started) / 1000);
    label.textContent = s < 5 ? "Searching the documents..." : `Writing the answer... ${s} s`;
  }, 1000);

  try {
    const client = await getClient();
    setStatus("busy", "Model server: answering");
    const res = await client.predict("/ask", [question]);
    const data = Array.isArray(res.data) ? res.data[0] : res.data;
    thinking.remove();
    if (!data || data.error) throw new Error(data && data.error ? data.error : "Empty response");
    if (data.refusal) {
      avatar.className = "avatar warn"; avatar.innerHTML = SEARCH;
      const box = el("div", "refusal");
      box.append(el("p", "answer", data.answer), el("small", null, SCOPE_NOTE));
      body.append(box);
    } else {
      body.append(el("p", "answer", data.answer));
    }
    renderSources(body, data.sources, data.refusal);
    body.append(el("div", "meta", `Answered in ${Number(data.seconds).toFixed(1)} s on the server (${data.device === "cuda" ? "GPU" : "CPU"})`));
    setStatus("ready", "Model server: ready");
  } catch (err) {
    thinking.remove();
    const box = el("div", "error");
    box.append(el("p", null, "The answer could not be loaded. The free model server may be waking up or busy. Please try again in a minute."));
    const retry = el("button", "btn", "Try again");
    retry.type = "button";
    retry.addEventListener("click", () => { wrap.previousElementSibling?.remove(); wrap.remove(); ask(question); });
    box.append(retry);
    body.append(box);
    console.error(err);
  } finally {
    clearInterval(timer);
    setBusy(false);
    scrollDown();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value;
  if (!q.trim() || busy) return;
  input.value = "";
  ask(q);
});
examples.forEach((b) => b.addEventListener("click", () => ask(b.textContent)));
document.getElementById("new-chat").addEventListener("click", () => {
  if (busy) return;
  chat.querySelectorAll(".msg-user, .msg-bot").forEach((n) => n.remove());
  empty.hidden = false;
  input.focus();
});

// Start waking the server as soon as the page opens, so the first answer is faster.
getClient().catch(() => {});
