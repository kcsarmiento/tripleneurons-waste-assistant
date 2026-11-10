"""TripleNeurons Waste Segregation Assistant - Hugging Face Space backend.

Runs the same RAG pipeline that Notebook 2 built and Notebook 3 evaluated:
C4 chunks (800/0) in Chroma, all-MiniLM-L6-v2 embeddings, Top-3 retrieval,
Qwen2.5-1.5B-Instruct with greedy decoding, shared code in rag_core.py.

It serves two things:
  * a Gradio page (the Notebook 4 interface), and
  * the API endpoint "/ask" used by the Vercel web page.
On the free CPU hardware the model runs in float32 instead of the float16 used on the
Colab T4 GPU, so a few answers can differ slightly in wording from the evaluated run.
"""
import datetime
import json
import os
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_core import WasteRAG

HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "config.json").read_text())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_QUESTION_CHARS = 500
torch.set_num_threads(max(1, os.cpu_count() or 1))
print(f"Device: {DEVICE}, dtype: {DTYPE}, threads: {torch.get_num_threads()}")

embeddings = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"], model_kwargs={"device": DEVICE},
                                   encode_kwargs={"normalize_embeddings": True})


def load_store():
    """Open the Chroma index saved by Notebook 2; rebuild it from the same chunks if it cannot be opened."""
    try:
        store = Chroma(collection_name=CONFIG["collection"], embedding_function=embeddings,
                       persist_directory=str(HERE / "chroma_db"))
        count = store._collection.count()
        if count > 0:
            print(f"Loaded saved Chroma index: {count} vectors")
            return store
        print("Saved index is empty; rebuilding.")
    except Exception as error:  # e.g. a different chromadb version
        print("Could not open the saved index, rebuilding from chunks_final.jsonl:", error)
    docs, ids = [], []
    with open(HERE / "chunks_final.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.pop("text")
            docs.append(Document(page_content=text, metadata=row))
            ids.append(row["chunk_id"])
    store = Chroma(collection_name=CONFIG["collection"] + "_rebuilt", embedding_function=embeddings,
                   persist_directory="/tmp/chroma_rebuilt")
    store.add_documents(docs, ids=ids)
    print(f"Rebuilt Chroma index: {len(docs)} vectors (same chunks and embedding model as Notebook 2)")
    return store


vectorstore = load_store()
tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm"])
try:
    model = AutoModelForCausalLM.from_pretrained(CONFIG["llm"], dtype=DTYPE).to(DEVICE)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(CONFIG["llm"], torch_dtype=DTYPE).to(DEVICE)
model.eval()
rag = WasteRAG(vectorstore, embeddings, tokenizer, model, top_k=CONFIG["top_k"],
               max_new_tokens=CONFIG["max_new_tokens"])
print("RAG pipeline ready")


def log(sheet, **row):
    """Usage is printed to the Space logs (Settings > Logs); nothing is stored about the visitor."""
    stamp = datetime.datetime.now(ZoneInfo("Asia/Manila")).isoformat(timespec="seconds")
    print(json.dumps({"timestamp": stamp, "sheet": sheet, **row}, ensure_ascii=False))


# ------------------------------------------------------------------ interface (same as Notebook 4)
import html
import re
import gradio as gr

GRADIO_MAJOR = int(gr.__version__.split(".")[0])

KNOWLEDGE_BASE = [  # label shown to the user, pages in the PDF
    ("RA 9003 (Ecological Solid Waste Management Act)", 27),
    ("RA 6969 (Toxic Substances and Hazardous Wastes Act)", 6),
    ("DENR DAO 2013-22 (hazardous wastes)", 71),
    ("NSWMC Guidebook", 56),
    ("RA 9003 Information Material", 1),
]
EXAMPLES = [
    "How should I store kitchen waste while waiting for collection?",
    "Are used diapers and Styrofoam food packs biodegradable, recyclable or residual waste?",
    "I have an old car battery at home. How should I classify and dispose of it?",
]
PASSAGE_CHARS = 700

CSS = """
.gradio-container {max-width: 1280px !important; background: #F6F7F4 !important;}
body, .gradio-container {font-family: Inter, system-ui, sans-serif;}
#tn-header {display:flex; align-items:center; gap:14px; padding:16px 20px; background:#FFFFFF;
  border:1px solid #DDE3DE; border-radius:16px;}
#tn-header .mark {width:44px; height:44px; border-radius:12px; background:#1F6B4A; display:flex;
  align-items:center; justify-content:center; flex:0 0 auto;}
#tn-header h1 {margin:0; font-size:20px; font-weight:600; color:#17201B;}
#tn-header p {margin:2px 0 0; font-size:13px; color:#4B5A52;}
.tn-card {background:#FFFFFF; border:1px solid #DDE3DE; border-radius:14px; padding:16px; color:#17201B;}
.tn-card h2 {margin:0 0 10px; font-size:15px; font-weight:600; color:#17201B;}
.tn-kb {list-style:none; margin:0; padding:0; font-size:13px;}
.tn-kb li {display:flex; justify-content:space-between; gap:8px; padding:5px 0; border-bottom:1px solid #EEF1EE;}
.tn-kb li:last-child {border-bottom:0;}
.tn-kb span.pp {color:#4B5A52; white-space:nowrap;}
.tn-note {background:#FFF5E1; border:1px solid #F0D9A8; border-radius:14px; padding:14px; font-size:13px;
  line-height:1.5; color:#5C3800;}
.tn-example button, button.tn-example {text-align:left !important; justify-content:flex-start !important;
  font-weight:400 !important; min-height:44px;}
#tn-chat {background:#FFFFFF; border:1px solid #DDE3DE; border-radius:16px; padding:20px; min-height:460px;
  max-height:640px; overflow-y:auto; color:#17201B;}
.tn-empty {color:#4B5A52; font-size:14px; text-align:center; padding:120px 20px;}
.tn-user {display:flex; justify-content:flex-end; margin:0 0 14px;}
.tn-user p {margin:0; max-width:78%; background:#1F6B4A; color:#FFFFFF; padding:12px 16px;
  border-radius:16px 16px 4px 16px; font-size:15px; line-height:1.5;}
.tn-bot {margin:0 0 20px; display:flex; flex-direction:column; gap:10px;}
.tn-answer {font-size:15px; line-height:1.6; color:#17201B; margin:0;}
.tn-refusal {background:#FFF5E1; border:1px solid #F0D9A8; border-radius:12px; padding:12px 14px;}
.tn-refusal .tn-answer {color:#3D2600; font-weight:500;}
.tn-refusal small {display:block; margin-top:6px; color:#5C3800; font-size:13px; line-height:1.5;}
.tn-sources {display:flex; flex-direction:column; gap:6px;}
.tn-sources details {border:1px solid #DDE3DE; border-radius:10px; background:#FFFFFF;}
.tn-sources details[open] {background:#F6F7F4;}
.tn-sources summary {cursor:pointer; padding:10px 12px; font-size:13px; color:#17201B; list-style-position:inside;}
.tn-sources summary b {font-weight:600;}
.tn-sources summary span {color:#4B5A52;}
.tn-sources blockquote {margin:0; padding:0 14px 12px; font-size:13px; line-height:1.55; color:#26302A;
  }
.tn-meta {font-size:12px; color:#4B5A52;}
#tn-send {background:#1F6B4A !important; color:#FFFFFF !important; border:0 !important;}
#tn-send:hover {background:#164D36 !important;}
@media (max-width: 760px) { #tn-side {order: 2;} #tn-main {order: 1;} }
.tn-thinking {color:#4B5A52; font-size:14px; font-style:italic;}
"""

HEADER_HTML = """
<div id="tn-header">
  <div class="mark" aria-hidden="true">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="1.8"
      stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16"/><path d="M9 7V4h6v3"/>
      <path d="M6 7l1 13h10l1-13"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>
  </div>
  <div><h1>Waste Segregation Assistant</h1>
  <p>Answers from 5 Philippine sources, with the document and page for every answer &middot; TripleNeurons</p></div>
</div>
"""

KB_HTML = ('<div class="tn-card"><h2>Knowledge base</h2><ul class="tn-kb">'
           + "".join(f"<li><span>{html.escape(n)}</span><span class='pp'>{p} pp.</span></li>"
                     for n, p in KNOWLEDGE_BASE)
           + "</ul></div>")

NOTE_HTML = ('<div class="tn-note">Information only, not legal advice. Collection schedules and local rules '
             'differ per barangay and city: ask your LGU or city environment office.</div>')


def render_turn(turn):
    """One question and its answer as HTML (all text from the user or the model is escaped)."""
    q = html.escape(turn["question"])
    out = [f'<div class="tn-user"><p>{q}</p></div>']
    if turn.get("pending"):
        out.append('<div class="tn-bot"><p class="tn-thinking">Searching the documents and writing an answer...</p></div>')
        return "".join(out)
    answer = html.escape(turn["answer"]).replace("\n", "<br>")
    if turn["refusal"]:
        body = (f'<div class="tn-refusal"><p class="tn-answer">{answer}</p>'
                '<small>This assistant only covers RA 9003, RA 6969, DAO 2013-22 and two NSWMC materials. '
                'For other laws or local rules, check the official source or your LGU.</small></div>')
    else:
        body = f'<p class="tn-answer">{answer}</p>'
    chips = []
    for s in turn["sources"]:
        chips.append(
            f'<details><summary><b>{html.escape(str(s["document"]))}</b> '
            f'<span>&middot; page {s["page"]} &middot; similarity {s["score"]:.2f}</span></summary>'
            f'<blockquote>{html.escape(s["passage"])}</blockquote></details>')
    label = "Retrieved passages (not used: answer refused)" if turn["refusal"] else "Sources (click to read the passage)"
    out.append(f'<div class="tn-bot">{body}<div class="tn-meta">{label}</div>'
               f'<div class="tn-sources">{"".join(chips)}</div>'
               f'<div class="tn-meta">Answered in {turn["seconds"]:.2f} s</div></div>')
    return "".join(out)


def render_chat(history):
    if not history:
        return ('<div id="tn-chat"><div class="tn-empty">Ask a question about classifying or segregating waste, '
                'or pick an example on the left.</div></div>')
    return '<div id="tn-chat">' + "".join(render_turn(t) for t in history) + "</div>"


def answer_question(question, history):
    question = (question or "").strip()
    if not question:
        yield gr.update(), history, render_chat(history)
        return
    history = list(history) + [{"question": question, "pending": True}]
    yield "", history, render_chat(history)
    result = rag.ask(question)
    passages = []
    for d in result["docs"]:
        text = re.sub(r"\s+", " ", d.page_content).strip()      # OCR line breaks -> spaces
        passages.append(text[:PASSAGE_CHARS] + ("..." if len(text) > PASSAGE_CHARS else ""))
    sources = [{**s, "passage": p} for s, p in zip(result["sources"], passages)]
    history[-1] = {"question": question, "answer": result["answer"], "refusal": result["refusal"],
                   "sources": sources, "seconds": result["total_s"]}
    log("App_Usage", question=question, answer=result["answer"], refusal_flag=result["refusal"],
        sources="; ".join(f"{s['document']} p.{s['page']} ({s['score']})" for s in result["sources"]),
        retrieval_s=round(result["retrieval_s"], 3), generation_s=round(result["generation_s"], 3),
        total_s=round(result["total_s"], 3), new_tokens=result["new_tokens"])
    yield "", history, render_chat(history)


def new_chat():
    return [], render_chat([]), ""


FORCE_LIGHT = "() => { document.body.classList.remove('dark'); }"
blocks_kwargs = {"title": "Waste Segregation Assistant"}
style_kwargs = {"css": CSS, "theme": gr.themes.Base(primary_hue="green", font=[gr.themes.GoogleFont("Inter"), gr.themes.Font("system-ui"), gr.themes.Font("sans-serif")]),
                "js": FORCE_LIGHT}
if GRADIO_MAJOR < 6:            # Gradio 6 moved css/theme/js from Blocks() to launch()
    blocks_kwargs.update(style_kwargs)

with gr.Blocks(**blocks_kwargs) as demo:
    history_state = gr.State([])
    gr.HTML(HEADER_HTML)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=260, elem_id="tn-side"):
            gr.HTML(KB_HTML)
            gr.Markdown("**Try asking**")
            example_buttons = [gr.Button(e, size="sm", variant="secondary", elem_classes=["tn-example"])
                               for e in EXAMPLES]
            gr.HTML(NOTE_HTML)
        with gr.Column(scale=3, min_width=360, elem_id="tn-main"):
            chat_html = gr.HTML(render_chat([]))
            with gr.Row(equal_height=True):
                question_box = gr.Textbox(label="Ask about waste classification or segregation",
                                          placeholder="e.g. Where do busted fluorescent lamps go?",
                                          scale=5, autofocus=True)
                send_button = gr.Button("Send", variant="primary", scale=1, min_width=100, elem_id="tn-send")
            with gr.Row():
                new_button = gr.Button("New chat", variant="secondary", size="sm")
            gr.Markdown("Every answer lists the document and page it came from. Open a source to check it before acting.")

    outputs = [question_box, history_state, chat_html]
    SCROLL = "() => { const c = document.getElementById('tn-chat'); if (c) c.scrollTop = c.scrollHeight; }"
    send_button.click(answer_question, [question_box, history_state], outputs).then(None, None, None, js=SCROLL)
    question_box.submit(answer_question, [question_box, history_state], outputs).then(None, None, None, js=SCROLL)
    def example_handler(text):
        def handler(history):
            yield from answer_question(text, history)
        return handler

    for button, text in zip(example_buttons, EXAMPLES):
        button.click(example_handler(text), [history_state], outputs).then(None, None, None, js=SCROLL)
    new_button.click(new_chat, None, [history_state, chat_html, question_box])

launch_kwargs = {"share": True, "debug": False}
if GRADIO_MAJOR >= 6:
    launch_kwargs.update(style_kwargs)


# ------------------------------------------------------------------ API used by the Vercel page
def ask_api(question):
    """Answer one question and return the answer, refusal flag and sources as JSON."""
    question = (question or "").strip()
    if not question:
        return {"error": "Please type a question."}
    if len(question) > MAX_QUESTION_CHARS:
        return {"error": f"Please keep the question under {MAX_QUESTION_CHARS} characters."}
    result = rag.ask(question)
    sources = []
    for s, d in zip(result["sources"], result["docs"]):
        text = re.sub(r"\s+", " ", d.page_content).strip()
        sources.append({"document": s["document"], "page": s["page"], "score": s["score"],
                        "passage": text[:PASSAGE_CHARS] + ("..." if len(text) > PASSAGE_CHARS else "")})
    log("API_Usage", question=question, refusal_flag=result["refusal"], total_s=round(result["total_s"], 2),
        sources="; ".join(f"{s['document']} p.{s['page']}" for s in result["sources"]))
    return {"answer": result["answer"], "refusal": result["refusal"], "sources": sources,
            "seconds": round(result["total_s"], 2), "device": DEVICE}


with demo:
    api_question = gr.Textbox(visible=False)
    api_result = gr.JSON(visible=False)
    api_button = gr.Button(visible=False)
    api_button.click(ask_api, api_question, api_result, api_name="ask")

demo.queue(default_concurrency_limit=1, max_size=20)   # one answer at a time on the free CPU

if __name__ == "__main__":
    launch_kwargs.pop("share", None)
    launch_kwargs.pop("debug", None)
    demo.launch(**launch_kwargs)
