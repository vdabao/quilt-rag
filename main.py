import hashlib
import os
import re
import sys

from pathlib import Path
from dotenv import load_dotenv
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QTextBrowser, QVBoxLayout, QWidget,
)

DB_PATH = "./chroma_db"
COLLECTION = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "gemini-3.5-flash-lite")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

load_dotenv()

# Loading & Chunking
def load_document(path: str) -> list[tuple[str, str]]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        return [(f"p.{i + 1}", page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if ext == ".docx":
        from docx import Document
        doc = Document(path)
        return [("doc", "\n\n".join(p.text for p in doc.paragraphs))]
    if ext in (".txt", ".md"):
        return [("text", Path(path).read_text(encoding="utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Break up any paragraph that is longer than one chunk.
    pieces = []
    for p in paragraphs:
        while len(p) > size:
            pieces.append(p[:size])
            p = p[size - overlap:]
        pieces.append(p)

    chunks, current = [], ""
    for p in pieces:
        if len(current) + len(p) + 2 <= size:
            current = f"{current}\n\n{p}" if current else p
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{p}" if tail else p
    if current:
        chunks.append(current)
    return chunks

# RAG Engine (Embeddings, Vector DB, & LLM)
class RagEngine:
    def __init__(self):
        self._embedder = None
        self._collection = None
        self._llm = None
        self._client = None

    # Lazy-load heavy objects so the window opens instantly.
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBED_MODEL)
        return self._embedder

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=DB_PATH)
            self._collection = self._client.get_or_create_collection(
                COLLECTION, metadata={"hnsw:space": "cosine"})
        return self._collection

    @property
    def llm(self):
        if self._llm is None:
            from google import genai
            self._llm = genai.Client()  # reads GEMINI_API_KEY
        return self._llm

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def ingest(self, path: str) -> int:
        """Chunk, embed and store a file. Returns number of chunks stored."""
        name = Path(path).name
        file_id = hashlib.md5(Path(path).read_bytes()).hexdigest()[:10]

        ids, docs, metas = [], [], []
        for label, text in load_document(path):
            for chunk in chunk_text(text):
                ids.append(f"{file_id}-{len(ids)}")
                docs.append(chunk)
                metas.append({"source": name, "location": label})
        if not docs:
            raise ValueError("No extractable text found (scanned PDF?).")

        # Re-uploading the same file won't create duplicates
        self.collection.upsert(
            ids=ids, documents=docs, metadatas=metas, embeddings=self.embed(docs))
        return len(docs)

    def clear(self) -> None:
        _ = self.collection
        self._client.delete_collection(COLLECTION)
        self._collection = None

    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        if self.collection.count() == 0:
            return []
        res = self.collection.query(
            query_embeddings=self.embed([question]), n_results=k)
        return [
            {"text": d, "source": m["source"], "location": m["location"], "distance": dist}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def answer(self, question: str) -> tuple[str, list[dict]]:
        hits = self.retrieve(question)
        if not hits:
            return "Please upload a document first.<br>", []

        context = "\n\n".join(
            f"[{i + 1}] ({h['source']}, {h['location']})\n{h['text']}" for i, h in enumerate(hits)
        )
        system = (
            "You answer questions using ONLY the numbered context passages provided."
            "Paraphrase the passages in your own words, and do not copy them verbatim."
            "Cite passages like [1] or [2] and avoid putting them alone on a line."
            "If the answer isn't in the context, say you couldn't find it in the document."
            "Answer in the same language as the question."
            "Output a plain HTML fragment only. Use line breaks for readability."
            "Always avoid filler words and unnecessary line breaks."
            "Always be concise and to the point. Avoid repeating the question in your answer."
        )
        from google.genai import types
        resp = self.llm.models.generate_content(
            model=LLM_MODEL,
            contents=f"Context:\n{context}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return resp.text, hits


# Background Workers (GUI Responsiveness)
class IngestWorker(QThread):
    done = pyqtSignal(str, int)
    failed = pyqtSignal(str)

    def __init__(self, engine, path):
        super().__init__()
        self.engine, self.path = engine, path

    def run(self):
        try:
            n = self.engine.ingest(self.path)
            self.done.emit(Path(self.path).name, n)
        except Exception as e:
            self.failed.emit(str(e))

class ClearWorker(QThread):
    done = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def run(self):
        try:
            self.engine.clear()
            self.done.emit()
        except Exception as e:
            self.failed.emit(str(e))

class AskWorker(QThread):
    done = pyqtSignal(str, list)
    failed = pyqtSignal(str)

    def __init__(self, engine, question):
        super().__init__()
        self.engine, self.question = engine, question

    def run(self):
        try:
            answer, hits = self.engine.answer(self.question)
            self.done.emit(answer, hits)
        except Exception as e:
            self.failed.emit(str(e))


# Graphical User Interface
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuiltRAG")
        self.setWindowIcon(QIcon("assets/images/app_icon.ico"))
        self.resize(780, 620)
        self.engine = RagEngine()
        self.worker = None

        self.upload_btn = QPushButton("Upload Document")
        self.upload_btn.clicked.connect(self.pick_file)
        self.status = QLabel("Supported: PDF, DOCX, TXT, MD")

        self.clear_btn = QPushButton("Clear Memory")
        self.clear_btn.clicked.connect(self.clear_memory)

        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(False)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask a question about your documents…")
        self.input.returnPressed.connect(self.ask)
        self.send_btn = QPushButton("Ask")
        self.send_btn.clicked.connect(self.ask)

        top = QHBoxLayout()
        top.addWidget(self.upload_btn)
        top.addWidget(self.status, 1)
        top.addWidget(self.clear_btn, 0, Qt.AlignmentFlag.AlignRight)
        bottom = QHBoxLayout()
        bottom.addWidget(self.input, 1)
        bottom.addWidget(self.send_btn)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.chat, 1)
        layout.addLayout(bottom)
        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

    def set_busy(self, busy: bool, msg: str = ""):
        for w in (self.upload_btn, self.clear_btn, self.send_btn, self.input):
            w.setEnabled(not busy)
        if msg:
            self.status.setText(msg)

    # Upload
    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a document", "",
            "Documents (*.pdf *.docx *.txt *.md)")
        if not path:
            return
        self.set_busy(True, f"Indexing {Path(path).name}… (first run downloads the embedding model)")
        self.worker = IngestWorker(self.engine, path)
        self.worker.done.connect(self.on_ingested)
        self.worker.failed.connect(self.on_error)
        self.worker.start()

    # Clear Memory
    def clear_memory(self):
        confirm = QMessageBox.question(
            self, "Delete memory",
            "Delete all indexed documents? You'll need to upload them again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self.set_busy(True, "Deleting memory…")
        self.worker = ClearWorker(self.engine)
        self.worker.done.connect(self.on_cleared)
        self.worker.failed.connect(self.on_error)
        self.worker.start()

    def on_ingested(self, name, n):
        self.set_busy(False, f"Indexed {name} ({n} chunks)")
        self.chat.append(f"<div><i>Indexed <b>{name}</b> into {n} chunks.</i><br></div>")

    def on_cleared(self):
        self.set_busy(False, "Memory cleared.")
        self.chat.append("<div><i>All indexed documents were deleted.</i><br></div>")

    # Ask
    def ask(self):
        question = self.input.text().strip()
        if not question:
            return
        self.input.clear()
        self.chat.append(f"<div><span style='color:lightgreen'>> {question}</span><br></div>")
        self.set_busy(True, "Thinking…")
        self.worker = AskWorker(self.engine, question)
        self.worker.done.connect(self.on_answer)
        self.worker.failed.connect(self.on_error)
        self.worker.start()

    def on_answer(self, answer, hits):
        html = f"<div>{answer}</div>"
        if hits:
            srcs = "<br>".join(
                f"[{i + 1}] {h['source']} ({h['location']}) – "
                f"{h['text'][:90].replace(chr(10), ' ')}…"
                for i, h in enumerate(hits))
            html += f"<div><small style='color:gray'>Sources:<br>{srcs}</small><br></div>"
        self.chat.append(html)
        self.set_busy(False, "Ready")

    def on_error(self, msg):
        self.chat.append(f"<div><span style='color:red'>Error: {msg}</span><br></div>")
        self.set_busy(False, "Error")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())