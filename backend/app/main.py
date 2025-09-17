import os
import time
import logging
import datetime as dt
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from pydantic import BaseModel, field_validator
from openai import OpenAI, APIStatusError
from flask import make_response
import easyocr
from PIL import Image, ImageOps

reader = easyocr.Reader(['en'])

load_dotenv()


# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CORS util
def add_cors(resp, origin):
    resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp

def create_app():
    app = Flask(__name__)

    # Config
    db_url = os.getenv("DATABASE_URL", "sqlite:///homework.db")
    allowed_origin = os.getenv("ALLOWED_ORIGIN", "*")
    default_model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini")

    # OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # DB setup
    engine = create_engine(db_url, echo=False, future=True)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    Base = declarative_base()

    class User(Base):
        __tablename__ = "users"
        id = Column(Integer, primary_key=True)
        external_id = Column(String, unique=True, index=True)
        created_at = Column(DateTime, default=dt.datetime.utcnow)
        chats = relationship("Chat", back_populates="user")

    class Chat(Base):
        __tablename__ = "chats"
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, ForeignKey("users.id"))
        level = Column(String)
        subject = Column(String)
        title = Column(String, default="New Chat")
        created_at = Column(DateTime, default=dt.datetime.utcnow)
        user = relationship("User", back_populates="chats")
        messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")

    class Message(Base):
        __tablename__ = "messages"
        id = Column(Integer, primary_key=True)
        chat_id = Column(Integer, ForeignKey("chats.id"))
        role = Column(String)
        content = Column(Text)
        created_at = Column(DateTime, default=dt.datetime.utcnow)
        chat = relationship("Chat", back_populates="messages")

    Base.metadata.create_all(engine)

    PRESETS = {
      "Kindergarten": {
        "Math":    {"style":"Use very simple words, 1-2 step examples, emojis ok.","temperature":0.4},
        "Science": {"style":"Concrete examples from daily life, avoid jargon.","temperature":0.5},
        "English": {"style":"Short sentences, phonics-friendly examples.","temperature":0.6},
      },
      "Lower Secondary": {
        "Math":    {"style":"Show steps 1-2-3, no fluff, final answer boxed.","temperature":0.2},
        "Science": {"style":"Explain cause→effect, add key terms at the end.","temperature":0.4},
        "English": {"style":"Explain, then give a short example paragraph.","temperature":0.5},
      },
      "Upper Secondary": {
        "Math":    {"style":"Be rigorous, list assumptions, show algebra cleanly.","temperature":0.1},
        "Science": {"style":"Define terms, add 1 real-world application.","temperature":0.3},
        "English": {"style":"Thesis→points→mini-conclusion, concise.","temperature":0.4},
      },
    }

    def build_system_prompt(level, subject, practice_mode=False):
        p = PRESETS[level][subject]
        core_rules = [
            "You are a helpful homework tutor.",
            "Tailor difficulty to the specified level.",
            "If math: show steps as bullet points; keep symbols clean.",
            "If code/math, avoid flowery prose; be concise.",
            "If unsure, state what’s missing and ask one focused follow-up.",
            "Avoid hallucinations; if not in scope, say so briefly.",
            "Do not write complete essays or long-form assignments; instead, provide a structured outline or guidance on how to approach the task."
        ]
        if practice_mode:
            core_rules.append("Practice mode is ON. Guide the user with hints and steps, but DO NOT provide the final answer to the question.")
        return f"{' '.join(core_rules)} Style: {p['style']}"

    def llm_call(model: str, system_prompt: str, user_text: str, temperature: float):
        temp_used = temperature
        omitted_temp = False
        try:
            logger.info(f"LLM call: model={model}, temp={temperature}")
            resp = client.responses.create(
                model=model,
                input=[{"role":"system","content":system_prompt},
                       {"role":"user","content":user_text}],
                temperature=temperature
            )
            return resp.output_text
        except APIStatusError as e:
            logger.warning(f"APIStatusError calling LLM: {e}")
            if "temperature" in str(e).lower():
                logger.info("Retrying LLM call without temperature.")
                omitted_temp = True
                temp_used = None
                resp = client.responses.create(
                    model=model,
                    input=[{"role":"system","content":system_prompt},
                           {"role":"user","content":user_text}]
                )
                return resp.output_text
            raise
        finally:
            logger.info(f"LLM call complete. Model={model}, Temp={temp_used}, TempOmitted={omitted_temp}")

    class AskPayload(BaseModel):
        user_external_id: str
        level: str
        subject: str
        question: str
        chat_id: int | None = None
        model: str | None = None
        practice_mode: bool = False

        @field_validator("level")
        @classmethod
        def v_level(cls, v):
            ok = {"Kindergarten","Lower Secondary","Upper Secondary"}
            if v not in ok: raise ValueError(f"level must be one of {ok}")
            return v

        @field_validator("subject")
        @classmethod
        def v_subject(cls, v):
            ok = {"Math","Science","English"}
            if v not in ok: raise ValueError(f"subject must be one of {ok}")
            return v

    @app.after_request
    def cors_headers(resp):
        return add_cors(resp, allowed_origin)

    @app.route("/", methods=["GET"])
    def health():
        return add_cors(make_response(jsonify({"ok": True, "service": "homework-helper-backend"}), 200), allowed_origin)

    @app.route("/api/upload", methods=["POST", "OPTIONS"])
    def upload_file():
        if request.method == "OPTIONS":
            return add_cors(make_response("", 204), allowed_origin)

        if 'file' not in request.files:
            return add_cors(make_response(jsonify({"ok": False, "error": "No file uploaded"}), 400), allowed_origin)

        file = request.files['file']
        if file.filename == '':
            return add_cors(make_response(jsonify({"ok": False, "error": "Empty filename"}), 400), allowed_origin)

        try:
            # Open image
            image = Image.open(file.stream)
            
            # Preprocess
            image = image.convert('L')  # grayscale
            image = ImageOps.autocontrast(image)
            if min(image.size) < 800:
                image = image.resize((image.width*2, image.height*2), Image.Resampling.LANCZOS)
            
            # OCR with EasyOCR
            import numpy as np
            img_np = np.array(image)
            result = reader.readtext(img_np, detail=0)
            raw_text = "\n".join(result)

            logger.info(f"Raw OCR result: {raw_text!r}")

            # Optional AI cleanup
            system_prompt = (
                "You are a text assistant. "
                "The following text was extracted from an image and may have OCR errors. "
                "Please correct spelling, punctuation, and formatting, "
                "but do not change the meaning or remove any content."
            )
            cleaned_text = client.responses.create(
                model=default_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_text}
                ]
            ).output_text

            logger.info(f"Cleaned OCR result: {cleaned_text!r}")

            return add_cors(make_response(jsonify({"ok": True, "text": cleaned_text}), 200), allowed_origin)

        except Exception as e:
            logger.error(f"OCR + AI cleanup error: {e}", exc_info=True)
            return add_cors(make_response(jsonify({"ok": False, "error": str(e)}), 500), allowed_origin)
            
    @app.route("/ask", methods=["POST", "OPTIONS"])
    def ask():
        start_time = time.time()
        if request.method == "OPTIONS":
            return add_cors(make_response("", 204), allowed_origin)
        
        try:
            data = AskPayload(**request.get_json())
            logger.info(f"ASK request for user '{data.user_external_id}' in chat '{data.chat_id}'")
            db = SessionLocal()

            user = db.query(User).filter_by(external_id=data.user_external_id).first()
            if not user:
                user = User(external_id=data.user_external_id)
                db.add(user); db.commit(); db.refresh(user)

            if data.chat_id:
                chat = db.query(Chat).filter_by(id=data.chat_id, user_id=user.id).first()
                if not chat: 
                    return add_cors(make_response(jsonify({"ok": False, "error": "chat not found"}), 404), allowed_origin)
            else:
                chat = Chat(user_id=user.id, level=data.level, subject=data.subject,
                            title=f"{data.subject} ({data.level})")
                db.add(chat); db.commit(); db.refresh(chat)

            sys_prompt = build_system_prompt(chat.level, chat.subject, data.practice_mode)
            db.add(Message(chat_id=chat.id, role="system", content=sys_prompt))
            db.add(Message(chat_id=chat.id, role="user", content=data.question))
            db.commit()

            preset = PRESETS[chat.level][chat.subject]
            model = data.model or default_model
            answer = llm_call(model, sys_prompt, data.question, temperature=preset["temperature"])
            
            answer += "\n\n---\n*AI can be wrong—verify with your notes.*"

            msg = Message(chat_id=chat.id, role="assistant", content=answer)
            db.add(msg); db.commit(); db.refresh(msg)

            return add_cors(make_response(jsonify({"ok": True, "chat_id": chat.id, "answer": answer}), 200), allowed_origin)
        except Exception as e:
            logger.error(f"Error in /ask endpoint: {e}", exc_info=True)
            return add_cors(make_response(jsonify({"ok": False, "error": "Internal server error"}), 500), allowed_origin)
        finally:
            latency = time.time() - start_time
            logger.info(f"ASK request finished. Latency: {latency:.2f}s")

    @app.route("/chats", methods=["GET", "OPTIONS"])
    def list_chats():
        start_time = time.time()
        if request.method == "OPTIONS":
            return add_cors(make_response("", 204), allowed_origin)
        user_external_id = request.args.get("user_external_id")
        logger.info(f"CHATS request for user '{user_external_id}'")
        if not user_external_id:
            return add_cors(make_response(jsonify({"ok": False, "error": "user_external_id required"}), 400), allowed_origin)
        db = SessionLocal()
        user = db.query(User).filter_by(external_id=user_external_id).first()
        if not user: 
            return add_cors(make_response(jsonify({"ok": True, "chats": []}), 200), allowed_origin)
        rows = [{"id": c.id, "title": c.title, "level": c.level, "subject": c.subject,
                 "created_at": c.created_at.isoformat()} for c in user.chats]
        latency = time.time() - start_time
        logger.info(f"CHATS request finished. Found {len(rows)} chats. Latency: {latency:.2f}s")
        return add_cors(make_response(jsonify({"ok": True, "chats": rows}), 200), allowed_origin)

    @app.route("/history", methods=["GET", "OPTIONS"])
    def history():
        start_time = time.time()
        if request.method == "OPTIONS":
            return add_cors(make_response("", 204), allowed_origin)
        chat_id = request.args.get("chat_id")
        logger.info(f"HISTORY request for chat_id '{chat_id}'")
        if not chat_id:
            return add_cors(make_response(jsonify({"ok": False, "error": "chat_id required"}), 400), allowed_origin)
        db = SessionLocal()
        chat = db.query(Chat).filter_by(id=int(chat_id)).first()
        if not chat: 
            return add_cors(make_response(jsonify({"ok": False, "error": "chat not found"}), 404), allowed_origin)
        
        msgs = [{"role": m.role, "content": m.content, "time": m.created_at.isoformat()}
                for m in chat.messages]
        latency = time.time() - start_time
        logger.info(f"HISTORY request finished. Found {len(msgs)} messages. Latency: {latency:.2f}s")
        return add_cors(make_response(jsonify({"ok": True, "chat": {"id": chat.id, "title": chat.title,
                        "level": chat.level, "subject": chat.subject}, "messages": msgs}), 200), allowed_origin)

    return app
