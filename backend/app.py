# -*- coding: utf-8 -*-
"""
Backend for the Navia.life "AI accountability companion" demo.

Three endpoints:
  POST /api/companion-note  — writes a short text check-in note from the
                               subscriber's worksheet progress (Gemini text).
  POST /api/token            — mints a short-lived Gemini Live token so the
                               browser can start a voice conversation
                               directly, seeded with the same progress data.
  POST /api/refine-note      — takes a short spoken recording for one
                               worksheet field and returns a distilled,
                               cleaned-up note in the person's own words.

The Gemini API key never leaves this server.
"""

import datetime
import functools
import json
import os
import re
import sqlite3
import urllib.error
import urllib.request

from dotenv import load_dotenv
from flask import Flask, g, jsonify, request, session
from flask_cors import CORS
from google import genai
from google.genai import types
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
CORS(app, supports_credentials=True)
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_HTTPONLY=True,
)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "navia.db")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kv_store (
            user_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (user_id, key),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


init_db()


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            raise ApiError("Please log in first.", 401)
        return view(*args, **kwargs)
    return wrapped


USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")


@app.post("/api/auth/register")
def register():
    body = request.get_json(force=True) or {}
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""

    if not USERNAME_RE.match(username):
        raise ApiError("Username must be 3-32 characters: letters, numbers, . _ -", 400)
    if len(password) < 8:
        raise ApiError("Password must be at least 8 characters.", 400)

    db = get_db()
    existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        raise ApiError("That username is already taken.", 409)

    now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    cur = db.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, generate_password_hash(password), now),
    )
    db.commit()
    session["user_id"] = cur.lastrowid
    session["username"] = username
    return jsonify({"username": username})


@app.post("/api/auth/login")
def login():
    body = request.get_json(force=True) or {}
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""

    db = get_db()
    row = db.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,)).fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        raise ApiError("Incorrect username or password.", 401)

    session["user_id"] = row["id"]
    session["username"] = username
    return jsonify({"username": username})


@app.post("/api/auth/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.get("/api/auth/me")
def me():
    if not session.get("user_id"):
        raise ApiError("Not logged in.", 401)
    return jsonify({"username": session["username"]})


@app.get("/api/kv/<path:key>")
@login_required
def kv_get(key):
    db = get_db()
    row = db.execute(
        "SELECT value FROM kv_store WHERE user_id = ? AND key = ?",
        (session["user_id"], key),
    ).fetchone()
    if not row:
        raise ApiError("Not found.", 404)
    return jsonify({"value": row["value"]})


@app.post("/api/kv/<path:key>")
@login_required
def kv_set(key):
    body = request.get_json(force=True) or {}
    value = body.get("value", "")
    db = get_db()
    db.execute(
        "INSERT INTO kv_store (user_id, key, value) VALUES (?, ?, ?) "
        "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value",
        (session["user_id"], key, value),
    )
    db.commit()
    return jsonify({"ok": True})

TEXT_MODEL = "gemini-flash-lite-latest"
LIVE_MODEL = "gemini-3.1-flash-live-preview"
JOURNEY_MODEL = "gemini-3.6-flash"  # stronger model — used for search + journey design only
USE_SEARCH_GROUNDING = True  # requires real (non-free-tier) quota — confirmed working 2026-09-09


class ApiError(Exception):
    def __init__(self, message, status):
        self.message = message
        self.status = status


@app.errorhandler(ApiError)
def handle_api_error(e):
    return jsonify({"error": e.message}), e.status


def generate_text(**kwargs):
    """Wraps client.models.generate_content with a clean, user-facing error
    on failure — most commonly Gemini's free-tier daily quota (429)."""
    try:
        resp = client.models.generate_content(**kwargs)
        return resp.text or ""
    except Exception as e:
        print(f"[generate_text] call failed: {e}")
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            raise ApiError(quota_error_message(msg), 429)
        raise ApiError("The AI request failed. Please try again.", 502)


def quota_error_message(raw_error_text):
    """Turn a 429 into an accurate, actionable message — prepaid credits
    being depleted is a different problem (and fix) than the free-tier
    daily request cap, so tell them apart instead of always saying the
    same generic thing."""
    if "prepayment credits" in raw_error_text or "prepay" in raw_error_text.lower():
        return (
            "Your Gemini API prepaid credits are depleted. Add funds at "
            "https://ai.studio/projects (Billing) to continue."
        )
    return (
        "Daily AI quota reached for this model. Try again tomorrow, "
        "or enable billing on the Gemini API key for higher limits."
    )

NOTE_SYSTEM_PROMPT = """\
You are a warm, calm accountability companion inside a self-paced growth \
program called Navia. You are NOT a therapist and must never diagnose, give \
clinical advice, or use clinical labels. Your only job is to write one short, \
personal check-in note (3-5 sentences, second person, plain language) based on \
the specific progress data you're given. Reference at least one concrete, \
specific detail from their notes so it feels genuinely read, not generic. Keep \
the tone quiet and grounded — no exclamation marks, no hype, no gamified praise \
("Amazing job!!"), no toxic positivity. If the data suggests things are \
genuinely hard (very low ratings, notes describing distress), acknowledge that \
plainly and gently suggest they also lean on a person they trust or a \
professional — without alarm. End with one gentle, non-pressuring pointer \
toward whatever seems like a natural next step. Respond in the same language \
as the user's notes if they are not in English; otherwise respond in English. \
Output only the note itself, no preamble, no headers, no quotation marks \
around it."""

VOICE_SYSTEM_PROMPT = """\
You are a warm, calm voice accountability companion inside a self-paced \
growth program called Navia. The person is working through a journey \
called "{journey_title}" — use that context, don't assume it's about \
burnout or any other specific topic unless the snapshot below says so. You \
are NOT a therapist — never diagnose, never use clinical language.

The moment this conversation starts, YOU speak first — greet them briefly \
and ask your first question right away. Never wait silently. Speak in \
short, natural sentences (this is a voice conversation, not text). Start by \
acknowledging where the person actually is in their progress, using the \
snapshot below, then ask one open question about how they're doing right \
now. Let them lead. If the data or the conversation suggests real \
difficulty, gently point them toward a trusted person or professional — \
without alarm. Respond in whatever language the person speaks to you in.

Progress snapshot (JSON):
{snapshot}
"""


@app.post("/api/companion-note")
def companion_note():
    snapshot = request.get_json(force=True) or {}
    text = generate_text(
        model=TEXT_MODEL,
        contents=f"Progress snapshot (JSON):\n{snapshot}",
        config=types.GenerateContentConfig(system_instruction=NOTE_SYSTEM_PROMPT),
    )
    return jsonify({"text": text})


REFINE_PROMPT = """\
The person is filling out a worksheet field titled "{context}" as part of a \
calm, self-paced growth program. They just spoke freely, out loud, \
about how they feel or what happened. Listen to the recording and write what \
they'd want written in that field: a clear, concise note in first person, in \
their own words, distilled to the essence — not a padded summary, not a \
transcript, not clinical. Keep it to 1-3 short sentences unless they clearly \
said more that matters. Respond in the same language they spoke. Output only \
the note itself, no preamble, no quotation marks."""


@app.post("/api/refine-note")
def refine_note():
    audio_file = request.files["audio"]
    context = request.form.get("context", "")
    text = generate_text(
        model=TEXT_MODEL,
        contents=[
            types.Part.from_bytes(
                data=audio_file.read(),
                mime_type=audio_file.mimetype or "audio/webm",
            ),
            REFINE_PROMPT.format(context=context),
        ],
    )
    return jsonify({"text": text.strip()})


DRAFT_SYSTEM_PROMPT = """\
You are helping someone fill out one field of a self-paced growth-program \
worksheet. You are given everything else they have already entered in this \
same worksheet (as JSON) and the specific question you need to draft an \
answer for. Synthesize a first-person answer using ONLY concrete details \
already present in their data — never invent facts, numbers, or events that \
aren't there. Keep it to 1-3 short, plain sentences. This is a DRAFT the \
person will review and edit themselves, so be direct and specific rather than \
hedged or vague. If the existing data is too sparse to answer meaningfully, \
say so briefly instead of guessing. Respond in the same language as their \
existing entries. Output only the draft text, no preamble, no quotation marks."""

LIST_DRAFT_SYSTEM_PROMPT = """\
You are helping someone fill out a short list field of a self-paced \
growth-program worksheet. You are given everything else they have already \
entered in this same worksheet (as JSON) and a list field to draft. Using \
ONLY concrete details already present in their data — never invent facts, \
numbers, or events that aren't there — write the requested number of items, \
each a short phrase (not a full sentence), one per line, no numbering, no \
bullets, no extra text. This is a DRAFT the person will review and edit \
themselves. If the existing data is too sparse to draft some items \
meaningfully, write fewer lines rather than inventing content. Respond in the \
same language as their existing entries."""


@app.post("/api/draft-note")
def draft_note():
    body = request.get_json(force=True) or {}
    list_count = body.get("listCount")
    week_data = body.get("weekData")
    field_label = body.get("fieldLabel")

    if list_count:
        prompt = (
            f"Everything already entered in this worksheet (JSON):\n{week_data}\n\n"
            f"Draft exactly {list_count} short, distinct items for: \"{field_label}\""
        )
        system_instruction = LIST_DRAFT_SYSTEM_PROMPT
    else:
        prompt = (
            f"Everything already entered in this worksheet (JSON):\n{week_data}\n\n"
            f"Draft the answer to this specific question: \"{field_label}\""
        )
        system_instruction = DRAFT_SYSTEM_PROMPT

    text = generate_text(
        model=TEXT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    return jsonify({"text": text.strip()})


LANG_NAMES = {"LT": "Lithuanian", "LV": "Latvian", "EE": "Estonian"}

RESOURCE_SYSTEM_PROMPT = """\
You are a warm, calm voice companion inside a self-paced growth program \
called Navia. The person just read a curated summary of this resource:

Title: {title} ({outlet})
Key points:
{points}
Takeaway: {takeaway}

The moment this conversation starts, YOU speak first — greet them briefly \
and open with a question or observation about the resource above. Never \
wait silently for them to speak first.

Start by discussing the resource with them: answer questions, elaborate on \
any point if asked, offer perspective. Never invent facts beyond what's in \
the summary above — if asked something it doesn't cover, say so plainly \
rather than guessing.

Once the discussion feels complete — they seem ready, or ask to move on — \
guide them through this week's task, conversationally, one thing at a time:

Task: {task}
Worksheet: {sheet_name} — {sheet_desc}

Fields you can fill, using the set_worksheet_field tool. Call it as soon as \
you have a confident value for a field — don't wait until the end of the \
conversation, and don't ask the person to repeat something just to fill a \
field; use what they already told you.
{fields_desc}

Already filled in (don't re-ask about these unless they bring it up):
{current_values}

For a 1-10 rating field, either ask for a number directly or infer one from \
how they describe it, and briefly confirm the number back to them. After \
filling a field, a short natural acknowledgement is enough — don't recite \
the value back mechanically every time.

Speak in short, natural sentences (this is a voice conversation, not text). \
Primarily respond in {lang_name}, but switch language naturally if they \
speak to you in a different one. Tone: guiding, not preaching — plain and \
warm, no hype, no exclamation marks."""


def describe_fields(fields):
    lines = []
    for f in fields:
        ftype = f.get("type")
        key = f.get("key")
        label = f.get("label")
        if ftype == "rows":
            rows = ", ".join(f"{idx}={rl}" for idx, rl in enumerate(f.get("rows", [])))
            cols = ", ".join(
                f"\"{c['key']}\" ({c.get('type', 'text')}, \"{c.get('label', '')}\")"
                for c in f.get("columns", [])
            )
            lines.append(
                f"- field=\"{key}\" (\"{label}\"): a table. rows: [{rows}]. "
                f"columns: [{cols}]. Address a cell with field=\"{key}\", "
                f"row=<row index>, col=<column key>."
            )
        elif ftype == "list":
            lines.append(
                f"- field=\"{key}\" (\"{label}\"): a list of {f.get('count')} "
                f"short items. Address an item with field=\"{key}\", "
                f"listidx=<0-based index>."
            )
        else:
            lines.append(f"- field=\"{key}\" (\"{label}\", {ftype}): a single field.")
    return "\n".join(lines)


INTAKE_SYSTEM_PROMPT = """\
You are a warm, calm intake companion inside a growth platform called \
Navia. The moment this conversation starts, YOU speak first — greet them \
briefly and ask your first question right away. Never wait silently for \
them to speak first.

Your first question must be open, not a script: ask what they're hoping to \
work on, and who it's for — themselves, or are they preparing to help, \
teach, coach, or lead something for other people (e.g. a consultant \
preparing to run a training for clients, a manager preparing to coach \
their team)? This distinction changes what kind of journey makes sense, so \
really listen to the answer rather than assuming it's always a personal \
self-improvement journey — it might instead be a preparation plan for \
something they're going to deliver to others.

From there, have a real, adaptive conversation — not a fixed checklist. \
Follow up naturally on what they actually said, in whatever order makes \
sense. By the end you need a genuine, specific understanding of:
1. The real problem or goal, in their own words — including who it's for \
   (themselves, or someone/something else they're preparing for).
2. Where they are now — current situation, experience, what they already \
   have in place, or how long this has been going on.
3. How much time they can realistically give to this per week.
4. How THEY (the person you're talking to) prefer to work through \
   material — reading, listening, or a mix.

The moment you have a genuine, specific answer to all four — even rough, \
you don't need perfect detail — call submit_needs_assessment IMMEDIATELY, \
in that same turn. Put anything else important in the "notes" field — \
especially if this journey is to prepare something for other people, say \
so explicitly there and describe who. Do not summarize back to them first, \
do not ask "shall I build it now?", do not keep chatting past that point. \
Estimate a journey length yourself (4-8 weeks) based on how deep the \
problem sounds — never ask the person to pick a number of weeks. Calling \
the tool is the correct way to end this conversation; say one short \
closing line as you do it (e.g. "building it now"), not before.

Speak in short, natural sentences (this is a voice conversation, not text). \
Respond in whatever language the person speaks to you in — note it as \
"LT", "LV", "EE", or "EN" when you call the tool. Tone: guiding, not \
preaching — plain and warm, no hype, no exclamation marks."""

# The submit_needs_assessment tool's schema is declared client-side (frontend
# passes it to ai.live.connect) — keep it in sync with the description above
# if either changes.


@app.post("/api/token")
def issue_token():
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "companion")

    if mode == "intake":
        system_instruction = INTAKE_SYSTEM_PROMPT
    elif mode == "resource":
        points = "\n".join(f"- {p}" for p in body.get("points", []))
        system_instruction = RESOURCE_SYSTEM_PROMPT.format(
            title=body.get("title", ""),
            outlet=body.get("outlet", ""),
            points=points,
            takeaway=body.get("takeaway", ""),
            task=body.get("task", ""),
            sheet_name=body.get("sheetName", ""),
            sheet_desc=body.get("sheetDesc", ""),
            fields_desc=describe_fields(body.get("fields", [])),
            current_values=json.dumps(body.get("weekData", {}), ensure_ascii=False),
            lang_name=LANG_NAMES.get(body.get("lang"), "English"),
        )
    else:
        snapshot = {k: v for k, v in body.items() if k != "mode"}
        system_instruction = VOICE_SYSTEM_PROMPT.format(
            snapshot=snapshot,
            journey_title=body.get("journeyTitle") or "their growth journey",
        )

    now = datetime.datetime.now(tz=datetime.timezone.utc)
    token = client.auth_tokens.create(
        config={
            "uses": 1,
            "expire_time": now + datetime.timedelta(minutes=30),
            "new_session_expire_time": now + datetime.timedelta(minutes=1),
        }
    )
    return jsonify({
        "token": token.name,
        "model": LIVE_MODEL,
        "systemInstruction": system_instruction,
    })


RESEARCH_PROMPT = """\
Design a self-paced growth journey for someone with this need:

Problem: {problem}
Current level / context: {current_level}
Time available: about {weekly_minutes} minutes per week
Format preference: {format_pref}
Additional context from the intake conversation: {notes}

If the additional context says this journey is to prepare something for \
OTHER people (e.g. a consultant preparing to run a training, a manager \
preparing to coach their team) rather than the person's own personal \
growth, design the weeks accordingly: each week should build toward the \
person being ready to DELIVER that training/session — e.g. researching the \
topic, structuring the content, preparing materials or talking points, \
anticipating questions, rehearsing — not a personal self-improvement \
program. The worksheet fields for such a journey should produce actual \
prep artifacts (outlines, talking points, a rehearsal checklist), not \
mood/energy tracking.

Plan {weeks} weekly steps that build on each other logically (e.g. awareness \
→ foundational skills → application → consolidation — adapt this arc to fit \
the actual problem). EVERY week must have a real resource — do not skip \
this for any week. Search the web for the best fit, choosing whichever \
format actually serves that sub-topic best:
- An article, research summary, or well-known essay (freely readable, not \
  paywalled)
- A podcast episode (a specific episode, not just the show's homepage)
- A video or talk (e.g. a specific YouTube video or conference talk)
- A well-known book — but never point to the whole book. Name ONE specific \
  chapter or section that fits this week, and write the summary from what \
  that chapter argues, so the person gets real value even if they never \
  buy the book.

For each week, report:
- A short week title (2-4 words)
- A one-sentence blurb of what the week covers
- The resource type: article, podcast, video, or book
- The resource's exact title (for a book: "Book Title — Chapter N: Chapter \
  Name"), its publisher/outlet/author/show name, and its exact URL (for a \
  book, the URL should point to a real page about it — publisher page, \
  author's site, or a retailer listing — not a summary/pirated-copy site)
- 2-3 sentences summarizing what the resource actually says (for a book \
  chapter, summarize that chapter specifically, not the whole book)
- One practical task the person could do that week, grounded in the resource

Be precise about the URL — it must be the real, exact address of a page \
that exists. Only if you are completely unable to find any real resource \
in any format for a sub-topic — which should be rare — say so explicitly \
instead of inventing one."""

STRUCTURE_PROMPT = """\
Convert the research below into strict JSON matching this exact shape — \
output ONLY the JSON, no markdown fences, no commentary:

{{
  "journeyTitle": "string, 3-6 words",
  "weeks": [
    {{
      "title": "string, 2-4 words",
      "blurb": "one sentence",
      "sourceType": "article | podcast | video | book",
      "sourceTitle": "string, or null if the week has no resource",
      "sourceOutlet": "string (publisher, show name, channel, or author)",
      "sourceUrl": "string, or null",
      "sourceBlurb": "1-2 sentences describing the resource",
      "summaryPoints": {{
        "LT": ["3-5 short bullet points in Lithuanian, each a concrete insight from the resource"],
        "LV": ["the same points in Latvian"],
        "EE": ["the same points in Estonian"]
      }},
      "summaryTakeaway": {{ "LT": "one sentence", "LV": "one sentence", "EE": "one sentence" }},
      "task": "1-2 sentences, second person, a concrete practical task for this week",
      "sheetName": "short worksheet name, 2-4 words",
      "sheetDesc": "one sentence describing the worksheet",
      "fields": [
        {{ "type": "textarea", "key": "shortkey", "label": "question or prompt", "aiDraft": false }}
      ]
    }}
  ]
}}

Field design rules for "fields" (2-4 fields per week):
- "type" is one of: "text" (short one-line answer), "textarea" (longer \
written reflection), "list" (short enumerated items — add "count": 2-4 and \
"placeholder"), "rows" (a small table — add "rowLabels": ["...", ...] and \
"columns": [{{ "key": "...", "type": "text"|"range"|"textarea", "label": "..." }}]).
- Every field needs a unique "key" (short, lowercase, no spaces) within its week.
- Set "aiDraft": true only on a "textarea" or "list" field whose answer \
could be synthesized FROM the week's OTHER fields (e.g. a reflection \
question that summarizes a tracking table above it). Omit or set false \
otherwise.
- Keep "label", "task", "sheetName", "sheetDesc", "blurb", "title" in \
English — only "summaryPoints" and "summaryTakeaway" are LT/LV/EE.

Research:
{research}
"""


def verify_url(url, timeout=5):
    """Best-effort check that a URL resolves to a real page. Treats bot-block
    responses (403/429) as "exists" — many real sites reject scripted
    requests but the page is still genuinely there. Only 404/410 (or a
    total connection failure) count as "doesn't exist"."""
    if not url:
        return False
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    for method, extra in (("HEAD", {}), ("GET", {"Range": "bytes=0-512"})):
        try:
            req = urllib.request.Request(url, method=method, headers={**headers, **extra})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if 200 <= resp.status < 400:
                    return True
        except urllib.error.HTTPError as e:
            if e.code in (403, 405, 406, 429):
                return True  # blocked as a bot, but the page is real
            if e.code in (404, 410):
                continue  # try the next method before giving up
        except Exception:
            continue
    return False


def clean_text(value):
    """Strip stray U+FFFD replacement characters — an occasional artifact
    of scraped web text coming back through search grounding — from any
    string, recursively through lists/dicts."""
    if isinstance(value, str):
        return value.replace("�", "-")
    if isinstance(value, list):
        return [clean_text(v) for v in value]
    if isinstance(value, dict):
        return {k: clean_text(v) for k, v in value.items()}
    return value


def build_phases(raw):
    raw = clean_text(raw)
    weeks_out = []
    for w in raw.get("weeks", []):
        url = w.get("sourceUrl")
        if url and not verify_url(url):
            url = None
        summary = {}
        for lang in ("LT", "LV", "EE"):
            summary[lang] = {
                "points": (w.get("summaryPoints") or {}).get(lang, []),
                "takeaway": (w.get("summaryTakeaway") or {}).get(lang, ""),
            }
        weeks_out.append({
            "title": w.get("title") or "Untitled",
            "blurb": w.get("blurb") or "",
            "sourceType": w.get("sourceType") or "article",
            "sourceTitle": w.get("sourceTitle"),
            "sourceOutlet": w.get("sourceOutlet") or "",
            "sourceUrl": url,
            "sourceBlurb": w.get("sourceBlurb") or "",
            "summary": summary,
            "task": w.get("task") or "",
            "sheetName": w.get("sheetName") or "Worksheet",
            "sheetDesc": w.get("sheetDesc") or "",
            "fields": w.get("fields") or [],
        })
    return {
        "journeyTitle": raw.get("journeyTitle") or "Your Journey",
        "phases": [{"label": "Your Journey", "weeks": weeks_out}],
    }


@app.post("/api/generate-journey")
def generate_journey():
    body = request.get_json(force=True) or {}
    weeks = max(4, min(8, int(body.get("estimatedWeeks") or 6)))

    research_prompt = RESEARCH_PROMPT.format(
        problem=body.get("problem", ""),
        current_level=body.get("currentLevel", ""),
        weekly_minutes=body.get("weeklyTimeMinutes", "unspecified"),
        format_pref=body.get("formatPreference", "no strong preference"),
        notes=body.get("notes") or "none given",
        weeks=weeks,
    )

    research_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())] if USE_SEARCH_GROUNDING else None,
    )
    try:
        research_resp = client.models.generate_content(
            model=JOURNEY_MODEL,
            contents=research_prompt,
            config=research_config,
        )
    except Exception as e:
        print(f"[generate-journey] research call failed: {e}")
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            raise ApiError(quota_error_message(msg), 429)
        raise ApiError("Couldn't research this journey right now. Please try again.", 502)

    research_text = (research_resp.text or "").strip()
    if not research_text:
        raise ApiError("Couldn't find real resources for this topic. Try describing it differently.", 502)

    structure_text = generate_text(
        model=JOURNEY_MODEL,
        contents=STRUCTURE_PROMPT.format(research=research_text),
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    try:
        raw = json.loads(structure_text)
    except json.JSONDecodeError:
        raise ApiError("The AI's journey design wasn't valid JSON — please try again.", 502)

    return jsonify(build_phases(raw))


if __name__ == "__main__":
    app.run(port=5051, debug=True)
