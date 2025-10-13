import streamlit as st
import os
import sqlite3
import json
from openai import OpenAI

# load .env into os.environ
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
except KeyError:
    st.error("There is No Key !!")
    st.stop()

api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is missing. Set it via .env or st.secrets.")
    st.stop()

from openai import OpenAI
client = OpenAI(api_key=api_key)

# ------------ DB ------------
@st.cache_resource
def get_conn():
    conn = sqlite3.connect("recruitment.db", check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

st.set_page_config(page_title="AI Job Interview", layout="centered")
st.title("🧠 AI Job Interview")

# ------------ Session Guards ------------
if not st.session_state.get("interview_started"):
    st.warning("Please start from the Apply page.")
    st.stop()

if not st.session_state.get("cv_text"):
    st.error("CV text is missing. Re-upload your CV from the Apply page.")
    st.stop()

job = st.session_state.get("interview_job", {})
plan = st.session_state.get("interview_plan", {"mode":"auto","total":3,"from_cv":2,"from_req":1,"manual_selected":[]})
application_id = st.session_state.get("application_id")

# ------------ Prompts ------------
def build_system_prompt(job, plan):
    title = job.get("title","")
    desc = job.get("description","")
    manual_qs = job.get("manual_questions", []) or []
    cv_text = st.session_state["cv_text"]

    if plan["mode"] == "manual" and manual_qs:
        manual_intro = (
            "You MUST ask the following questions in order (one by one). "
            "Do not add new questions unless you run out of manual ones.\n"
            + "\n".join([f"{i+1}. {q}" for i, q in enumerate(plan.get("manual_selected",[]))])
        )
    else:
        manual_intro = (
            f"Ask {plan['total']} questions total, one by one. "
            f"Generate ~{plan.get('from_cv',0)} from the candidate's CV and ~{plan.get('from_req',0)} from the Job Description. "
            "Vary styles: behavioral/technical/experience. Keep each question concise."
        )

    system_prompt = f"""
You are a professional HR interviewer (name: Rakaz). Language: English.
Company: KSU (fictional for interview).

JOB TITLE: {title}

JOB DESCRIPTION:
---
{desc}
---

CANDIDATE CV:
---
{cv_text}
---

INTERVIEW PLAN:
Mode: {plan['mode']}
{manual_intro}

Rules:
1) Ask ONLY one question at a time, then wait for candidate answer.
2) Keep questions short and specific. Avoid multi-part long questions.
3) After finishing all planned questions, politely end the interview.
""".strip()

    return system_prompt

def build_evaluation_prompt(job, chat_transcript_markdown: str):
    desc = job.get("description","")
    cv_text = st.session_state["cv_text"]
    rubric = """
Score the candidate out of 100 using this rubric (weights in parentheses):
1) Relevance of experience to job requirements (25)
2) Technical depth and correctness (25)
3) Communication clarity and structure (15)
4) Problem-solving/Reasoning in answers (20)
5) Overall fit and professionalism (15)

Return STRICT JSON with keys: {"score": int, "feedback": string}.
Score must be 0..100 (integer). Feedback: 2-4 concise bullet points (single line each).
"""
    prompt = f"""
You are an HR evaluator. Evaluate the following interview based on the job and CV.

JOB DESCRIPTION:
---
{desc}
---

CANDIDATE CV:
---
{cv_text}
---

INTERVIEW TRANSCRIPT (assistant=interviewer, user=candidate):
---
{chat_transcript_markdown}
---

{rubric}
""".strip()
    return prompt

def parse_score_json(s: str):
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = s[start:end+1]
            data = json.loads(raw)
            score = int(data.get("score", 0))
            fb = str(data.get("feedback","")).strip()
            score = max(0, min(100, score))
            return score, fb
    except Exception:
        pass
    return 0, "No feedback."

# ------------ Init Chat State ------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "asked_count" not in st.session_state:
    st.session_state.asked_count = 0
if "interview_closed" not in st.session_state:
    st.session_state.interview_closed = False

# Add system once
if not any(m.get("role") == "system" for m in st.session_state.messages):
    system_prompt = build_system_prompt(job, plan)
    st.session_state.messages.append({"role": "system", "content": system_prompt})

# First question
if st.session_state.asked_count == 0 and not st.session_state.interview_closed:
    init_user = "I'm ready to begin."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages] + [{"role":"user","content":init_user}]
    )
    first_q = response.choices[0].message.content
    st.session_state.messages.append({"role":"assistant", "content": first_q})
    st.session_state.asked_count += 1

# ------------ UI ------------
st.success(f"Interview • {job.get('title','')} • Plan: {plan['mode']} • Total Qs: {plan['total']}")
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_answer = st.chat_input("Write your answer here...")

def to_md_transcript(msgs):
    lines = []
    for m in msgs:
        if m["role"] == "assistant":
            lines.append("assistant: " + m["content"])
        elif m["role"] == "user":
            lines.append("user: " + m["content"])
    return "\n".join(lines)

def finalize_and_score():
    if st.session_state.interview_closed:
        return
    st.session_state.interview_closed = True

    with st.spinner("Scoring the interview..."):
        transcript_md = to_md_transcript(st.session_state.messages)
        eval_prompt = build_evaluation_prompt(job, transcript_md)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise evaluator that only returns strict JSON."},
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0.1
        )
        raw = resp.choices[0].message.content
        score, feedback = parse_score_json(raw)

        # Save in DB
        conn = get_conn()
        try:
            conn.execute(
                "UPDATE applications SET score=?, ai_feedback=?, status=? WHERE id=?",
                (int(score), feedback, "completed", int(application_id))
            )
            conn.commit()
        except Exception as e:
            st.error(f"DB save error: {e}")

        st.markdown("---")
        st.subheader("Your Interview Score")
        st.metric("Score", f"{score}/100")
        st.write("**Feedback:**")
        for line in feedback.split("\n"):
            if line.strip():
                st.write("• " + line.strip())

        closing = "Thank you for your time. The interview is complete. You may close this page."
        st.session_state.messages.append({"role":"assistant","content":closing})

if user_answer:
    st.session_state.messages.append({"role": "user", "content": user_answer})
    with st.chat_message("user"): st.markdown(user_answer)

    if st.session_state.asked_count >= plan["total"]:
        finalize_and_score()
        st.rerun()
    else:
        with st.spinner("Interviewer is thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            )
            next_q = response.choices[0].message.content
            st.session_state.messages.append({"role":"assistant","content":next_q})
            st.session_state.asked_count += 1

        if st.session_state.asked_count >= plan["total"]:
            st.info("This is the final question. After your answer, the interview will finish and be scored.")
        st.rerun()

if not st.session_state.interview_closed and st.button("Finish now & Score"):
    finalize_and_score()
    st.rerun()
