# pages/apply.py
import streamlit as st
import sqlite3
import os
import io
import json
import random
from datetime import datetime
import PyPDF2

ss = st.session_state

# ============================ DB ============================
@st.cache_resource
def get_conn():
    conn = sqlite3.connect("recruitment.db", check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

# ============================ Helpers ============================
def get_job(conn, job_id: int):
    row = conn.execute("""
        SELECT id, title, description, questions_count, cv_weight, req_weight,
               extra_questions, location, certificates, projects, experience, education, skills,
               created_by, created_at
        FROM jobs WHERE id=?
    """, (job_id,)).fetchone()
    if not row:
        return None
    keys = ["id","title","description","questions_count","cv_weight","req_weight",
            "extra_questions","location","certificates","projects","experience","education","skills",
            "created_by","created_at"]
    j = dict(zip(keys, row))
    j["extra_questions"] = json.loads(j["extra_questions"]) if j["extra_questions"] else []
    return j

def ensure_upload_dir():
    os.makedirs("uploads", exist_ok=True)

def read_pdf_bytes_to_text(file_bytes: bytes) -> str:
    text = ""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def plan_questions(job, use_manual: bool):
    q_count = int(job["questions_count"] or 3)
    q_count = max(1, q_count)

    if use_manual and job["extra_questions"]:
        sample = job["extra_questions"][:]
        random.shuffle(sample)
        manual_take = min(q_count, len(sample))
        return {
            "mode": "manual",
            "total": q_count,
            "manual_selected": sample[:manual_take],
            "from_cv": 0,
            "from_req": q_count - manual_take
        }

    cv_ratio = float(job["cv_weight"] or 0.5)
    from_cv = round(q_count * cv_ratio)
    from_req = q_count - from_cv
    return {
        "mode": "auto",
        "total": q_count,
        "manual_selected": [],
        "from_cv": from_cv,
        "from_req": from_req
    }

def get_job_id_from_url_or_session() -> int:
    # Only use st.query_params plus sticky session fallback
    job_id = 0
    try:
        raw = st.query_params.get("job_id")
        if isinstance(raw, str):
            job_id = int(raw) if raw.isdigit() else 0
        elif isinstance(raw, list) and raw:
            job_id = int(raw[0]) if str(raw[0]).isdigit() else 0
    except Exception:
        pass

    if job_id == 0:
        job_id = int(st.session_state.get("current_job_id") or
                     st.session_state.get("pending_job_id") or 0)

    if job_id:
        st.session_state["current_job_id"] = int(job_id)  # make it sticky for reruns

    return job_id


def goto_chatbot(job_id: int):
    st.query_params.update({"job_id": str(job_id)})
    try:
        st.switch_page("pages/chatbot_a.py")  # keep your filename
    except Exception:
        st.page_link("pages/chatbot_a.py", label="Continue to Interview ➜", icon="💬")
        st.stop()


# ============================ UI ============================
def main():
    st.set_page_config(page_title="Apply", layout="centered")
    conn = get_conn()

    # must be logged in (candidate)
    if "user" not in ss or not ss.get("user"):
        st.error("Please login as Candidate before applying.")
        st.stop()

    # job_id (sticky across reruns)
    job_id = get_job_id_from_url_or_session()
    if not job_id:
        st.error("No job selected. Please start from the Jobs page.")
        st.stop()

    job = get_job(conn, job_id)
    if not job:
        st.error("Job not found.")
        st.stop()

    st.title("📄 Apply to Job")
    st.subheader(job["title"])
    st.caption(f"Location: {job.get('location') or '—'} • Posted: {job['created_at']}")

    st.markdown("### Candidate Profile (expected)")
    for label, val in [
        ("Certificates / Courses", job.get("certificates")),
        ("Projects", job.get("projects")),
        ("Experience", job.get("experience")),
        ("Education", job.get("education")),
        ("Skills", job.get("skills")),
    ]:
        if val:
            st.markdown(f"**{label}**")
            st.write(val)

    st.markdown("### Job Description")
    st.write(job["description"])

    st.divider()
    st.markdown("### Your Application")

    use_manual_default = True if job["extra_questions"] else False
    mode = st.radio(
        "Interview Question Source",
        options=["Use HR Manual Questions (if any)", "Auto-generate from CV & Job Description"],
        index=0 if use_manual_default else 1,
        key=f"mode_{job_id}",  # key avoids cross-job bleed
    )
    use_manual = (mode == "Use HR Manual Questions (if any)")

    # Use a stable key so the uploader keeps its state on rerun
    with st.form(f"apply_form_{job_id}", clear_on_submit=False):
        uploaded_pdf = st.file_uploader(
            "Upload your CV (PDF only)",
            type=["pdf"],
            key=f"cv_uploader_job_{job_id}"
        )
        note = st.text_area("Cover note / short intro (Optional)", key=f"note_{job_id}")
        start = st.form_submit_button("Start Interview 🚀", use_container_width=True)

    if start:
        if not uploaded_pdf:
            st.error("Please upload your CV (PDF).")
            st.stop()

        # read CV text
        try:
            file_bytes = uploaded_pdf.read()
            cv_text = read_pdf_bytes_to_text(file_bytes)
            if not cv_text:
                st.error("Could not extract text from PDF. Please upload a text-based PDF.")
                st.stop()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            st.stop()

        # save CV file + insert application
        ensure_upload_dir()
        filename = f"uploads/job{job_id}_user{ss['user']['id']}_{int(datetime.utcnow().timestamp())}.pdf"
        with open(filename, "wb") as f:
            f.write(file_bytes)

        payload = {"note": note}
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO applications(user_id,job_id,answers,cv_url,created_at,status) VALUES(?,?,?,?,?,?)",
            (ss["user"]["id"], job_id, json.dumps(payload), filename, datetime.utcnow().isoformat(), "in review"),
        )
        conn.commit()
        app_id = cur.lastrowid

        # plan + prime session for chatbot
        interview_plan = plan_questions(job, use_manual=use_manual)
        ss["interview_started"] = True
        ss["cv_text"] = cv_text
        ss["application_id"] = app_id
        ss["interview_job"] = {
            "id": job["id"],
            "title": job["title"],
            "description": job["description"],
            "manual_questions": job["extra_questions"],
        }
        ss["interview_plan"] = interview_plan

        # NOW it's safe to clear the fallbacks (we're done on Apply)
        ss.pop("pending_job_id", None)
        # keep current_job_id if you want refresh to stay on job page; or clear:
        # ss.pop("current_job_id", None)

        goto_chatbot(job_id)

if __name__ == "__main__":
    main()
