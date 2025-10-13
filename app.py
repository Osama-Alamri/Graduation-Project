import streamlit as st
import sqlite3
import hashlib
import json
from datetime import datetime

# ============================ CONFIG ============================
st.set_page_config(page_title="Recruitment System", page_icon="🤖", layout="wide")
st.markdown("""
<style>
.stTabs [role="tablist"]{justify-content:center}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;background:rgba(124,58,237,.12);color:#7c3aed;font-weight:600;font-size:12px}
.meta{color:gray;font-size:13px}
</style>
""", unsafe_allow_html=True)

ss = st.session_state

# ============================ UTIL ============================
def _rerun():
    try: st.rerun()
    except Exception:
        try: st.experimental_rerun()
        except Exception: pass

def hash_pw(p: str) -> str:
    import hashlib
    return hashlib.sha256(p.encode()).hexdigest()

@st.cache_resource
def get_conn():
    conn = sqlite3.connect("recruitment.db", check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table});")
    return col in [r[1] for r in cur.fetchall()]

# ============================ DB & SCHEMA ============================
def ensure_schema(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        role TEXT CHECK(role IN ('hr','candidate')) NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        questions_count INTEGER DEFAULT 0,
        cv_weight REAL DEFAULT 0.5,
        req_weight REAL DEFAULT 0.5,
        extra_questions TEXT,
        created_by INTEGER,
        created_at TEXT,
        location TEXT,
        certificates TEXT,
        projects TEXT,
        experience TEXT,
        education TEXT,
        skills TEXT,
        FOREIGN KEY(created_by) REFERENCES users(id) ON DELETE SET NULL
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        job_id INTEGER NOT NULL,
        answers TEXT,
        cv_url TEXT,
        status TEXT DEFAULT 'submitted',
        created_at TEXT,
        score INTEGER,
        ai_feedback TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
    );
    """)

    # Safe upgrades
    def add_col(tbl, col, typ):
        if not column_exists(conn, tbl, col):
            conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {typ};")

    add_col("jobs", "location", "TEXT")
    add_col("jobs", "certificates", "TEXT")
    add_col("jobs", "projects", "TEXT")
    add_col("jobs", "experience", "TEXT")
    add_col("jobs", "education", "TEXT")
    add_col("jobs", "skills", "TEXT")
    add_col("applications", "score", "INTEGER")
    add_col("applications", "ai_feedback", "TEXT")

    conn.commit()

def seed_demo_hr(conn: sqlite3.Connection):
    try:
        conn.execute(
            "INSERT OR IGNORE INTO users(name,email,role,password_hash,created_at) VALUES(?,?,?,?,?)",
            ("Demo HR", "hr@demo.com", "hr", hash_pw("hr123"), datetime.utcnow().isoformat()),
        )
        conn.commit()
    except Exception:
        pass

# ============================ AUTH ============================
def get_user_by_email(conn: sqlite3.Connection, email: str):
    cur = conn.execute("SELECT id,name,email,role,password_hash FROM users WHERE email=?", (email,))
    r = cur.fetchone()
    if not r: return None
    return {"id": r[0], "name": r[1], "email": r[2], "role": r[3], "password_hash": r[4]}

def create_user(conn: sqlite3.Connection, name: str, email: str, role: str, password: str):
    conn.execute(
        "INSERT INTO users(name,email,role,password_hash,created_at) VALUES(?,?,?,?,?)",
        (name, email, role, hash_pw(password), datetime.utcnow().isoformat()),
    )
    conn.commit()

def authenticate(conn: sqlite3.Connection, email: str, password: str):
    u = get_user_by_email(conn, email)
    if u and u["password_hash"] == hash_pw(password):
        return {k: v for k, v in u.items() if k != "password_hash"}
    return None

# ============================ UI: HEADER & AUTH ============================
def ui_header_and_auth(conn: sqlite3.Connection):
    if "user" not in ss: ss.user = None
    if "show_login" not in ss: ss.show_login = False
    if "show_signup" not in ss: ss.show_signup = False

    left, right = st.columns([3, 2])
    with left:
        st.title("🤖 AI Recruitment System (Demo)")
        st.caption("Login/signup for Candidates & HR. HR gets Management Jobs & Dashboard.")
    with right:
        if ss.user:
            st.success(f"Signed in as {ss.user['name']} ({ss.user['role']})")
            if st.button("Logout", use_container_width=True):
                ss.user = None; ss.show_login = False; ss.show_signup = False; _rerun()
        else:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Login", use_container_width=True):
                    ss.show_login, ss.show_signup = True, False; _rerun()
            with c2:
                if st.button("Sign up", use_container_width=True):
                    ss.show_signup, ss.show_login = True, False; _rerun()

            if ss.show_login:
                with st.form("login_form"):
                    st.markdown("**Login**")
                    email = st.text_input("Email", placeholder="you@example.com")
                    pw = st.text_input("Password", type="password")
                    submit = st.form_submit_button("Login", use_container_width=True)
                if submit:
                    user = authenticate(conn, email, pw)
                    if user:
                        ss.user = user; ss.show_login = False; _rerun()
                    else:
                        st.error("Invalid email or password")

            if ss.show_signup:
                with st.form("signup_form"):
                    st.markdown("**Create account**")
                    name = st.text_input("Full name")
                    email = st.text_input("Email", placeholder="you@example.com")
                    pw1 = st.text_input("Password", type="password")
                    pw2 = st.text_input("Confirm password", type="password")
                    submit = st.form_submit_button("Create account", use_container_width=True)
                if submit:
                    if not name or not email or not pw1:
                        st.error("Please fill all fields")
                    elif pw1 != pw2:
                        st.error("Passwords do not match")
                    elif get_user_by_email(conn, email):
                        st.error("Email already registered")
                    else:
                        try:
                            create_user(conn, name, email, role="candidate", password=pw1)
                            st.success("Account created! Please log in.")
                            ss.show_signup = False; ss.show_login = True
                        except sqlite3.IntegrityError:
                            st.error("Email already registered")
                        except Exception as e:
                            st.error(f"Error: {e}")

# ============================ UI: PUBLIC TABS ============================
def ui_public_tabs(conn: sqlite3.Connection):
    homeTab, aboutTab, contactTab, publicJobsTab = st.tabs(["Home", "About us", "Contact", "Jobs"])

    with homeTab:
        st.header("Home")
        st.write("Sign up as **Candidate** to browse/apply to jobs, or as **HR** to post jobs and view a dashboard.")
        st.info("Demo HR account: **hr@demo.com** / **hr123**")

    with aboutTab:
        st.header("About us")
        st.write("Minimal demo using Streamlit + SQLite.")

    with contactTab:
        st.header("Contact")
        st.write("Get in touch: hello@example.com")

    with publicJobsTab:
        st.header("Jobs")

        def go_apply(jid: int):
            # keep a fallback in session (useful across reruns)
            st.session_state["pending_job_id"] = int(jid)

            # set query params (new API only)
            st.query_params.update({"job_id": str(jid)})

            # switch to the Apply page (same filename/casing you’re using)
            try:
                st.switch_page("pages/apply.py")   # or "pages/Apply.py" if your file is capitalized
            except Exception:
                st.page_link("pages/apply.py", label="Continue to Apply ➜", icon="📝")
                st.stop()

        jobs = conn.execute("SELECT id,title,location,created_at FROM jobs ORDER BY id DESC").fetchall()

        for jid, title, loc, ts in jobs:
            with st.expander(f"#{jid} • {title}"):
                st.markdown(f"**Location:** {loc or '—'}")
                details = conn.execute("""
                    SELECT certificates, projects, experience, education, skills, description
                    FROM jobs WHERE id=?
                """, (jid,)).fetchone()

                if details:
                    certs, projects, exp, edu, skl, desc_full = details
                    if any([certs, projects, exp, edu, skl]):
                        st.markdown("### Candidate Profile")
                        if certs:   st.markdown("**Certificates / Courses**"); st.write(certs)
                        if projects:st.markdown("**Projects**"); st.write(projects)
                        if exp:     st.markdown("**Experience**"); st.write(exp)
                        if edu:     st.markdown("**Education**"); st.write(edu)
                        if skl:     st.markdown("**Skills**"); st.write(skl)

                    st.markdown("### Job Description")
                    st.write(desc_full or "")

                st.caption(f"Posted: {ts}")

                if not ss.user:
                    st.warning("Login or sign up to apply")
                elif ss.user["role"] == "candidate":
                    if st.button("Start application", key=f"start_app_{jid}", use_container_width=True):
                        go_apply(jid)
                else:
                    st.info("Switch to a Candidate account to apply.")

# ============================ UI: HR AREA ============================
def ui_hr_job_form(conn: sqlite3.Connection):
    st.markdown("### Create a Job Post")
    with st.form("job_post_form"):
        title = st.text_input("Job Title", placeholder="e.g., Data Scientist")
        location = st.text_input("Location", placeholder="e.g., Riyadh, KSA")

        st.markdown("#### Candidate Profile Sections")
        certs = st.text_area("Certificates / Courses", placeholder="e.g., AWS Practitioner, Coursera ML...")
        projects = st.text_area("Projects", placeholder="Notable projects related to the role…")
        experience = st.text_area("Experience", placeholder="Years, roles, companies, responsibilities…")
        education = st.text_area("Education", placeholder="Degrees, universities, GPA…")
        skills = st.text_area("Skills", placeholder="Python, NLP, SQL, FastAPI, Streamlit…")

        description = st.text_area("Job Description", placeholder="Describe the role, responsibilities, and requirements...", height=160)
        q_count = st.number_input("Number of questions", min_value=0, max_value=100, value=5, step=1)

        cv_pct = st.slider("Questions from CV (%)", 0, 100, 50, step=5, help="Requirements% will auto = 100 - CV%")
        req_pct = 100 - cv_pct
        st.caption(f"Questions from Requirements (%): **{req_pct}%**  •  Total = 100%")

        extra = st.text_area("Manual questions (optional, one per line)")
        submitted = st.form_submit_button("Post Job")

    if not submitted:
        return

    if not title or not description:
        st.error("Please provide both Job Title and Job Description."); return

    extras = [q.strip() for q in extra.splitlines() if q.strip()]
    conn.execute("""
        INSERT INTO jobs(
            title, description, questions_count,
            cv_weight, req_weight, extra_questions,
            created_by, created_at, location,
            certificates, projects, experience, education, skills
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        title, description, int(q_count),
        float(cv_pct)/100.0, float(req_pct)/100.0, json.dumps(extras),
        ss.user["id"], datetime.utcnow().isoformat(), location,
        certs, projects, experience, education, skills
    ))
    conn.commit()
    st.success("Job posted!")

def ui_hr_job_card(conn: sqlite3.Connection, jid: int, jt: str, ts: str):
    with st.expander(f"#{jid} • {jt}"):
        (desc, qn, cv_w, req_w, extra_json, loc,
         certs, projects, exp, edu, skl) = conn.execute("""
            SELECT description,questions_count,cv_weight,req_weight,extra_questions,location,
                   certificates, projects, experience, education, skills
            FROM jobs WHERE id=?
        """, (jid,)).fetchone()

        st.markdown(f"**Location:** {loc or '—'}")
        st.caption(f"Questions: {qn} | From CV: {int(cv_w*100)}% | From Requirements: {int(req_w*100)}% | Posted: {ts}")

        if any([certs, projects, exp, edu, skl]):
            st.markdown("### Candidate Profile")
            if certs:   st.markdown("**Certificates / Courses**"); st.write(certs)
            if projects:st.markdown("**Projects**"); st.write(projects)
            if exp:     st.markdown("**Experience**"); st.write(exp)
            if edu:     st.markdown("**Education**"); st.write(edu)
            if skl:     st.markdown("**Skills**"); st.write(skl)

        st.markdown("### Job Description")
        st.write(desc)

        extras = json.loads(extra_json) if extra_json else []
        if extras:
            st.markdown("**Manual Questions:**")
            for i, q in enumerate(extras, 1):
                st.write(f"{i}. {q}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Delete", key=f"del_{jid}"):
                conn.execute("DELETE FROM jobs WHERE id=?", (jid,))
                conn.commit()
                st.success("Job deleted"); _rerun()
        with c2:
            if st.button("✏️ Edit", key=f"edit_{jid}"):
                ss["edit_job_id"] = jid; _rerun()

        # Leaderboard with score/feedback
        with st.expander("📊 Leaderboard (Applications)"):
            rows = conn.execute("""
                SELECT a.id, u.name, u.email, a.status, a.cv_url, a.created_at, a.score, a.ai_feedback
                FROM applications a
                JOIN users u ON u.id = a.user_id
                WHERE a.job_id = ?
                ORDER BY COALESCE(a.score, -1) DESC, a.id DESC
            """, (jid,)).fetchall()
            if rows:
                for aid, uname, uemail, status, cvurl, ats, score, fb in rows:
                    score_txt = f"{score}/100" if score is not None else "—"
                    st.write(f"- #{aid} • **{uname}** — _{status}_ • CV: {cvurl or '—'} • {ats} • **Score:** {score_txt}")
                    if fb:
                        with st.expander(f"Feedback for #{aid}"):
                            st.write(fb)
            else:
                st.info("No applications yet for this job.")

def ui_hr_area(conn: sqlite3.Connection):
    manageTab, dashboardTab, accountTab = st.tabs(["Management Jobs", "Dashboard", "Account"])

    with manageTab:
        st.subheader("Management Jobs")
        if st.button("➕ Post Job"):
            ss.show_post = True
        if ss.get("show_post"):
            ui_hr_job_form(conn)
        st.divider()
        st.subheader("Your Jobs")
        jobs = conn.execute(
            "SELECT id,title,created_at FROM jobs WHERE created_by=? ORDER BY id DESC",
            (ss.user["id"],),
        ).fetchall()
        if not jobs:
            st.info("No jobs yet. Click **Post Job** to create one.")
        for jid, jt, ts in jobs:
            ui_hr_job_card(conn, jid, jt, ts)

    with dashboardTab:
        st.subheader("HR Dashboard")
        c1, c2, c3 = st.columns(3)
        total_jobs = conn.execute("SELECT COUNT(*) FROM jobs WHERE created_by=?", (ss.user["id"],)).fetchone()[0]
        total_apps = conn.execute("""
            SELECT COUNT(*) FROM applications a JOIN jobs j ON j.id=a.job_id WHERE j.created_by=?
        """, (ss.user["id"],)).fetchone()[0]
        in_review = conn.execute("""
            SELECT COUNT(*) FROM applications a JOIN jobs j ON j.id=a.job_id
            WHERE j.created_by=? AND a.status='in review'
        """, (ss.user["id"],)).fetchone()[0]
        c1.metric("Jobs", total_jobs); c2.metric("Applications", total_apps); c3.metric("In review", in_review)

    with accountTab:
        st.subheader("Account")
        st.write(f"**Name:** {ss.user['name']}")
        st.write(f"**Email:** {ss.user['email']}")
        st.write(f"**Role:** {ss.user['role']}")

# ============================ UI: CANDIDATE AREA ============================
def ui_candidate_area(conn: sqlite3.Connection):
    browseTab, myAppsTab, accountTab = st.tabs(["Browse Jobs", "My Applications", "Account"])

    with browseTab:
        st.subheader("Open positions")
        jobs = conn.execute("SELECT id,title,location,created_at FROM jobs ORDER BY id DESC").fetchall()
        if not jobs:
            st.info("No jobs yet. Check back later!")
        for jid, title, loc, ts in jobs:
            with st.expander(f"#{jid} • {title}"):
                st.markdown(f"**Location:** {loc or '—'}")
                details = conn.execute("""
                    SELECT certificates, projects, experience, education, skills, description
                    FROM jobs WHERE id=?
                """, (jid,)).fetchone()
                if details:
                    certs, projects, exp, edu, skl, desc_full = details
                    if any([certs, projects, exp, edu, skl]):
                        st.markdown("### Candidate Profile")
                        if certs:   st.markdown("**Certificates / Courses**"); st.write(certs)
                        if projects:st.markdown("**Projects**"); st.write(projects)
                        if exp:     st.markdown("**Experience**"); st.write(exp)
                        if edu:     st.markdown("**Education**"); st.write(edu)
                        if skl:     st.markdown("**Skills**"); st.write(skl)
                    st.markdown("### Job Description")
                    st.write(desc_full or "")

                st.caption(f"Posted: {ts}")
                if st.button("Start application", key=f"start_app_cand_{jid}", use_container_width=True):
                    # same nav helper logic
                    try:
                        st.query_params.update({"job_id": str(jid)})
                    except Exception:
                        st.experimental_set_query_params(job_id=str(jid))
                    try:
                        st.switch_page("pages/apply.py")
                    except Exception:
                        st.page_link("pages/apply.py", label="Continue to Apply ➜", icon="📝")
                        st.stop()

    with myAppsTab:
        st.subheader("My Applications")
        rows = conn.execute("""
            SELECT a.id, j.title, a.status, a.created_at, a.score
            FROM applications a JOIN jobs j ON j.id=a.job_id
            WHERE a.user_id=? ORDER BY a.id DESC
        """, (ss.user["id"],)).fetchall()
        if rows:
            for aid, jtitle, status, ts, score in rows:
                sc = f"{score}/100" if score is not None else "—"
                st.write(f"- #{aid} • **{jtitle}** — _{status}_  ({ts}) • Score: {sc}")
        else:
            st.info("You haven't applied to any jobs yet.")

    with accountTab:
        st.subheader("Account")
        st.write(f"**Name:** {ss.user['name']}")
        st.write(f"**Email:** {ss.user['email']}")
        st.write(f"**Role:** {ss.user['role']}")

# ============================ MAIN ============================
def main():
    conn = get_conn()
    ensure_schema(conn)
    seed_demo_hr(conn)

    ui_header_and_auth(conn)
    ui_public_tabs(conn)

    if ss.get("user"):
        if ss.user["role"] == "hr":
            ui_hr_area(conn)
        else:
            ui_candidate_area(conn)

if __name__ == "__main__":
    main()
