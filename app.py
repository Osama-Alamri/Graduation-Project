import streamlit as st
import sqlite3
import hashlib
import json
from datetime import datetime

# ============================ PAGE CONFIG & STYLES ============================
st.set_page_config(page_title="Recruitment System", page_icon=":guardsman:", layout="wide")

st.markdown(
    """
    <style>
    .stTabs [role="tablist"]{justify-content:center}
    .stTabs [role="tab"]{font-size:50px;padding:30px 50px}
    .badge{display:inline-block;padding:4px 10px;border-radius:999px;
           background:rgba(124,58,237,.12);color:#7c3aed;font-weight:600;font-size:12px}
    .meta{color:gray;font-size:13px}
    </style>
    """,
    unsafe_allow_html=True,
)

ss = st.session_state

# --------- compatibility helpers ---------
def segmented(label, options, default, key=None):
    """Use segmented_control if available, else radio (horizontal)."""
    try:
        return st.segmented_control(label, options=options, default=default, key=key)
    except Exception:
        return st.radio(label, options, index=options.index(default), horizontal=True, key=key)


def _rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ============================ DB HELPERS ============================
@st.cache_resource
def get_conn():
    conn = sqlite3.connect("recruitment.db", check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

conn = get_conn()

# ---------- utilities ----------
def hash_pw(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()


def column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table});")
    cols = [r[1] for r in cur.fetchall()]
    return col in cols


def ensure_schema(conn: sqlite3.Connection):
    # users
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            role TEXT CHECK(role IN ('hr','candidate')) NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT
        );
        """
    )

    # jobs
    conn.execute(
        """
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
            FOREIGN KEY(created_by) REFERENCES users(id) ON DELETE SET NULL
        );
        """
    )

    # migrations (add columns if missing)
    try:
        if not column_exists(conn, "jobs", "location"):
            conn.execute("ALTER TABLE jobs ADD COLUMN location TEXT;")
    except sqlite3.OperationalError:
        pass

    # applications
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            job_id INTEGER NOT NULL,
            answers TEXT,
            cv_url TEXT,
            status TEXT DEFAULT 'submitted',
            created_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
        );
        """
    )

    conn.commit()


ensure_schema(conn)

# Seed a demo HR user (only once)
try:
    conn.execute(
        "INSERT OR IGNORE INTO users(name,email,role,password_hash,created_at) VALUES(?,?,?,?,?)",
        (
            "Demo HR",
            "hr@demo.com",
            "hr",
            hash_pw("hr123"),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
except Exception:
    pass

# ============================ AUTH HELPERS ============================

def get_user_by_email(email: str):
    cur = conn.execute("SELECT id,name,email,role,password_hash FROM users WHERE email=?", (email,))
    row = cur.fetchone()
    if row:
        return {"id": row[0], "name": row[1], "email": row[2], "role": row[3], "password_hash": row[4]}
    return None


def create_user(name: str, email: str, role: str, password: str):
    conn.execute(
        "INSERT INTO users(name,email,role,password_hash,created_at) VALUES(?,?,?,?,?)",
        (name, email, role, hash_pw(password), datetime.utcnow().isoformat()),
    )
    conn.commit()


def authenticate(email: str, password: str):
    user = get_user_by_email(email)
    if not user:
        return None
    return user if user["password_hash"] == hash_pw(password) else None


# Keep logged-in user in session
if "user" not in ss:
    ss.user = None

# ============================ AUTH UI (TOP-RIGHT) ============================
with st.container():
    left, right = st.columns([3, 2])
    with left:
        st.title("🤖 AI Recruitment System (Demo)")
        st.caption("Login/signup for HR & Candidates. HR gets Jobs, Applicants & Dashboard.")
    with right:
        if ss.user:
            st.success(f"Signed in as {ss.user['name']} ({ss.user['role']})")
            if st.button("Logout", use_container_width=True):
                ss.user = None
                _rerun()
        else:
            forced = bool(ss.pop("_show_login", False))
            auth_mode = segmented("Auth", options=["Login", "Sign up"], default="Login", key="auth_segmented_main")
            auth_mode = st.segmented_control("Auth", options=["Login", "Sign up"], default="Login")
            if auth_mode == "Login":
                with st.form("login_form"):
                    email = st.text_input("Email", placeholder="you@example.com")
                    password = st.text_input("Password", type="password")
                    submitted = st.form_submit_button("Login")
                if submitted:
                    user = authenticate(email, password)
                    if user:
                        ss.user = {k: v for k, v in user.items() if k != "password_hash"}
                        _rerun()
                    else:
                        st.error("Invalid email or password")
            else:
                with st.form("signup_form"):
                    name = st.text_input("Full name")
                    email = st.text_input("Email", placeholder="you@example.com")
                    role = st.selectbox("Role", ["candidate", "hr"], index=0)
                    password = st.text_input("Password", type="password")
                    confirm = st.text_input("Confirm password", type="password")
                    submitted = st.form_submit_button("Create account")
                if submitted:
                    if not name or not email or not password:
                        st.error("Please fill all fields")
                    elif password != confirm:
                        st.error("Passwords do not match")
                    elif get_user_by_email(email):
                        st.error("Email already registered")
                    else:
                        try:
                            create_user(name, email, role, password)
                            st.success("Account created! Please log in.")
                        except sqlite3.IntegrityError:
                            st.error("Email already registered")
                        except Exception as e:
                            st.error(f"Error: {e}")

# ============================ PUBLIC TABS ============================
homeTab, aboutTab, contactTab, publicJobsTab = st.tabs(["Home", "About us", "Contact", "Jobs"])

with homeTab:
    st.header("Home")
    st.write("Sign up as **Candidate** to browse/apply to jobs, or as **HR** to post jobs and view a dashboard.")
    st.info("Demo HR account: **hr@demo.com** / **hr123**")

with aboutTab:
    st.header("About us")
    st.write("This is a minimal demo to showcase role-based tabs with Streamlit and SQLite.")

with contactTab:
    st.header("Contact")
    st.write("Get in touch: hello@example.com")

with publicJobsTab:
    st.header("Open Jobs (Public)")
    st.caption("Anyone can view jobs. **Login to apply.**")
    jobs = conn.execute(
        "SELECT id,title,description,location,created_at FROM jobs ORDER BY id DESC"
    ).fetchall()
    if not jobs:
        st.info("No jobs yet. Check back later!")
    for jid, title, desc, loc, ts in jobs:
        with st.expander(f"#{jid} • {title}"):
            st.markdown(f"**Location:** {loc or '—'}")
            st.write(desc)
            st.caption(f"Posted: {ts}")
            if not ss.user:
                st.warning("Login or sign up to apply")
                if st.button("Login to apply", key=f"login_prompt_{jid}"):
                    st.session_state["_show_login"] = True
                    _rerun()
            elif ss.user and ss.user.get("role") == "candidate":
                with st.form(f"apply_public_{jid}"):
                    cv_url = st.text_input("CV link (Optional)", placeholder="https://…")
                    note = st.text_area("Cover note / short intro (Optional)")
                    submitted = st.form_submit_button("Apply")
                if submitted:
                    payload = {"note": note}
                    conn.execute(
                        "INSERT INTO applications(user_id,job_id,answers,cv_url,created_at) VALUES(?,?,?,?,?)",
                        (ss.user["id"], jid, json.dumps(payload), cv_url, datetime.utcnow().isoformat()),
                    )
                    conn.commit()
                    st.success("Applied!")
            else:
                st.info("Switch to a Candidate account to apply.")
    st.header("Contact")
    st.write("Get in touch: hello@example.com")

# ============================ ROLE-BASED AREA ============================
if ss.user:
    if ss.user["role"] == "hr":
        manageTab, dashboardTab, accountTab = st.tabs(["Management Jobs", "Dashboard", "Account"])

        # ---------------- Management Jobs (HR Only) ----------------
        with manageTab:
            st.subheader("Management Jobs")
            st.caption("Only HR can see this page. Post jobs and view each job's leaderboard of applicants.")

            if "show_post_form" not in ss:
                ss.show_post_form = False

            if st.button("➕ Post Job"):
                ss.show_post_form = True

            if ss.show_post_form:
                st.divider()
                st.markdown("### Create a Job Post")
                with st.form("job_post_form", clear_on_submit=False):
                    title = st.text_input("Job Title", placeholder="e.g., Data Scientist")
                    reqs = st.text_area("Job Requirements", placeholder="List the key requirements…")
                    location = st.text_input("Location", placeholder="e.g., Riyadh, KSA")
                    q_count = st.number_input("Number of questions", min_value=0, max_value=100, value=5, step=1)
                    colA, colB = st.columns(2)
                    with colA:
                        cv_pct = st.slider("Questions from CV (%)", 0, 100, 50, step=5)
                    with colB:
                        req_pct = st.slider("Questions from Requirements (%)", 0, 100, 50, step=5)
                    extra = st.text_area("Manual questions (optional, one per line)")
                    submitted = st.form_submit_button("Post Job")

                if submitted:
                    if not title or not reqs:
                        st.error("Please provide both Job Title and Job Requirements.")
                    elif cv_pct + req_pct != 100:
                        st.error("The sum of CV% and Requirement% must equal 100.")
                    else:
                        extras = [q.strip() for q in extra.splitlines() if q.strip()]
                        conn.execute(
                            """
                            INSERT INTO jobs(title,description,questions_count,cv_weight,req_weight,extra_questions,created_by,created_at,location)
                            VALUES(?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                title,
                                reqs,
                                int(q_count),
                                float(cv_pct) / 100.0,
                                float(req_pct) / 100.0,
                                json.dumps(extras),
                                ss.user["id"],
                                datetime.utcnow().isoformat(),
                                location,
                            ),
                        )
                        conn.commit()
                        st.success("Job posted!")
                        ss.show_post_form = False
                        _rerun()

            st.divider()
            st.subheader("Your Jobs")
            jobs = conn.execute(
                "SELECT id,title,created_at FROM jobs WHERE created_by=? ORDER BY id DESC",
                (ss.user["id"],),
            ).fetchall()
            if jobs:
                for jid, jt, ts in jobs:
                    with st.expander(f"#{jid} • {jt}"):
                        j = conn.execute(
                            "SELECT description,questions_count,cv_weight,req_weight,extra_questions,location FROM jobs WHERE id=?",
                            (jid,),
                        ).fetchone()
                        reqs, qn, cv_w, req_w, extra_json, loc = j
                        st.markdown(f"**Location:** {loc or '—'}")
                        st.markdown("**Job Requirements:**")
                        st.write(reqs)
                        st.caption(f"Questions: {qn} | From CV: {int(cv_w*100)}% | From Requirements: {int(req_w*100)}% | Posted: {ts}")
                        extras = json.loads(extra_json) if extra_json else []
                        if extras:
                            st.markdown("**Manual Questions:**")
                            for i, q in enumerate(extras, 1):
                                st.write(f"{i}. {q}")

                        # ---- Leaderboard (Applications for this job) ----
                        with st.expander("📊 Leaderboard (Applications)"):
                            rows = conn.execute(
                                """
                                SELECT a.id, u.name, u.email, a.status, a.cv_url, a.created_at
                                FROM applications a
                                JOIN users u ON u.id = a.user_id
                                WHERE a.job_id=?
                                ORDER BY a.id DESC
                                """,
                                (jid,),
                            ).fetchall()
                            if rows:
                                for aid, uname, uemail, status, cvurl, ats in rows:
                                    st.write(f"- #{aid} • **{uname}** — _{status}_ • CV: {cvurl or '—'} • {ats}")
                            else:
                                st.info("No applications yet for this job.")
            else:
                st.info("No jobs yet. Click **Post Job** to create one.")

        # ---------------- Dashboard (HR) ----------------
        with dashboardTab:
            st.subheader("HR Dashboard")
            c1, c2, c3 = st.columns(3)
            total_jobs = conn.execute("SELECT COUNT(*) FROM jobs WHERE created_by=?", (ss.user["id"],)).fetchone()[0]
            total_apps = conn.execute(
                "SELECT COUNT(*) FROM applications a JOIN jobs j ON j.id=a.job_id WHERE j.created_by=?",
                (ss.user["id"],),
            ).fetchone()[0]
            in_review = conn.execute(
                "SELECT COUNT(*) FROM applications a JOIN jobs j ON j.id=a.job_id WHERE j.created_by=? AND a.status='in review'",
                (ss.user["id"],),
            ).fetchone()[0]
            c1.metric("Jobs", total_jobs)
            c2.metric("Applications", total_apps)
            c3.metric("In review", in_review)

        # ---------------- Account (HR) ----------------
        with accountTab:
            st.subheader("Account")
            st.write(f"**Name:** {ss.user['name']}")
            st.write(f"**Email:** {ss.user['email']}")
            st.write(f"**Role:** {ss.user['role']}")

    else:  # Candidate
        browseTab, myAppsTab, accountTab = st.tabs(["Browse Jobs", "My Applications", "Account"])

        # ---------------- Browse Jobs (Candidate) ----------------
        with browseTab:
            st.subheader("Open positions")
            jobs = conn.execute(
                "SELECT id,title,description,location,created_at FROM jobs ORDER BY id DESC"
            ).fetchall()
            if not jobs:
                st.info("No jobs yet. Check back later!")
            for jid, title, desc, loc, ts in jobs:
                with st.expander(f"#{jid} • {title}"):
                    st.markdown(f"**Location:** {loc or '—'}")
                    st.write(desc)
                    st.caption(f"Posted: {ts}")
                    with st.form(f"apply_{jid}"):
                        cv_url = st.text_input("CV link (Optional)", placeholder="https://…")
                        note = st.text_area("Cover note / short intro (Optional)")
                        submitted = st.form_submit_button("Apply")
                    if submitted:
                        payload = {"note": note}
                        conn.execute(
                            "INSERT INTO applications(user_id,job_id,answers,cv_url,created_at) VALUES(?,?,?,?,?)",
                            (ss.user["id"], jid, json.dumps(payload), cv_url, datetime.utcnow().isoformat()),
                        )
                        conn.commit()
                        st.success("Applied!")

        # ---------------- My Applications (Candidate) ----------------
        with myAppsTab:
            st.subheader("My Applications")
            rows = conn.execute(
                """
                SELECT a.id, j.title, a.status, a.created_at
                FROM applications a JOIN jobs j ON j.id=a.job_id
                WHERE a.user_id=?
                ORDER BY a.id DESC
                """,
                (ss.user["id"],),
            ).fetchall()
            if rows:
                for aid, jtitle, status, ts in rows:
                    st.write(f"- #{aid} • **{jtitle}** — _{status}_  ({ts})")
            else:
                st.info("You haven't applied to any jobs yet.")

        # ---------------- Account (Candidate) ----------------
        with accountTab:
            st.subheader("Account")
            st.write(f"**Name:** {ss.user['name']}")
            st.write(f"**Email:** {ss.user['email']}")
            st.write(f"**Role:** {ss.user['role']}")

else:
    st.info("🔐 Log in or sign up (top-right) to access role-based tabs.")
