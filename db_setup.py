import sqlite3

DB_FILE = "recruitment1.db"

def create_tables():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # === Users Table ===
    c.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT CHECK(role IN ('hr', 'candidate')) NOT NULL
        );
    """)

    # === Jobs Table ===
    c.execute("""
        CREATE TABLE IF NOT EXISTS Jobs (
            job_id INTEGER PRIMARY KEY AUTOINCREMENT,
            hr_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            jd_skills TEXT,
            jd_experience TEXT,
            jd_education TEXT,
            jd_certificates TEXT,
            jd_projects TEXT,
            jd_years_rule TEXT,
            num_questions INTEGER DEFAULT 10,
            cv_question_ratio INTEGER DEFAULT 50,
            jd_question_ratio INTEGER DEFAULT 50,
            question_difficulty TEXT DEFAULT 'Normal',
            manual_questions TEXT,
            FOREIGN KEY (hr_id) REFERENCES Users(user_id)
        );
    """)

    # === Applications Table ===
    c.execute("""
        CREATE TABLE IF NOT EXISTS Applications (
            application_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            candidate_id INTEGER NOT NULL,
            status TEXT DEFAULT 'applied',
            resume_filename TEXT,

            scanned_experience_years TEXT,
            scanned_skills_text TEXT,
            scanned_experience_text TEXT,
            scanned_education_text TEXT,
            scanned_certificates_text TEXT,
            scanned_projects_text TEXT,

            match_overall_score REAL,
            match_skills_score REAL,
            match_experience_score REAL,
            match_education_score REAL,
            match_certificates_score REAL,
            match_projects_score REAL,
            match_years_score REAL,

            interview_status TEXT DEFAULT 'pending',
            interview_questions_json TEXT,
            interview_answers_json TEXT,
            interview_ai_score REAL,

            FOREIGN KEY (job_id) REFERENCES Jobs(job_id),
            FOREIGN KEY (candidate_id) REFERENCES Users(user_id)
        );
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized successfully.")

def migrate_existing_jobs():
    """small helper to add the new interview columns to Jobs table if the DB is old."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # columns we expect to have in Jobs table
    new_cols = [
        ("num_questions", "INTEGER", "10"),
        ("cv_question_ratio", "INTEGER", "50"),
        ("jd_question_ratio", "INTEGER", "50"),
        ("question_difficulty", "TEXT", "'Normal'"),
        ("manual_questions", "TEXT", "NULL")
    ]

    c.execute("PRAGMA table_info(Jobs);")
    existing = {row[1] for row in c.fetchall()}

    for col, col_type, default in new_cols:
        if col not in existing:
            print(f"🆕 Adding missing column: {col}")
            c.execute(f"ALTER TABLE Jobs ADD COLUMN {col} {col_type} DEFAULT {default};")

    conn.commit()
    conn.close()
    print("✅ Migration complete: All columns ensured.")

if __name__ == "__main__":
    create_tables()
    migrate_existing_jobs()
