
import sqlite3

# --- 1. الاتصال بقاعدة البيانات ---
# هذا الكود سينشئ ملف "recruitment1.db" إذا لم يكن موجوداً
# أو سيتصل به إذا كان موجوداً
conn = sqlite3.connect('recruitment1.db')

# "المؤشر" (cursor) هو الأداة التي نستخدمها لتنفيذ الأوامر
c = conn.cursor()

# --- 2. تفعيل مفاتيح الربط (Foreign Keys) ---
# خطوة مهمة لضمان ترابط الجداول بشكل صحيح
c.execute("PRAGMA foreign_keys = ON;")

print("Conected to the DB..")

# --- 3. سنكتب أوامر إنشاء الجداول هنا ---

# الجدول 1: المستخدمون (Users)
c.execute('''
CREATE TABLE IF NOT EXISTS Users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('hr', 'candidate'))
);
''')
print("USER Table has been created")


# الجدول 2: الوظائف (Jobs)
c.execute('''
CREATE TABLE IF NOT EXISTS Jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    hr_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    jd_skills TEXT,
    jd_education TEXT,
    jd_experience TEXT,
    jd_years_rule TEXT,
    jd_projects TEXT,      
    jd_certificates TEXT,
    FOREIGN KEY (hr_id) REFERENCES Users (user_id)
);
''')
print("JOBS Table has been created")

# الجدول 3: التقديمات (Applications)
c.execute('''
CREATE TABLE IF NOT EXISTS Applications (
    application_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    candidate_id INTEGER NOT NULL,
    status TEXT DEFAULT 'applied',
    resume_filename TEXT,
    
    scanned_skills_text TEXT,
    scanned_education_text TEXT,
    scanned_experience_text TEXT,
    scanned_experience_years REAL,
    scanned_projects_text TEXT,
    scanned_certificates_text TEXT,
    
    match_overall_score INTEGER,
    match_skills_score INTEGER,
    match_education_score INTEGER,
    match_experience_score INTEGER,
    match_years_score INTEGER,
    match_projects_score INTEGER,
    match_certificates_score INTEGER,
    
    interview_ai_score INTEGER,
    interview_status TEXT DEFAULT 'pending',
          
    

    FOREIGN KEY (job_id) REFERENCES Jobs (job_id),
    FOREIGN KEY (candidate_id) REFERENCES Users (user_id)
);
''')
print("APPLICATION Table has been created")


try:
    c.execute("ALTER TABLE Applications ADD COLUMN interview_questions_json TEXT;")
    print("Added 'interview_questions_json' column to Applications table.")
except sqlite3.OperationalError as e:
    # This will probably print "duplicate column name: interview_questions_json"
    print(f"Note: Failed to add column, it likely already exists. (Error: {e})")

try:
    c.execute("ALTER TABLE Applications ADD COLUMN interview_answers_json TEXT;")
    print("Added 'interview_answers_json' column to Applications table.")
except sqlite3.OperationalError as e:
    print(f"Note: Failed to add 'interview_answers_json' column, it likely already exists. (Error: {e})")

# --- 4. حفظ التغييرات وإغلاق الاتصال ---
conn.commit() # حفظ أي تغييرات قمنا بها
conn.close() # إغلاق الاتصال

print("ALL TABELS HAVE BEEN CREATED/UPDATED")