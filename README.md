# AI-Powered Recruitment and Candidate Ranking System  
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite)
![OpenAI](https://img.shields.io/badge/OpenAI-LLM-412991?logo=openai)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikitlearn)
![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A full **AI-powered recruitment platform** that automates resume parsing, CV–JD matching, LLM-driven interviews, evaluation scoring, and HR candidate ranking.

🔗 **GitHub Repository:**  
https://github.com/Osama-Alamri/Graduation-Project.git

---

<p align="center">
  <img src="assets/PROJECT_LOGO.png" alt="AI Recruitment Logo" width="260">
</p>

---

# 📌 Overview

This system provides:
- Resume analysis using NLP  
- TF-IDF + cosine similarity scoring  
- Dynamic weight-based evaluation  
- AI-driven interview generation & scoring  
- HR dashboard for ranking and analytics  
- Candidate portal for CV submission and interview access  

---

# 🧩 Core Libraries (Quick Summary)

### • Streamlit  
Frontend framework for the UI interfaces.

### • SQLite  
Database engine for storing system data.

### • OpenAI GPT-4o-mini  
Generates interview questions and grades answers.

### • Scikit-Learn  
Handles TF-IDF vectorization and similarity scores.

### • NLTK  
Used to preprocess CV text.

### • Plotly  
Creates radar charts to visualize candidate performance.

---

# ⚙️ Installation & Setup

## 1) Clone the Project
```bash
git clone https://github.com/Osama-Alamri/Graduation-Project.git
cd Graduation-Project
```

## 2) Install Required Libraries
```bash
pip install -r requirements.txt
```

---

# 🔐 API Key (Required for AI Interview)

AI features require an **OpenAI API Key**.

**

Make a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

Place `.env` in the root directory of the project.

---

# ▶️ Running the System

Start the app:
```bash
streamlit run app.py
```

Then open:
```
http://localhost:8501
```

---

<br>

# 📦 System Components  

---

<details>
  <summary><b>🧠 Resume Parsing Pipeline</b></summary>
  <br>

- Extracts PDF text  
- Detects CV sections (skills, education, projects…)  
- Cleans & preprocesses using NLTK  
- Calculates real working years  
- Generates structured CV output for scoring  

</details>

---

<details>
  <summary><b>🎯 Candidate Matching Engine</b></summary>
  <br>

Uses **TF-IDF + Cosine Similarity** with weighted scoring:

| Section | Weight |
|--------|--------|
| Years of Experience | 10% |
| Skills | 25% |
| Experience | 15% |
| Education | 20% |
| Certificates | 30% |

Outputs:
- Section similarity  
- Experience validation  
- Final match score (%)  

</details>

---

<details>
  <summary><b>🤖 AI Interview Engine</b></summary>
  <br>

- Auto-generated questions  
- Difficulty levels  
- CV-based + JD-based  
- Chat-style UI  
- AI scoring (0–100)  
- Transcript saved into the database  

</details>

---

<details>
  <summary><b>📊 HR Dashboard</b></summary>
  <br>

- Candidate leaderboard  
- Radar charts  
- Application breakdown  
- View match score + interview score  

</details>

---

<br>

# 🗄 Database Schema (Expanded for Detail)

<details>
  <summary><b>👥 Users Table</b></summary>
  <br>

| Column | Type | Description |
|--------|------|-------------|
| user_id | INTEGER | Primary key |
| username | TEXT | Unique |
| password_hash | TEXT | Secure hash |
| role | TEXT | hr / candidate |

</details>

---

<details>
  <summary><b>📌 Jobs Table</b></summary>
  <br>

| Column | Type | Description |
|--------|------|-------------|
| job_id | INTEGER | Primary key |
| title | TEXT | Job title |
| skills | TEXT | Required skills |
| experience | TEXT | Experience rules |
| education | TEXT | Education requirements |
| certificates | TEXT | Certifications |
| projects | TEXT | Required projects |
| years_rule | TEXT | "3-5", "5+" ... |
| num_questions | INTEGER | Interview Q count |
| cv_question_ratio | INTEGER | CV % |
| jd_question_ratio | INTEGER | JD % |
| question_difficulty | TEXT | Simple/Normal/Professional |
| manual_questions | TEXT | Custom HR questions |

</details>

---

<details>
  <summary><b>📄 Applications Table</b></summary>
  <br>

| Column | Type | Description |
|--------|------|-------------|
| application_id | INTEGER | Primary key |
| job_id | INTEGER | Job ID |
| user_id | INTEGER | Candidate ID |
| resume_filename | TEXT | Uploaded CV |
| scanned_experience_years | REAL | Years |
| scanned_skills_text | TEXT | Skills |
| scanned_experience_text | TEXT | Experience |
| scanned_education_text | TEXT | Education |
| scanned_certificates_text | TEXT | Certificates |
| scanned_projects_text | TEXT | Projects |
| match_scores | REAL | Similarity results |
| interview_questions_json | TEXT | AI Questions |
| interview_answers_json | TEXT | Answers |
| interview_ai_score | REAL | Score |
| interview_status | TEXT | pending/completed |
| status | TEXT | Application status |

</details>

---

# 📂 Project Structure

```
📁 Graduation-Project
 ├── app.py
 ├── scanner.py
 ├── matcher.py
 ├── db_setup.py
 ├── job_description.json
 ├── requirements.txt
 ├── .env
 ├── resumes/
 ├── data/
 ├── uploads/
 ├── ai_detector/
 └── README.md
```

---

# 📄 License (MIT)

See: **License** file in project root.

