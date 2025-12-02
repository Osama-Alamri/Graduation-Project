# AI-Powered Recruitment and Candidate Ranking System  
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite)
![OpenAI](https://img.shields.io/badge/OpenAI-LLM-412991?logo=openai)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikitlearn)
![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A complete **AI-powered recruitment and evaluation system** that performs automated resume parsing, smart CV–JD matching, AI-generated interview questions, LLM interview scoring, and HR candidate ranking.

---

# 🧩 Libraries & What They Do

Below is a simple explanation of the most important libraries used in the project:

### **🔹 Streamlit**
Framework for building the interactive web interface (candidate portal, HR dashboard).

### **🔹 SQLite3**
Lightweight database used to store users, jobs, applications, scoring, and interview transcripts.

### **🔹 OpenAI (GPT-4o-mini)**
Used for:
- Generating interview questions  
- Grading interview answers  
- Understanding CV/JD content  

**Requires an API key → must be placed in `.env` file**

---

### **🔹 pdfplumber / PyPDF2**
Extract text from uploaded PDF resumes.

### **🔹 NLTK**
Used for:
- Tokenization  
- Stopword removal  
- Lemmatization  
- Cleaning resume content  

### **🔹 Scikit-Learn**
Provides:
- TF-IDF vectorization  
- Cosine similarity for CV–JD matching  

### **🔹 Plotly**
Creates interactive radar charts for candidate evaluation visualization.

### **🔹 Passlib**
Used to securely hash passwords before saving them in the database.

---

# ⚙️ Installation & Setup

## **1. Clone the Project**
```bash
git clone https://github.com/yourusername/ai-recruitment.git
cd ai-recruitment
```

## **2. Install Required Libraries**
```bash
pip install -r requirements.txt
```

---

# 🔐 API Key Requirement (Very Important)

This system **requires an OpenAI API Key** to run AI interviews and scoring.

✔️ بدون API KEY النظام يعمل، لكن **ميزة المقابلة الذكية (AI Interview) لن تعمل**.

## **Place your API key inside `.env` file**

Create a file named:

```
.env
```

Inside it put:

```
OPENAI_API_KEY=your_api_key_here
```

Then make sure your system loads environment variables automatically.

---

# 🗄 Database Initialization

Before running the app for the first time:

```bash
python db_setup.py
```

This will create:

- Users Table  
- Jobs Table  
- Applications Table  
- Add missing migration columns automatically  

---

# ▶️ Running the Application

Start the Streamlit interface:

```bash
streamlit run app.py
```

Then open the system in your browser:
```
http://localhost:8501
```

---

# 🧠 System Components

## **1. Resume Parsing Pipeline**
Extracts and cleans:
- Skills  
- Education  
- Certificates  
- Projects  
- Experience  
- Working years  

## **2. Candidate Matching Engine**
Uses TF-IDF + Cosine Similarity with dynamic weights.

## **3. AI Interview Engine**
- Auto-generated questions  
- Difficulty levels  
- CV-based + JD-based questions  
- Real-time chat UI  
- Automatic LLM scoring  
- Transcript stored in the DB  

## **4. HR Dashboard**
- Leaderboard  
- Candidate ranking  
- Radar charts  
- Full application breakdown  

---

# 🗄 Database Schema

## **Users Table**
| Column | Type | Description |
|--------|------|-------------|
| user_id | INTEGER | Primary key |
| username | TEXT | Unique |
| password_hash | TEXT | Hashed password |
| role | TEXT | hr / candidate |

---

## **Jobs Table**
| Column | Type | Description |
|--------|------|-------------|
| job_id | INTEGER | Primary key |
| title | TEXT | Job title |
| skills | TEXT | Required skills |
| experience | TEXT | Experience rules |
| education | TEXT | Education requirements |
| certificates | TEXT | Certifications |
| projects | TEXT | Required projects |
| years_rule | TEXT | e.g., "3-5", "5+", "2" |
| num_questions | INTEGER | Interview questions |
| cv_question_ratio | INTEGER | CV % |
| jd_question_ratio | INTEGER | JD % |
| question_difficulty | TEXT | Simple / Normal / Professional |
| manual_questions | TEXT | Optional HR-added questions |

---

## **Applications Table**
| Column | Type | Description |
|--------|------|-------------|
| application_id | INTEGER | Primary key |
| job_id | INTEGER | Job ID |
| user_id | INTEGER | Candidate ID |
| resume_filename | TEXT | Uploaded file |
| scanned_experience_years | REAL | Years |
| scanned_skills_text | TEXT | CV skills |
| scanned_experience_text | TEXT | CV experience |
| scanned_education_text | TEXT | CV education |
| scanned_certificates_text | TEXT | CV certificates |
| scanned_projects_text | TEXT | CV projects |
| match_scores | REAL | Similarity results |
| interview_questions_json | TEXT | AI questions |
| interview_answers_json | TEXT | Candidate answers |
| interview_ai_score | REAL | Final score |
| interview_status | TEXT | pending/completed |
| status | TEXT | Application status |

---

# 📂 Project Structure

```
📁 ai-recruitment
 ├── app.py
 ├── db_setup.py
 ├── scanner.py
 ├── matcher.py
 ├── requirements.txt
 ├── .env
 ├── data/
 ├── resumes/
 ├── assets/
 └── README.md
```

---

# 📄 License (MIT)

See below for full license.

