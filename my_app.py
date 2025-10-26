import streamlit as st
import sqlite3
from passlib.hash import pbkdf2_sha256  # For password hashing
from streamlit_option_menu import option_menu
import os
import json
from openai import OpenAI
import time
import pandas as pd

# --- Import your project scripts ---
try:
    from scanner import (
        extract_text_from_pdf, 
        extract_sections, 
        clean_text, 
        calculate_experience_python,
        setup_nltk # <-- Add this import
    )
    from matcher import compute_similarity, check_experience_match, normalize_experience_rule
    
    # --- Run NLTK Setup ---
    # This will check and download 'punkt', 'stopwords', etc. if missing
    setup_nltk() 
    
except ImportError as e:
    st.error(f"Error importing helper scripts: {e}")
    st.error("Please make sure 'scanner.py' and 'matcher.py' are in the same directory.")

# --- OpenAI Client Setup ---
api_key = os.getenv("OPENAI_API_KEY") 

if not api_key:
    st.error("OPENAI_API_KEY environment variable not set. Please set it to use the AI interview feature.")
    # You can stop the app or disable AI features here
    client = None
else:
    client = OpenAI(api_key=api_key)

# Check if user is logged in, if not, set default values
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.session_state['role'] = None
    st.session_state['user_id'] = None
    st.session_state['apply_to_job_id'] = None
    st.session_state['active_interview_id'] = None

    # --- New variables for chat ---
    st.session_state['current_question_index'] = 0
    st.session_state['interview_answers'] = []
    st.session_state['current_interview_questions'] = []

# --- 1. Database Connection Setup ---
DB_FILE = "recruitment1.db"

def get_db_connection():
    """Creates a connection to the database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # To get results as dictionaries
    return conn

def stream_text(text, delay=0.07):
    """Typing animation for assistant messages."""
    placeholder = st.empty()
    output = ""
    for word in text.split():
        output += word + " "
        placeholder.markdown(output + "▌")
        time.sleep(delay)
    placeholder.markdown(output)

# --- 2. Database Helper Functions ---

def get_jobs_by_hr(hr_id):
    """Fetches all jobs posted by a specific HR user."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT job_id, title FROM Jobs WHERE hr_id = ? ORDER BY title", (hr_id,))
    jobs = c.fetchall()
    conn.close()
    return jobs

def get_applications_for_job(job_id):
    """Fetches all applications for a specific job, ranked."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT 
            A.application_id,
            U.username AS candidate_name,
            A.match_overall_score,
            A.interview_ai_score,
            A.status,
            A.interview_status
        FROM Applications A
        JOIN Users U ON A.candidate_id = U.user_id
        WHERE A.job_id = ?
        ORDER BY 
            A.interview_ai_score DESC, 
            A.match_overall_score DESC
    """, (job_id,))
    applications = c.fetchall()
    conn.close()
    return applications

def safe_compute_similarity(text1, text2):
    """
    A safe wrapper for compute_similarity that handles empty strings 
    to prevent TfidfVectorizer from crashing.
    """
    # 1. Clean up inputs
    text1_cleaned = str(text1).strip()
    text2_cleaned = str(text2).strip()

    # 2. Check for empty strings
    if not text1_cleaned and not text2_cleaned:
        # Both are empty. This is a 100% match (or 0%? Let's say 0% to be safe).
        # You can also return 1.0 (for 100%) if you consider "empty" a perfect match.
        return 0.0 
    
    if not text1_cleaned or not text2_cleaned:
        # One is empty and the other is not. This is a 0% match.
        return 0.0

    # 3. Both have content, now we can safely compute.
    try:
        # Call the original function from matcher.py
        return compute_similarity(text1_cleaned, text2_cleaned)
    except ValueError as e:
        # Catch "empty vocabulary" if they only contained stop words
        if 'empty vocabulary' in str(e):
            return 0.0
        else:
            raise e # Re-raise any other unexpected errors


def check_login(username, password):
    """Checks if username and password are correct. 
       Returns (True/False, user_role, user_id)
    """
    conn = get_db_connection()
    c = conn.cursor()
    # Fetch role AND user_id
    c.execute("SELECT user_id, password_hash, role FROM Users WHERE username = ?", (username,))
    user_data = c.fetchone()
    conn.close()

    if user_data:
        # User found, now check password
        hashed_password = user_data['password_hash']
        if verify_password(password, hashed_password):
            # Password is correct
            return True, user_data['role'], user_data['user_id']
    
    # User not found or password incorrect
    return False, None, None

def hash_password(password):
    """Hashes the password."""
    return pbkdf2_sha256.hash(password)

def verify_password(plain_password, hashed_password):
    """Verifies the password against the hash."""
    return pbkdf2_sha256.verify(plain_password, hashed_password)

def create_user(username, password, role):
    """Creates a new user in the database."""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        hashed_pass = hash_password(password)
        c.execute(
            "INSERT INTO Users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, hashed_pass, role)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        # This error occurs if the username is already taken (UNIQUE constraint)
        return False
    except Exception as e:
        print(f"Error creating user: {e}")
        return False


def create_job(hr_id, title, skills, experience, education, certificates, projects, years_rule):
    """Inserts a new job into the Jobs table."""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO Jobs (hr_id, title, jd_skills, jd_experience, jd_education, 
                              jd_certificates, jd_projects, jd_years_rule)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (hr_id, title, skills, experience, education, certificates, projects, years_rule)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating job: {e}")
        return False
    

def get_all_jobs():
    """Fetches all posted jobs from the database with HR username."""
    conn = get_db_connection()
    c = conn.cursor()
    # Join Jobs with Users to get the HR's username
    c.execute("""
        SELECT 
            Jobs.job_id, 
            Jobs.title, 
            Jobs.jd_skills, 
            Jobs.jd_experience, 
            Jobs.jd_education, 
            Jobs.jd_certificates, 
            Jobs.jd_projects, 
            Jobs.jd_years_rule,
            Users.username AS posted_by_hr
        FROM Jobs
        JOIN Users ON Jobs.hr_id = Users.user_id
        ORDER BY Jobs.job_id DESC
    """)
    jobs = c.fetchall()
    conn.close()
    return jobs


def get_job_details(job_id):
    """Fetches all details for a single job."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM Jobs WHERE job_id = ?", (job_id,))
    job_details = c.fetchone()
    conn.close()
    return job_details

def create_application(app_data):
    """
    Saves the complete application (scanned data + match scores) to the database.
    'app_data' is a dictionary containing all the column names and values.
    """
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Create dynamic query
        columns = ', '.join(app_data.keys())
        placeholders = ', '.join('?' * len(app_data))
        query = f"INSERT INTO Applications ({columns}) VALUES ({placeholders})"
        
        c.execute(query, tuple(app_data.values()))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating application: {e}")
        return False

def get_pending_interviews(candidate_id):
    """Fetches applications that are ready for an interview."""
    conn = get_db_connection()
    c = conn.cursor()
    # Fetches jobs the candidate applied to, qualified for (>10%), and haven't interviewed for
    c.execute("""
        SELECT 
            A.application_id, 
            J.title AS job_title
        FROM Applications A
        JOIN Jobs J ON A.job_id = J.job_id
        WHERE A.candidate_id = ? 
          AND A.match_overall_score > 10
          AND A.interview_status = 'pending'
    """, (candidate_id,))
    interviews = c.fetchall()
    conn.close()
    return interviews

def get_full_application_details(application_id):
    """Fetches all CV and JD text for a specific application."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT 
            A.*, 
            J.title, J.jd_skills, J.jd_experience, J.jd_education, 
            J.jd_certificates, J.jd_projects
        FROM Applications A
        JOIN Jobs J ON A.job_id = J.job_id
        WHERE A.application_id = ?
    """, (application_id,))
    details = c.fetchone()
    conn.close()
    return details

def save_interview_questions(application_id, questions_json_string):
    """Saves the generated questions to the database."""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "UPDATE Applications SET interview_questions_json = ? WHERE application_id = ?",
            (questions_json_string, application_id)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving questions: {e}")
        return False

def save_interview_results(application_id, ai_score, status):
    """Saves the final AI score and updates the interview status."""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "UPDATE Applications SET interview_ai_score = ?, interview_status = ? WHERE application_id = ?",
            (ai_score, status, application_id)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

# --- 3. AI Helper Functions ---

def generate_interview_questions(cv_text_dict, jd_text_dict):
    """Generates interview questions using GPT-4o-mini."""
    if not client:
        return None # Return None if client isn't initialized

    # Combine the dictionaries into a formatted string
    cv_prompt_text = "\n".join([f"CV {key}: {value}" for key, value in cv_text_dict.items() if value])
    jd_prompt_text = "\n".join([f"JD {key}: {value}" for key, value in jd_text_dict.items() if value])

    prompt = f"""
    You are an expert HR interviewer. Based on the candidate's CV and the Job Description provided, generate exactly 10 unique interview questions.
    - 5 questions must be about their CV.
    - 5 questions must be about their fit for the job.

    Return the 10 questions as a single JSON list of strings. Do not add any preamble, introduction, or text other than the JSON list itself.

    ---CANDIDATE CV---
    {cv_prompt_text}

    ---JOB DESCRIPTION---
    {jd_prompt_text}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"}
        )
        questions_json = response.choices[0].message.content
        # The prompt asks for a list, but gpt-4o-mini might return {"questions": [...]}. 
        # We need to parse it robustly.
        data = json.loads(questions_json)
        
        # Handle if it returns {"questions": [...]} or just [...]
        if isinstance(data, dict):
            questions_list = data.get("questions", [])
        elif isinstance(data, list):
            questions_list = data
        else:
            questions_list = []
            
        return questions_list
        
    except Exception as e:
        print(f"Error generating questions: {e}")
        return None

def grade_interview_answers(cv_text_dict, jd_text_dict, questions, answers):
    """Grades the interview answers using GPT-4o-mini."""
    if not client:
        return None

    cv_prompt_text = "\n".join([f"CV {key}: {value}" for key, value in cv_text_dict.items() if value])
    jd_prompt_text = "\n".join([f"JD {key}: {value}" for key, value in jd_text_dict.items() if value])
    
    qa_pairs = "\n".join([f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(zip(questions, answers))])

    prompt = f"""
    You are an expert HR manager. A candidate is applying for a job (JD provided) and submitted a CV (provided).
    The candidate provided the following answers to 10 interview questions.

    Please grade the candidate's suitability based *only* on the quality of their answers and their relevance to the CV and Job Description. 
    Provide a total interview score from 0 to 100.
    
    Return *only the integer number* and nothing else (e.g., "85").

    ---CANDIDATE CV---
    {cv_prompt_text}

    ---JOB DESCRIPTION---
    {jd_prompt_text}
    
    ---INTERVIEW Q&A---
    {qa_pairs}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        score_text = response.choices[0].message.content
        # Find the first number in the response
        import re
        match = re.search(r'\d+', score_text)
        if match:
            return int(match.group(0))
        else:
            return 0 # Default score if parsing fails
            
    except Exception as e:
        print(f"Error grading answers: {e}")
        return 0 # Default score on error
    

# --- 4. UI Setup ---
st.set_page_config(page_title="AI Recruitment System", layout="centered")

# --- 5. Sidebar Navigation ---
with st.sidebar:
    # If user is logged in
    if st.session_state['logged_in']:
        # Show different menu based on role
        if st.session_state['role'] == 'hr':
            selected_page = option_menu(
                menu_title=f"Welcome, {st.session_state['username']}",
                options=["Post a Job", "View Applications", "Account"],
                icons=["plus-square", "list-task", "person"],
                menu_icon="briefcase"
            )
        else: # Role is 'candidate'
             selected_page = option_menu(
                menu_title=f"Welcome, {st.session_state['username']}",
                options=["Find Jobs", "My Applications", "Account"],
                icons=["search", "files", "person"],
                menu_icon="person"
            )
    
    # If user is NOT logged in
    else:
        selected_page = option_menu(
            menu_title="Main Menu",
            options=["Login", "Sign Up"],
            icons=["box-arrow-in-right", "person-plus-fill"],
            menu_icon="cast",
            default_index=0,
        )

# --- 6. Page Logic ---

if selected_page == "Post a Job":
    st.header("Post a New Job")

    # Check if user is an HR
    if st.session_state.get('role') != 'hr':
        st.error("You do not have permission to view this page.")
    else:
        with st.form("post_job_form", clear_on_submit=True):
            st.write("Fill in the job description details:")
            
            job_title = st.text_input("Job Title")
            job_years = st.text_input("Years of Experience Rule (e.g., '3-5', '5+', '2')")
            
            # Use text_area for larger inputs
            job_skills = st.text_area("Required Skills (one per line or comma-separated)")
            job_exp = st.text_area("Required Experience Description")
            job_edu = st.text_area("Required Education")
            job_certs = st.text_area("Required Certificates / Courses")
            job_projects = st.text_area("Required Projects")
            
            submit_job = st.form_submit_button("Post Job")

        if submit_job:
            if not job_title or not job_years or not job_skills:
                st.warning("Please fill in at least Title, Years, and Skills.")
            else:
                # Get the hr_id from the session state
                hr_user_id = st.session_state['user_id']
                
                success = create_job(
                    hr_id=hr_user_id,
                    title=job_title,
                    years_rule=job_years,
                    skills=job_skills,
                    experience=job_exp,
                    education=job_edu,
                    certificates=job_certs,
                    projects=job_projects
                )
                
                if success:
                    st.success(f"Job '{job_title}' has been posted successfully!")
                else:
                    st.error("An error occurred while posting the job.")


elif selected_page == "Sign Up":
    st.header("Create a New Account")
    
    with st.form("signup_form"):
        st.write("Please fill in the details below:")
        
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        role = st.selectbox("Select Account Type:", ["candidate", "hr"])
        
        submitted = st.form_submit_button("Create Account")

    if submitted:
        if not new_username or not new_password or not confirm_password:
            st.warning("Please fill in all fields.")
        elif new_password != confirm_password:
            st.error("Passwords do not match!")
        else:
            # Try to create the user
            success = create_user(new_username, new_password, role)
            if success:
                st.success(f"Account created successfully for {new_username} as {role}.")
                st.info("You can now log in from the 'Login' page.")
            else:
                st.error("This username is already taken. Please choose another one.")





elif selected_page == "Login":
    st.header("Login")

    # If user is already logged in, show a different message
    if st.session_state['logged_in']:
        st.success(f"You are already logged in as {st.session_state['username']} ({st.session_state['role']}).")
        if st.button("Log Out"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = None
            st.session_state['role'] = None
            st.rerun() # Refresh the page
    else:
        # Show login form if not logged in
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            login_submitted = st.form_submit_button("Login")

        if login_submitted:
            if not username or not password:
                st.warning("Please enter both username and password.")
            else:
                # Check login credentials
                # Note: We now capture user_id as well
                is_correct, user_role, user_id = check_login(username, password) 
                
                if is_correct:
                    # Save login state
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.session_state['role'] = user_role
                    st.session_state['user_id'] = user_id # <-- Add this line
                    st.success(f"Welcome, {username}!")
                    st.info(f"Your role is: {user_role}")
                    st.rerun() # Refresh the page to reflect login
                else:
                    st.error("Invalid username or password.")


elif selected_page == "Find Jobs":
    st.header("Find Jobs")

    # Check if user is a candidate
    if st.session_state.get('role') != 'candidate':
        st.error("You do not have permission to view this page.")
        
    # === Part 2: Show CV Upload Form ===
    # Check if the user has clicked "Apply" on a job
    elif 'apply_to_job_id' in st.session_state and st.session_state['apply_to_job_id'] is not None:
        
        job_id_to_apply = st.session_state['apply_to_job_id']
        st.subheader(f"Applying for Job (ID: {job_id_to_apply})")
        st.write("Please upload your CV (PDF only) to proceed.")
        
        uploaded_file = st.file_uploader("Upload your CV", type=["pdf"])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Application"):
                if uploaded_file is not None:
                    # Show a spinner while processing
                    with st.spinner("Analyzing your CV... This may take a moment."):
                        try:
                            # --- 1. Setup paths and save temp file ---
                            # Create a 'temp' folder if it doesn't exist
                            temp_dir = os.path.join(os.getcwd(), "temp")
                            if not os.path.exists(temp_dir):
                                os.makedirs(temp_dir)
                            
                            # Save the uploaded PDF to the temp folder
                            temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(temp_pdf_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # --- 2. Get Job Details (JD) from DB ---
                            job_id = st.session_state['apply_to_job_id']
                            job_details = get_job_details(job_id)
                            
                            # Prepare JD text (clean it for matching)
                            jd_cleaned = {
                                "Skills": clean_text(job_details['jd_skills']),
                                "Experience": clean_text(job_details['jd_experience']),
                                "Education": clean_text(job_details['jd_education']),
                                "Certificates": clean_text(job_details['jd_certificates']),
                                "Projects": clean_text(job_details['jd_projects'])
                            }
                            jd_years_rule = job_details['jd_years_rule']

                            # --- 3. Run SCANNER functions ---
                            raw_text = extract_text_from_pdf(temp_pdf_path)
                            scanned_sections = extract_sections(raw_text)
                            scanned_exp_years = calculate_experience_python(scanned_sections.get('Experience', ''))
                            
                            # Prepare scanned text (clean it for matching)
                            scanned_cleaned = {
                                "Skills": clean_text(scanned_sections.get('Skills', '')),
                                "Experience": clean_text(scanned_sections.get('Experience', '')),
                                "Education": clean_text(scanned_sections.get('Education', '')),
                                "Certificates": clean_text(scanned_sections.get('Certificates / Courses', '')),
                                "Projects": clean_text(scanned_sections.get('Projects', ''))
                            }
                            
                            # --- 4. Run MATCHER functions ---
                            # A. Experience Years Score
                            match_years_score = check_experience_match(jd_years_rule, scanned_exp_years)
                            
                            # B. Section Similarity Scores (TF-IDF)
                           # --- This is your NEW, fixed code ---
                            match_skills_score = safe_compute_similarity(jd_cleaned["Skills"], scanned_cleaned["Skills"]) * 100
                            match_exp_score = safe_compute_similarity(jd_cleaned["Experience"], scanned_cleaned["Experience"]) * 100
                            match_edu_score = safe_compute_similarity(jd_cleaned["Education"], scanned_cleaned["Education"]) * 100
                            match_certs_score = safe_compute_similarity(jd_cleaned["Certificates"], scanned_cleaned["Certificates"]) * 100
                            match_projects_score = safe_compute_similarity(jd_cleaned["Projects"], scanned_cleaned["Projects"]) * 100
                            
                            # C. Calculate Overall Score (using weights from your matcher.py)
                            weights = {
                                "Years_of_Experience_match": 0.10,
                                "Skills_match": 0.25,
                                "Experience_match": 0.15,
                                "Education_match": 0.20,
                                "Certificates / Courses_match": 0.30,
                                "Projects_match": 0.00 # Set projects weight to 0 for now
                            }
                            
                            # Quick fix for matcher keys (Certificates / Courses vs Certificates)
                            scores = {
                                "Years_of_Experience_match": match_years_score,
                                "Skills_match": match_skills_score,
                                "Experience_match": match_exp_score,
                                "Education_match": match_edu_score,
                                "Certificates / Courses_match": match_certs_score,
                                "Projects_match": match_projects_score
                            }
                            
                            overall_score = sum(scores[key] * weight for key, weight in weights.items())
                            
                            # --- 5. Prepare data for Database ---
                            application_data_to_save = {
                                "job_id": job_id,
                                "candidate_id": st.session_state['user_id'],
                                "status": 'applied',
                                "resume_filename": uploaded_file.name,
                                
                                "scanned_experience_years": scanned_exp_years,
                                "scanned_skills_text": scanned_cleaned["Skills"],
                                "scanned_experience_text": scanned_cleaned["Experience"],
                                "scanned_education_text": scanned_cleaned["Education"],
                                "scanned_certificates_text": scanned_cleaned["Certificates"],
                                "scanned_projects_text": scanned_cleaned["Projects"],
                                
                                "match_overall_score": round(overall_score),
                                "match_skills_score": round(match_skills_score),
                                "match_experience_score": round(match_exp_score),
                                "match_education_score": round(match_edu_score),
                                "match_certificates_score": round(match_certs_score),
                                "match_projects_score": round(match_projects_score),
                                "match_years_score": round(match_years_score),
                                
                                "interview_status": 'pending' # Ready for interview
                            }

                            # --- 6. Save to Database ---
                            save_success = create_application(application_data_to_save)
                            
                            if save_success:
                                st.success("Application Submitted Successfully!")
                                st.balloons()
                                
                                # Check if they qualify for the interview
                                if overall_score > 10:
                                    st.info("Congratulations! You have qualified for the next step: The AI Interview.")
                                    # We will build this 'AI Interview' page next
                                    st.session_state['interview_pending'] = True # Set flag for next step
                                else:
                                    st.warning("Your application has been received. A recruiter will review it.")
                                    
                                # Reset state and clean up
                                os.remove(temp_pdf_path) # Delete the temp file
                                st.session_state['apply_to_job_id'] = None
                                # Use st.experimental_rerun() for older Streamlit, st.rerun() for new
                                st.rerun() 
                                
                            else:
                                st.error("Failed to save application to database.")
                                os.remove(temp_pdf_path)

                        except Exception as e:
                            st.error(f"An error occurred during processing: {e}")
                            # Clean up if file was created
                            if os.path.exists(temp_pdf_path):
                                os.remove(temp_pdf_path)

                else:
                    st.warning("Please upload a PDF file first.")
        with col2:
            if st.button("Cancel"):
                st.session_state['apply_to_job_id'] = None
                st.rerun()

    # === Part 1: Show Job List (Default View) ===
    # This is the default view if not applying
    else:
        all_jobs = get_all_jobs()
        
        if not all_jobs:
            st.info("No jobs are currently posted. Please check back later.")
        else:
            st.write(f"Found {len(all_jobs)} available jobs:")
            
            # Loop through each job and display it
            for job in all_jobs:
                # Use st.expander for each job post
                with st.expander(f"**{job['title']}** (Posted by: {job['posted_by_hr']})"):
                    st.markdown(f"**Required Years of Experience:** `{job['jd_years_rule']}`")
                    
                    st.subheader("Skills")
                    st.markdown(f"```\n{job['jd_skills']}\n```")

                    st.subheader("Experience Details")
                    st.markdown(job['jd_experience'] if job['jd_experience'] else "N/A")
                    
                    st.subheader("Education")
                    st.markdown(job['jd_education'] if job['jd_education'] else "N/A")
                    
                    st.subheader("Certificates / Courses")
                    st.markdown(job['jd_certificates'] if job['jd_certificates'] else "N/A")

                    st.subheader("Projects")
                    st.markdown(job['jd_projects'] if job['jd_projects'] else "N/A")
                    
                    # Apply Button
                    # We use job['job_id'] to make the button key unique
                    if st.button("Apply for this Job", key=f"apply_{job['job_id']}"):
                        # When clicked, set the job ID and rerun
                        st.session_state['apply_to_job_id'] = job['job_id']
                        st.rerun()


elif selected_page == "My Applications":
    st.header("My Applications")
    
    if st.session_state.get('role') != 'candidate':
        st.error("You do not have permission to view this page.")

    # === STATE 2: An Interview is Active ===
    elif st.session_state['active_interview_id'] is not None:
        
        app_id = st.session_state['active_interview_id']
        
        # --- 1. Load or Generate Questions (Run Once) ---
        if not st.session_state['current_interview_questions']:
            with st.spinner("Preparing your interview..."):
                app_details = get_full_application_details(app_id)
                
                if app_details['interview_questions_json']:
                    # Questions already exist, load them
                    questions = json.loads(app_details['interview_questions_json'])
                else:
                    # First time: Generate questions
                    cv_text_dict = {
                        "Skills": app_details['scanned_skills_text'],
                        "Experience": app_details['scanned_experience_text'],
                        "Education": app_details['scanned_education_text'],
                        "Certificates": app_details['scanned_certificates_text'],
                        "Projects": app_details['scanned_projects_text']
                    }
                    jd_text_dict = {
                        "Skills": app_details['jd_skills'],
                        "Experience": app_details['jd_experience'],
                        "Education": app_details['jd_education'],
                        "Certificates": app_details['jd_certificates'],
                        "Projects": app_details['jd_projects']
                    }
                    
                    questions = generate_interview_questions(cv_text_dict, jd_text_dict)
                    
                    if questions and len(questions) == 10:
                        save_interview_questions(app_id, json.dumps(questions))
                    else:
                        st.error("Failed to generate interview questions. Please try again later.")
                        st.session_state['active_interview_id'] = None
                        st.rerun() # Exit this state
                
                # Save questions to session state for quick access
                st.session_state['current_interview_questions'] = questions

        # --- 2. Display Chat Interface ---
        st.subheader("AI Interview (In Progress)")
        st.write("Please answer the questions one by one. There are 10 questions in total.")

        # Show full chat history
        for i, answer in enumerate(st.session_state['interview_answers']):
            with st.chat_message("assistant", avatar="🤖"):
                st.write(st.session_state['current_interview_questions'][i])
            with st.chat_message("user", avatar="👤"):
                st.write(answer)

        current_index = st.session_state['current_question_index']

        # If interview still has questions
        if current_index < len(st.session_state['current_interview_questions']):
            current_question = st.session_state['current_interview_questions'][current_index]

            # ✅ Show question with typing animation only on first appearance
            if st.session_state.get("last_displayed_index") != current_index:
                with st.chat_message("assistant", avatar="🤖"):
                    stream_text(current_question)
                st.session_state["last_displayed_index"] = current_index
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(current_question)

            # Get user's answer
            user_answer = st.chat_input("Your answer...")
            if user_answer:
                # Show user's message immediately
                with st.chat_message("user", avatar="👤"):
                    st.write(user_answer)

                # Save answer
                st.session_state['interview_answers'].append(user_answer)
                st.session_state['current_question_index'] += 1

                # Small delay before rerun to simulate natural flow
                time.sleep(0.3)
                st.rerun()

        # If interview is done
        else:
            st.success("You have completed all questions!")
            if st.button("Submit Interview for Grading"):
                with st.spinner("Submitting and grading your answers..."):
                    app_details = get_full_application_details(app_id)

                    cv_text_dict = {
                        "Skills": app_details['scanned_skills_text'],
                        "Experience": app_details['scanned_experience_text'],
                        "Education": app_details['scanned_education_text'],
                        "Certificates": app_details['scanned_certificates_text'],
                        "Projects": app_details['scanned_projects_text']
                    }
                    jd_text_dict = {
                        "Skills": app_details['jd_skills'],
                        "Experience": app_details['jd_experience'],
                        "Education": app_details['jd_education'],
                        "Certificates": app_details['jd_certificates'],
                        "Projects": app_details['jd_projects']
                    }

                    final_score = grade_interview_answers(
                        cv_text_dict,
                        jd_text_dict,
                        st.session_state['current_interview_questions'],
                        st.session_state['interview_answers']
                    )

                    if save_interview_results(app_id, final_score, 'completed'):
                        st.success(f"Interview Submitted! Your AI-graded score is: {final_score}")
                        st.info("A recruiter will review your complete application.")
                        st.balloons()

                        # Reset state
                        st.session_state['active_interview_id'] = None
                        st.session_state['current_question_index'] = 0
                        st.session_state['interview_answers'] = []
                        st.session_state['current_interview_questions'] = []
                        st.session_state['last_displayed_index'] = None
                        st.rerun()
                    else:
                        st.error("Failed to save your interview results. Please contact support.")



    # === STATE 1: Default View (List of Applications) ===
    else:
        pending_interviews = get_pending_interviews(st.session_state['user_id'])
        
        if not pending_interviews:
            st.info("You have no pending interviews. Apply for a job to get started!")
        else:
            st.subheader("You have qualified for an interview for the following jobs:")
            for interview in pending_interviews:
                st.markdown(f"**Job:** {interview['job_title']}")
                if st.button("Start AI Interview", key=f"start_{interview['application_id']}"):
                    # --- Reset state variables before starting ---
                    st.session_state['active_interview_id'] = interview['application_id']
                    st.session_state['current_question_index'] = 0
                    st.session_state['interview_answers'] = []
                    st.session_state['current_interview_questions'] = []
                    st.rerun()
        


elif selected_page == "View Applications":
    st.header("View Job Applications")
    
    if st.session_state.get('role') != 'hr':
        st.error("You do not have permission to view this page.")
    else:
        # --- 1. Select a Job ---
        hr_jobs = get_jobs_by_hr(st.session_state['user_id'])
        
        if not hr_jobs:
            st.info("You have not posted any jobs yet.")
        else:
            # Create a dictionary to map job titles to job_ids
            job_map = {job['title']: job['job_id'] for job in hr_jobs}
            job_titles = list(job_map.keys())
            
            selected_title = st.selectbox("Select a job to view its applications:", job_titles)
            
            if selected_title:
                selected_job_id = job_map[selected_title]
                
                # --- 2. Display Leaderboard ---
                st.subheader(f"Leaderboard for: {selected_title}")
                applications = get_applications_for_job(selected_job_id)
                
                if not applications:
                    st.info("No applications submitted for this job yet.")
                else:
                    # Convert list of Row objects to list of dictionaries
                    applications_list = [dict(row) for row in applications]
                    # Now, create the DataFrame from the list of dicts
                    df = pd.DataFrame(applications_list)
                    # Reorder and rename columns for display
                    df_display = df[['candidate_name', 'match_overall_score', 'interview_ai_score', 'interview_status', 'application_id']]
                    df_display = df_display.rename(columns={
                        'candidate_name': 'Candidate',
                        'match_overall_score': 'CV Match %',
                        'interview_ai_score': 'Interview Score',
                        'interview_status': 'Interview Status'
                    })
                    
                    st.dataframe(df_display, use_container_width=True, hide_index=True)

                    # --- 3. Select Candidate for Detail View ---
                    st.subheader("View Application Details")
                    
                    # Create a map of candidate names to their application_id
                    app_map = {f"{app['candidate_name']} (ID: {app['application_id']})": app['application_id'] for app in applications}
                    app_names = list(app_map.keys())
                    
                    selected_app_name = st.selectbox("Select an applicant to see full details:", app_names)
                    
                    if selected_app_name:
                        selected_app_id = app_map[selected_app_name]
                        
                        # Fetch all details
                        details = get_full_application_details(selected_app_id)
                        
                        st.markdown(f"#### Details for Application {selected_app_id}")
                        
                        # Scores in columns
                        col1, col2, col3 = st.columns(3)
                        col1.metric("CV Match Score", f"{details['match_overall_score']}%")
                        col2.metric("AI Interview Score", details['interview_ai_score'] or "N/A")
                        col3.metric("Experience Years", f"{details['scanned_experience_years']} yrs")
                        
                        # Use expanders for text sections
                        with st.expander("CV Match Score Breakdown"):
                            st.json({
                                "Skills Match": f"{details['match_skills_score']}%",
                                "Experience Match": f"{details['match_experience_score']}%",
                                "Education Match": f"{details['match_education_score']}%",
                                "Certificates Match": f"{details['match_certificates_score']}%",
                                "Projects Match": f"{details['match_projects_score']}%",
                                "Years of Exp. Match": f"{details['match_years_score']}%"
                            })

                        with st.expander("Interview Questions and Answers"):
                            if details['interview_questions_json']:
                                questions = json.loads(details['interview_questions_json'])
                                # Answers are stored in our session state (oops, not on reload)
                                # We need to add answers to the database!
                                # For now, let's just show the questions.
                                st.write("Questions generated by AI:")
                                st.json(questions)
                                st.warning("Note: Displaying candidate answers requires a database update.")
                            else:
                                st.info("Interview not yet started or questions not saved.")
                        
                        with st.expander("Scanned CV Text (Cleaned)"):
                            st.text_area("Skills", details['scanned_skills_text'], height=100)
                            st.text_area("Experience", details['scanned_experience_text'], height=150)
                            st.text_area("Education", details['scanned_education_text'], height=100)


# --- Keep the placeholder for the remaining page ---
elif selected_page == "Account":
    st.header(selected_page)
    st.info("This page is under construction.")