import json
import pandas as pd
from scanner import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import locale
import re

# Set CSV separator based on locale
sep = ";" if locale.getdefaultlocale()[0] in ["ar_SA", "fr_FR", "de_DE"] else ","

# Load resumes
resumes_df = pd.read_csv("data/cleaned_resumes.csv", sep=sep)

# Load JD JSON
with open("job_description.json", "r", encoding="utf-8") as f:
    jd = json.load(f)

# 1. Extract and remove the YOE rule before cleaning
#    Use .pop() to get the value and delete it from the dict
required_years_rule = str(jd.pop("Years of Experience", "")).strip()

# Clean all other JD text sections
for section in jd:
    jd[section] = clean_text(jd[section])


# --- Helper: Word to digit mapping ---
NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
}

# --- Helper: New function to clean and translate the experience rule ---
def normalize_experience_rule(rule_string):
    """
    Converts 'one year' -> '1'
    Converts 'three to four years' -> '3-4'
    Converts 'FOUR YEARS' -> '4'
    """
    if not rule_string:
        return ""
        
    s = str(rule_string).lower() # force lowercase
    
    # 1. Remove 'years' / 'year'
    s = s.replace("years", "").replace("year", "") # "four years" -> "four "
    
    # 2. Convert "to" to "-" for range parsing
    # (handles "three to four" -> "three-four")
    s = re.sub(r'\s+to\s+', '-', s) 
    
    # 3. Replace words with numbers (e.g., 'three' -> '3')
    for word, number in NUMBER_WORDS.items():
        s = re.sub(r'\b' + word + r'\b', number, s)
        
    return s.strip() # " 4 " -> "4"



def check_experience_match(rule_string, resume_years):
    """
    Parses the JD experience rule and compares to candidate's years.
    """

    rule_string = normalize_experience_rule(rule_string)
    
    if not rule_string:
        # No rule specified by HR, 100% match.
        return 100.0 
        
    rule_string = str(rule_string).strip()

    # Scenario 1: Range (e.g., "5-7")
    range_match = re.match(r'^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\b', rule_string)
    if range_match:
        min_val = float(range_match.group(1))
        max_val = float(range_match.group(2))
        return 100.0 if min_val <= resume_years <= max_val else 0.0

    # Scenario 2: Minimum (e.g., "10+" or "+10")
    if '+' in rule_string:
        min_match = re.search(r'(\d+(?:\.\d+)?)', rule_string)
        if min_match:
            min_val = float(min_match.group(1))
            return 100.0 if resume_years >= min_val else 0.0

    # Scenario 3: Exact (e.g., "5")
    exact_match = re.match(r'^\s*(\d+(?:\.\d+)?)\b', rule_string)
    if exact_match:
        exact_val = float(exact_match.group(1))
        # "Exactly 5" means a perfect match
        return 100.0 if resume_years == exact_val else 0.0

    # No pattern recognized, default to 0.
    return 0.0


# Example: Compute similarity for one resume
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

scores = []
for _, row in resumes_df.iterrows():
    resume_scores = {"resume_id": row["resume_id"], "filename": row["filename"]}
    
    # 2. Calculate experience years match (100 or 0)
    resume_years = row.get("total_experience_years", 0) # from csv file
    experience_score = check_experience_match(required_years_rule, resume_years)
    resume_scores["Years_of_Experience_match"] = round(experience_score)

    # This list of base names is correct
    for section in ["Skills", "Education", "Experience", "Certificates / Courses"]:
        
        # We must look for the column name that scanner.py created, e.g., "cleaned_Skills"
        resume_column_name = f"cleaned_{section}"
        
        # Get text from the resume row using the *correct* column name
        section_text = str(row.get(resume_column_name, "")) 
        
        # Get text from the job description (this part was already correct)
        jd_section_text = jd.get(section, "")
        
        sim = compute_similarity(section_text, jd_section_text) if section_text.strip() else 0.0
        resume_scores[section + "_match"] = round(sim * 100)
    
        # Step 1: Calculations are numeric
    # Define the importance of each section
    weights = {
        "Years_of_Experience_match": 0.10,
        "Skills_match": 0.25,
        "Experience_match": 0.15,
        "Education_match": 0.20,
        #"Projects_match": 0.05,
        "Certificates / Courses_match": 0.30
    }
    
    overall_score = 0.0
    
    # Calculate the weighted score
    for key, weight in weights.items():
        # Get the numeric score directly from the dictionary
        # (It's a number like 75, not a string like "75%")
        # Default to 0.0 if the key doesn't exist for some reason
        numeric_score = resume_scores.get(key, 0.0) 
        
        # Multiply the score by its importance (e.g., 75.0 * 0.40)
        overall_score += float(numeric_score) * weight
        
    resume_scores["overall_match"] = round(overall_score)

    # Step 2: After calculation, convert numbers to formatted strings (add %)
    for key in resume_scores:
        if "_match" in key:
            resume_scores[key] = f"{resume_scores[key]}%"
            
            
    scores.append(resume_scores)

results_df = pd.DataFrame(scores)

# 3. (Important) Reorder columns + sort results
#    Ensures order matches request (ID, Filename, then scores)
column_order = [
    "resume_id", 
    "filename", 
    "Years_of_Experience_match", 
    "Skills_match", 
    "Education_match", 
    "Experience_match", 
    #"Projects_match", 
    "Certificates / Courses_match",
    "overall_match"
]


results_df = results_df.reindex(columns=column_order).dropna(axis=1, how='all')
results_df_sorted = results_df.sort_values(by="overall_match", ascending=False, key=lambda col: col.str.replace('%', '').astype(float))
results_df_sorted.to_csv("data/matched_resumes.csv", index=False, sep=sep)

print("Matching complete! Results saved to data/matched_resumes.csv")