import json
import pandas as pd
from scanner import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import locale
import re

sep = ";" if locale.getdefaultlocale()[0] in ["ar_SA", "fr_FR", "de_DE"] else ","
# Load resumes
resumes_df = pd.read_csv("data/cleaned_resumes.csv", sep=sep)

# Load JD JSON
with open("job_description.json", "r", encoding="utf-8") as f:
    jd = json.load(f)

# 1. استخرج سنوات الخبرة المطلوبة كـ "رقم"
#    استخدم .pop() لأخذ القيمة وحذفها من القاموس، حتى لا يحاول تنظيفها كنص
required_years_rule = str(jd.pop("Years of Experience", "")).strip()

# Clean JD sections
for section in jd:
    jd[section] = clean_text(jd[section])



# --- إضافة ---: قاموس لترجمة الكلمات إلى أرقام
NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
}

# --- إضافة ---: دالة جديدة لتنظيف وترجمة قاعدة الخبرة
def normalize_experience_rule(rule_string):
    """
    يحول 'one year' إلى '1' 
    ويحول 'three to four years' إلى '3-4'
    ويحول 'FOUR YEARS' إلى '4'
    """
    if not rule_string:
        return ""
        
    s = str(rule_string).lower() # تحويل "FOUR YEARS" إلى "four years"
    
    # 1. إزالة الكلمات غير المهمة
    s = s.replace("years", "").replace("year", "") # "four years" -> "four "
    
    # 2. تحويل "to" إلى "-" للتعرف على النطاق
    # (يعالج "three to four" ويحولها إلى "three-four")
    s = re.sub(r'\s+to\s+', '-', s) 
    
    # 3. استبدال الكلمات بالأرقام
    # (يعالج 'three' ويحولها إلى '3' باستخدام القاموس)
    for word, number in NUMBER_WORDS.items():
        s = re.sub(r'\b' + word + r'\b', number, s)
        
    return s.strip() # " 4 " -> "4"



def check_experience_match(rule_string, resume_years):
    """
    يحلل قاعدة الخبرة من ملف JSON ويقارنها بسنوات خبرة المرشح.
    """

    rule_string = normalize_experience_rule(rule_string)
    
    if not rule_string:
        # إذا لم يضع HR أي شرط، نعتبره مطابقاً بنسبة 100
        return 100.0 
        
    rule_string = str(rule_string).strip()

    # السيناريو 1: نطاق (مثل "5-7")
    range_match = re.match(r'^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\b', rule_string)
    if range_match:
        min_val = float(range_match.group(1))
        max_val = float(range_match.group(2))
        return 100.0 if min_val <= resume_years <= max_val else 0.0

    # السيناريو 2: حد أدنى (مثل "+10" أو "10+")
    if '+' in rule_string:
        min_match = re.search(r'(\d+(?:\.\d+)?)', rule_string)
        if min_match:
            min_val = float(min_match.group(1))
            return 100.0 if resume_years >= min_val else 0.0

    # السيناريو 3: بالضبط (مثل "5" أو "5 years")
    exact_match = re.match(r'^\s*(\d+(?:\.\d+)?)\b', rule_string)
    if exact_match:
        exact_val = float(exact_match.group(1))
        # "بالضبط 5 ليس اقل او اكثر" - هذا يعني مطابقة تامة
        return 100.0 if resume_years == exact_val else 0.0

    # إذا لم يتم التعرف على أي نمط، نعتبره غير مطابق
    return 0.0


# Example: Compute similarity for one resume
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

scores = []
for _, row in resumes_df.iterrows():
    resume_scores = {"resume_id": row["resume_id"], "filename": row["filename"]}
    
    # 2. حساب درجة سنوات الخبرة (100 أو 0)
    resume_years = row.get("total_experience_years", 0) # من ملف csv
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
    
        # الخطوة 1: الحسابات تتم كأرقام عادية
    # Define the importance of each section (must add up to 1.0)
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

    # الخطوة 2: بعد انتهاء الحسابات، نقوم بتحويل الأرقام إلى نص مع إضافة علامة %
    for key in resume_scores:
        if "_match" in key:
            resume_scores[key] = f"{resume_scores[key]}%"
            
            
    scores.append(resume_scores)

results_df = pd.DataFrame(scores)

# 3. (مهم) ترتيب الأعمدة + فرز النتائج
#    هذا يضمن أن الترتيب يطابق طلبك (ID، Filename، ثم درجات المطابقة)
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