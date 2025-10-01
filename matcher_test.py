import json
import pandas as pd
from cv_scanner import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import locale

sep = ";" if locale.getdefaultlocale()[0] in ["ar_SA", "fr_FR", "de_DE"] else ","
# Load resumes
resumes_df = pd.read_csv("data/cleaned_resumes.csv", sep=sep)

# Load JD JSON
with open("job_description.json", "r", encoding="utf-8") as f:
    jd = json.load(f)

# Clean JD sections
for section in jd:
    jd[section] = clean_text(jd[section])

# Example: Compute similarity for one resume
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

scores = []
for _, row in resumes_df.iterrows():
    resume_scores = {"resume_id": row["resume_id"], "filename": row["filename"]}
    for section in ["Skills", "Education", "Experience", "Projects", "Certificates / Courses"]:
        section_text = str(row.get(section, ""))
        jd_section_text = jd.get(section, "")
        sim = compute_similarity(section_text, jd_section_text) if section_text.strip() else 0.0
        resume_scores[section + "_match"] = round(sim, 3)
    
    section_scores = [resume_scores[s] for s in resume_scores if "_match" in s]
    resume_scores["overall_match"] = round(sum(section_scores)/len(section_scores), 3)
    scores.append(resume_scores)

results_df = pd.DataFrame(scores)
results_df.to_csv("data/matched_resumes.csv", index=False, sep=sep)
