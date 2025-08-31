# matcher.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# # Load resumes
# resumes_df = pd.read_csv("./data/cleaned_resumes.csv")

# # Load job description
# with open("./data/job_description.txt", "r", encoding="utf-8") as file:
#     job_text = file.read()

# # Combine job + each resume to compute similarity
# texts = [job_text] + resumes_df["text"].tolist()

# # Convert text to vectors using TF-IDF
# vectorizer = TfidfVectorizer(stop_words="english")
# tfidf_matrix = vectorizer.fit_transform(texts)

# # Compute cosine similarity between job and each resume
# cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# # Add score to DataFrame
# resumes_df["match_score"] = (cosine_sim * 100).round(2)

# # Sort by best matches
# resumes_df = resumes_df.sort_values(by="match_score", ascending=False)

# # Save results
# resumes_df.to_csv("./data/resume_match_results.csv", index=False)
# print(resumes_df[["filename", "match_score"]].head(30))

# matcher.py
####################################################################################
####################################################################################
####################################################################################
####################################################################################

# def load_job_description(path="data/job_description.txt"):
#     with open(path, "r", encoding="utf-8") as f:
#         return f.read()

# def load_resumes(path="data/cleaned_resumes.csv"):
#     return pd.read_csv(path)

# def compute_similarity(resumes_df, job_description):
#     texts = resumes_df["text_preprocessed"].tolist()
#     all_texts = [job_description] + texts  # [JD, resume1, resume2...]

#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(all_texts)

#     job_vec = tfidf_matrix[0]           # JD vector
#     resume_vecs = tfidf_matrix[1:]      # Resume vectors

#     scores = cosine_similarity(job_vec, resume_vecs)[0]  # 1 x N

#     resumes_df["match_score"] = scores
#     return resumes_df.sort_values(by="match_score", ascending=False)

# def save_results(df, path="data/resume_match_results.csv"):
#     df[["resume_id", "filename", "match_score"]].to_csv(path, index=False)

# if __name__ == "__main__":
#     jd = load_job_description()
#     resumes = load_resumes()
#     matched = compute_similarity(resumes, jd)
#     save_results(matched)
#     print(matched[["resume_id", "match_score"]].head(5))


####################################################################################
####################################################################################
####################################################################################
####################################################################################

def extract_jd_sections(filepath="data/job_description.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().lower()

    def extract_section(name):
        match = re.search(rf'\[{name}\]\s*(.*?)\n(?:\[|$)', text + "\n[END]", re.DOTALL)
        return match.group(1).strip() if match else ""

    return {
        "skills": extract_section("skills required"),
        "education": extract_section("education required"),
        "experience": extract_section("experience required"),
        "projects": extract_section("projects"),
        "job_title": extract_section("job title")
    }

def compute_similarity(a, b):
    if pd.isna(a) or pd.isna(b) or not a.strip() or not b.strip():
        return 0.0
    tfidf = TfidfVectorizer().fit_transform([a, b])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def match_resumes(resumes_df, jd):
    scores = []
    for _, row in resumes_df.iterrows():
        skill_score = compute_similarity(row["skills"], jd["skills"])
        edu_score = compute_similarity(row["education"], jd["education"])
        exp_score = compute_similarity(row["experience"], jd["experience"])
        proj_score = compute_similarity(row["projects"], jd["projects"])
        title_score = compute_similarity(row["job_title"], jd["job_title"])

        final = 0.2 * skill_score + 0.4 * edu_score + 0.2 * exp_score + 0.1 * proj_score + 0.1 * title_score

        scores.append({
            "resume_id": row["resume_id"],
            "filename": row["filename"],
            "match_score": round(final, 3)
        })
    return pd.DataFrame(scores)

if __name__ == "__main__":
    resumes = pd.read_csv("data/cleaned_resumes.csv")
    jd = extract_jd_sections()
    matched = match_resumes(resumes, jd)
    matched.to_csv("data/resume_match_results.csv", index=False)
    print(matched.sort_values(by="match_score", ascending=False).head())