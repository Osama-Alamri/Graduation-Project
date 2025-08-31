import os
import pdfplumber
import pandas as pd
import re 

file_path = "./resumes"

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def clean_text(text):
    # Simple cleaning
    return text.replace('\n', ' ').replace('\xa0', ' ').strip()


def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()



# def process_resumes(folder_path):
#     resume_data = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".pdf"):
#             text = extract_text_from_pdf(os.path.join(folder_path, filename))
#         else:
#             continue

#         cleaned = clean_text(text)
#         preprocessed = preprocess_text(cleaned)
#         resume_data.append({
#             "resume_id": filename.split(".")[0],
#             "filename": filename,
#             "text": cleaned
#         })

#     return pd.DataFrame(resume_data)

def extract_sections(text):
    text = text.lower()

    def extract_section(keyword):
        match = re.search(rf'{keyword}[:\-]*\s*(.*)', text)
        return match.group(1) if match else ""

    return {
        "skills": extract_section("skills"),
        "education": extract_section("education"),
        "experience": extract_section("experience"),
        "projects": extract_section("projects"),
        "job_title": extract_section("title|position")
    }

def process_resumes(folder_path):
    resume_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(folder_path, filename))
            cleaned = clean_text(text)
            sections = extract_sections(cleaned)
            resume_data.append({
                "resume_id": filename.split(".")[0],
                "filename": filename,
                "skills": sections["skills"],
                "education": sections["education"],
                "experience": sections["experience"],
                "projects": sections["projects"],
                "job_title": sections["job_title"]
            })
    return pd.DataFrame(resume_data)



if __name__ == "__main__":
    df = process_resumes(file_path)
    df.to_csv("data/cleaned_resumes.csv", index=False)
    print(f"Processed: {"cleaned_resumes"}")