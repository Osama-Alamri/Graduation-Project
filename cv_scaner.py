import os
import pdfplumber
import pandas as pd
import re

# The path to the folder containing your resume PDF files
file_path = "./resumes"

def extract_text_from_pdf(file_path):
    """Extracts all text from a given PDF file."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Add a space instead of a newline to avoid breaking words
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

def clean_text(text):
    """
    Cleans the extracted text by removing extra newlines, non-breaking spaces,
    and multiple consecutive spaces.
    """
    # Replace newlines and non-breaking spaces with a single space
    text = text.replace('\n', ' ').replace('\xa0', ' ')
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_resumes(folder_path):
    """
    Processes all PDF resumes in a folder, extracts the full text,
    cleans it, and returns a DataFrame.
    """
    resume_data = []
    # Loop through each file in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_file_path = os.path.join(folder_path, filename)
            
            # 1. Extract raw text from the PDF
            raw_text = extract_text_from_pdf(full_file_path)
            
            # 2. Clean the raw text into a single string
            cleaned_text = clean_text(raw_text)
            
            # 3. Append the data to our list
            resume_data.append({
                "resume_id": filename.split(".")[0],
                "filename": filename,
                "text": cleaned_text  # The entire resume text is in this one column
            })
            
    # Convert the list of dictionaries into a pandas DataFrame
    return pd.DataFrame(resume_data)

# --- Main execution block ---
if __name__ == "__main__":
    # Ensure the output directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # Process the resumes and get the DataFrame
    df = process_resumes(file_path)
    
    # Save the DataFrame to a CSV file
    output_path = "data/cleaned_resumes.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Successfully processed {len(df)} resumes.")
    print(f"Output saved to: {output_path}")