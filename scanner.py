# --- Imports ---
import os
import re
import unicodedata
import locale # Added for universal CSV separator
from datetime import datetime, date

# Third-party libraries
import nltk
import pandas as pd
import pdfplumber
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Initial Setup ---

# This function checks if you have the necessary NLTK packages and downloads them if not.
def setup_nltk():
    """Checks for NLTK resources and downloads if missing."""
    resources = ["punkt", "stopwords", "wordnet"]
    print("Checking for NLTK resources...")
    for resource in resources:
        try:
            # The path is different for tokenizers vs. corpora
            if resource == "punkt":
                nltk.data.find(f"tokenizers/{resource}")
            else:
                nltk.data.find(f"corpora/{resource}")
            print(f"  - '{resource}' found.")
        except LookupError:
            print(f"  - '{resource}' not found. Downloading...")
            nltk.download(resource)

# --- PDF and Text Processing Functions ---

def extract_text_from_pdf(file_path):
    """Extracts all text from a given PDF file."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # Using x_tolerance helps fix words that are too close together.
                page_text = page.extract_text(x_tolerance=3) 
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"    - Could not read file {os.path.basename(file_path)}: {e}")
        return "" # Return empty string on failure

    # Basic cleanup to handle weird spaces and empty lines
    text = text.replace('\xa0', ' ')
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text) 
    return text.strip()


def extract_sections(text):
    """
    Extracts content into predefined sections from resume text.
    It finds all headers first, then slices the text to be more accurate.
    """
    # These are the keywords we'll look for as section headers.
    section_keywords = {
        'Experience': ['experience', 'work experience', 'professional experience', 'employment history', 'relevant experience', 'Professional Experience', 'work history'],
        'Education': ['education', 'academic background', 'education and training'],
        'Skills': ['skills', 'technical skills', 'core competencies', 'programming languages'],
        'Projects': ['projects', 'personal projects', 'academic projects'],
        'Certificates / Courses': ['certifications', 'certificates', 'courses', 'professional development', 'licenses & certifications']
    }

    sections = {key: "" for key in section_keywords.keys()}
    all_keywords = [item for sublist in section_keywords.values() for item in sublist]
    
    # This regex finds keywords only at the start of a line to avoid mistakes.
    pattern = re.compile(r'(?im)^(?:[ \t\r\f\v]*\b(' + '|'.join(all_keywords) + r')\b)', re.MULTILINE)
    matches = list(pattern.finditer(text))
    
    print("    - Detected headers:", [m.group(1).strip() for m in matches])

    if not matches:
        return sections

    for i, current_match in enumerate(matches):
        header_text = current_match.group(1).strip().lower()
        
        # Find which section this header belongs to
        current_section_name = None
        for section_name, keywords in section_keywords.items():
            # We check a lowercase version to make matching easier
            if header_text in [k.lower() for k in keywords]:
                current_section_name = section_name
                break
        
        if not current_section_name:
            continue
            
        # Get the text between this header and the next one
        start_index = current_match.end()
        end_index = matches[i+1].start() if i + 1 < len(matches) else len(text)
        
        content = text[start_index:end_index].strip()
        content = re.sub(r'^[:\s•*-]+', '', content).strip() # Remove leading bullets
        
        sections[current_section_name] = (sections[current_section_name] + "\n" + content).strip()

    return sections


def clean_text(text):
    """A pipeline to clean text for keyword matching."""
    if not isinstance(text, str):
        return ""

    words = word_tokenize(text)
    
    # 1. Convert to lowercase
    words = [word.lower() for word in words]
    
    # 2. Remove punctuation
    words = [re.sub(r'[^\w\s]', '', word) for word in words]
    
    # 3. Remove all numbers (digits)
    words = [re.sub(r'\d+', '', word) for word in words]

    # 4. Remove non-ASCII characters
    words = [unicodedata.normalize('NFKD', w).encode('ascii', 'ignore').decode('utf-8', 'ignore') for w in words]

    # 5. Remove stopwords (common words like 'the', 'a', 'is')
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word and word not in stop_words]
    
    # 6. Lemmatize (turn words like 'running' into 'run')
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    
    return " ".join(words)


def calculate_experience_python(text: str) -> float:
    """
    Calculates total unique years of experience from a list of job lines.
    Handles overlaps and gaps in employment.
    """
    today = datetime.now().date()
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    # This complex regex finds dates in "Month YYYY" or "MM/YYYY" format, or "Present"
    date_pattern = re.compile(
        r"""
        (\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\b\s*(\d{4})) | # Format 1
        (\b(\d{1,2})\s*/\s*(\d{4})\b) | # Format 2
        (\b(Present|Current)\b)         # Format 3
        """, re.IGNORECASE | re.VERBOSE)
    
    intervals = []
    for line in text.split('\n'):
        matches = list(date_pattern.finditer(line))
        
        if len(matches) >= 2:
            try:
                # --- Parse Start Date ---
                start_match = matches[0].groups()
                if start_match[0]: # "Month YYYY" format
                    start_month = month_map[start_match[1].lower()[:3]]
                    start_year = int(start_match[2])
                else: # "MM/YYYY" format
                    start_month = int(start_match[4])
                    start_year = int(start_match[5])
                start_date = date(start_year, start_month, 1)
                
                # --- Parse End Date ---
                end_match = matches[1].groups()
                if end_match[6]: # It's 'Present' or 'Current'
                    end_date = today
                else:
                    if end_match[0]: # "Month YYYY" format
                        end_month = month_map[end_match[1].lower()[:3]]
                        end_year = int(end_match[2])
                    else: # "MM/YYYY" format
                        end_month = int(end_match[4])
                        end_year = int(end_match[5])

                    # Use the 1st of the *next* month for correct duration
                    if end_month == 12:
                        end_date = date(end_year + 1, 1, 1)
                    else:
                        end_date = date(end_year, end_month + 1, 1)
                
                if start_date and end_date:
                    intervals.append((start_date, end_date))
            except (ValueError, TypeError, IndexError):
                # Silently skip lines that look like dates but aren't
                pass

    if not intervals: return 0.0

    # --- Merge Overlapping Intervals ---
    intervals.sort()
    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end: # Check for overlap
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    
    # --- Sum Durations ---
    total_days = sum((end - start).days for start, end in merged)
    return round(total_days / 365.25, 1)


def process_all_resumes(folder_path):
    """
    Loops through all PDFs, processes them, and returns a DataFrame.
    """
    resume_data = []
    
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found. Please create it and add your resumes.")
        return pd.DataFrame() # Return empty dataframe

    files_to_process = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    print(f"\nFound {len(files_to_process)} PDF files to process.")

    for filename in files_to_process:
        print(f"\n--- Processing '{filename}' ---")
        full_file_path = os.path.join(folder_path, filename)
        
        # Step 1: Extract raw text
        raw_text = extract_text_from_pdf(full_file_path)
        if not raw_text:
            continue
        
        # Step 2: Extract sections
        sections = extract_sections(raw_text)

        # Step 3: Filter experience text to find only the lines with job dates
        experience_text = sections.get('Experience', '')
        
        # This complex regex ensures we only grab lines that are very likely to be job headers
        date_range_pattern = re.compile(
            r".*("
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\b\s*\d{4}" # Month YYYY
            r"|\b\d{1,2}/\d{4}\b" # MM/YYYY
            r").*(to|–|-|Present|Current).*", re.IGNORECASE)
        
        filtered_lines = []
        if experience_text:
            for line in experience_text.split('\n'):
                if date_range_pattern.search(line):
                    filtered_lines.append(line)
        
        # Step 4: Calculate experience years
        experience_in_years = 0.0
        if filtered_lines:
            filtered_text = "\n".join(filtered_lines)
            experience_in_years = calculate_experience_python(filtered_text)
        print(f"    - Total Experience Found: {experience_in_years} years")

        # Step 5: Clean all sections and prepare for saving
        # The key will be 'cleaned_Experience', 'cleaned_Skills', etc.
        cleaned_sections = {f"cleaned_{k}": clean_text(v) for k, v in sections.items()}

        # Step 6: Assemble all the data for this resume
        resume_info = {
            "resume_id": os.path.splitext(filename)[0],
            "filename": filename,
            "total_experience_years": experience_in_years
        }
        resume_info.update(cleaned_sections)
        resume_data.append(resume_info)

    return pd.DataFrame(resume_data)

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # Setup NLTK (downloads packages if needed)
    setup_nltk()
    
    # Define file paths
    resume_folder = "./resumes"
    output_folder = "./data"
    output_filename = "cleaned_resumes.csv"

    # Make sure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process the resumes
    df = process_all_resumes(resume_folder)
    
    # Save the final data to a CSV file
    if not df.empty:
        output_path = os.path.join(output_folder, output_filename)
        
        # --- Universal Separator Logic ---
        # Using a semicolon ';' is often more compatible with versions of Excel
        # in regions that use a comma ',' as a decimal separator.
        csv_separator = ";"
        
        print(f"\nUsing '{csv_separator}' as the CSV separator for better Excel compatibility.")

        # Using 'utf-8-sig' encoding helps Excel open the file correctly.
        df.to_csv(output_path, index=False, sep=csv_separator, encoding='utf-8-sig')
        
        print(f"\nSuccessfully processed {len(df)} resumes.")
        print(f"Output saved to: {output_path}")
    else:
        print("\nNo resumes were processed.")

