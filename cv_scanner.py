import os
import pdfplumber
import pandas as pd
import re
import unicodedata
import inflect
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import nltk
import locale
from datetime import datetime, date

sep = ";" if locale.getdefaultlocale()[0] in ["ar_SA", "fr_FR", "de_DE"] else ","


# 🔽 ADD THIS BLOCK RIGHT HERE
resources = ["punkt", "punkt_tab", "stopwords", "wordnet"]
for r in resources:
    try:
        if "punkt" in r:
            nltk.data.find(f"tokenizers/{r}")
        else:
            nltk.data.find(f"corpora/{r}")
    except LookupError:
        nltk.download(r)
# 🔼 END OF BLOCK

###################################################################################################################
# AI Function
###################################################################################################################

# 🔴 REPLACE your AI function one last time with this definitive version 🔴

# 🔴 ADD THIS NEW FUNCTION. It replaces the AI.

def calculate_experience_python(text: str) -> float:
    """
    Calculates the total unique, non-overlapping years of experience
    from a block of text containing job/date lines using Python.
    
    Handles "Month YYYY" and "MM/YYYY" formats.
    """
    # Define current date for 'Present'
    today = datetime.now().date()
    
    # Map for converting short month names to numbers
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    # This pattern handles both "Month YYYY" and "MM/YYYY" and "Present"
    date_pattern = re.compile(
        r"""
        # Format 1: "Month YYYY"
        (\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\b) # Month (Group 2)
        \s*
        (\d{4}) # Year (Group 3)
        
        | # OR
        
        # Format 2: "MM/YYYY"
        \b(\d{1,2})\s*/\s*(\d{4})\b # Month (Group 4), Year (Group 5)
        
        | # OR
        
        # Format 3: "Present" / "Current"
        (\b(Present|Current)\b) # Present (Group 7)
        """,
        re.IGNORECASE | re.VERBOSE
    )
    
    intervals = []
    for line in text.split('\n'):
        matches = list(date_pattern.finditer(line))
        
        # We need at least two dates (start and end) to form a range
        if len(matches) >= 2:
            try:
                # --- Parse Start Date ---
                start_match = matches[0]
                start_month = 0
                start_year = 0
                
                if start_match.group(2): # "Month YYYY" format
                    start_month_str = start_match.group(2).lower()[:3]
                    start_month = month_map[start_month_str]
                    start_year = int(start_match.group(3))
                elif start_match.group(4): # "MM/YYYY" format
                    start_month = int(start_match.group(4))
                    start_year = int(start_match.group(5))
                
                start_date = date(start_year, start_month, 1)
                
                # --- Parse End Date ---
                end_match = matches[1]
                end_date = None
                
                if end_match.group(7): # It's 'Present' or 'Current'
                    end_date = today
                else:
                    end_month = 0
                    end_year = 0
                    if end_match.group(2): # "Month YYYY" format
                        end_month_str = end_match.group(2).lower()[:3]
                        end_month = month_map[end_month_str]
                        end_year = int(end_match.group(3))
                    elif end_match.group(4): # "MM/YYYY" format
                        end_month = int(end_match.group(4))
                        end_year = int(end_match.group(5))

                    # Use 1st of *next* month for correct duration
                    if end_month == 12:
                        end_date = date(end_year + 1, 1, 1)
                    else:
                        end_date = date(end_year, end_month + 1, 1)

                if start_date and end_date:
                    intervals.append((start_date, end_date))
                
            except Exception as e:
                print(f"Error parsing date line: {line} | Error: {e}")

    if not intervals:
        return 0.0

    # --- Merge Overlapping Intervals ---
    intervals.sort(key=lambda x: x[0])
    
    merged_intervals = []
    if not intervals:
        return 0.0
        
    merged_intervals.append(intervals[0])
    
    for i in range(1, len(intervals)):
        current_start, current_end = intervals[i]
        last_start, last_end = merged_intervals[-1]
        
        if current_start <= last_end: # Check for overlap
            merged_intervals[-1] = (last_start, max(last_end, current_end))
        else:
            merged_intervals.append((current_start, current_end))
    
    # --- Sum Durations ---
    total_days = 0
    for start, end in merged_intervals:
        total_days += (end - start).days

    total_years = total_days / 365.25
    
    return round(total_years, 1)
    
###################################################################################################################
# AI Function
###################################################################################################################



# The path to the folder containing your resume PDF files
file_path = "./resumes"

# 🔴 REPLACE this function in cv_scanner.py 🔴

def extract_text_from_pdf(file_path):
    """Extracts all text from a given PDF file."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            
            # --- THIS IS THE CRITICAL FIX ---
            # We add 'x_tolerance=3' to force pdfplumber to insert
            # spaces between words that are close horizontally.
            # This fixes the "CompanyNameJuly" -> "CompanyName July" problem.
            page_text = page.extract_text(x_tolerance=3) 
            
            if page_text:
                text += page_text + "\n" # Use a newline to separate pages

    # --- IMPROVED CLEANUP ---
    # We keep newlines (\n) because your section parser needs them.
    # 1. Replace non-breaking spaces
    text = text.replace('\xa0', ' ')
    # 2. Collapse multiple spaces into one space
    text = re.sub(r' {2,}', ' ', text)
    # 3. (Optional but good) Remove lines that are just whitespace
    text = re.sub(r'\n\s+\n', '\n', text) 
    
    return text.strip()

def remove_non_ascii(words):
    """Remove non-ASCII characters from a list of tokenized words."""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from a list of tokenized words."""
    new_words = []
    for word in words:
        new_words.append(word.lower())
    return new_words

def remove_punctuation(words):
    """Remove punctuation from a list of tokenized words."""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all integer occurrences in a list of tokenized words with textual representation."""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from a list of tokenized words."""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def lemmatize_verbs(words):
    """Lemmatize verbs in a list of tokenized words."""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def clean_text(text):
    

    words = word_tokenize(text)
    # Call the helper functions in a logical order
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_non_ascii(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    
    return " ".join(words)



# 🔴 REPLACE your entire section extractor with this new, more robust version 🔴

def extract_sections_with_regex(text):
    """
    Extracts content for predefined sections from resume text using a more
    robust method that finds all headers first before slicing.
    """
    # Define keywords for each section.
    section_keywords = {
        'Experience': ['experience', 'work experience', 'professional experience', 'employment history', 'relevant experience','Professional Experience','work history'],
        'Education': ['education', 'academic background', 'education and training'],
        'Skills': ['skills', 'technical skills', 'core competencies', 'programming languages'],
        'Projects': ['projects', 'personal projects', 'academic projects'],
        'Certificates / Courses': ['certifications', 'certificates', 'courses', 'professional development', 'licenses & certifications']
    }

    sections = {key: "" for key in section_keywords.keys()}
    
    # Flatten the list of all keywords for the main regex pattern
    all_keywords = [item for sublist in section_keywords.values() for item in sublist]
    
    # This pattern finds any of the keywords at the beginning of a line,
    # ignoring case and matching multiline.
    pattern = re.compile(r'(?im)^(?:[ \t\r\f\v]*\b(' + '|'.join(all_keywords) + r')\b)', re.MULTILINE)

    # Find all matches of section headers in the text
    matches = list(pattern.finditer(text))

    print("Detected section headers:", [m.group(1) for m in matches])

    if not matches:
        return sections # Return empty if no headers are found

    # Iterate through matches to slice the text based on header positions
    for i in range(len(matches)):
        current_match = matches[i]
        
        # Identify which section this header belongs to
        # Group 1 contains just the keyword text (e.g., "experience")
        current_header_text = current_match.group(1).strip().lower()
        current_section_name = None
        for section_name, keywords in section_keywords.items():
            if current_header_text in keywords:
                current_section_name = section_name
                break
        
        if not current_section_name:
            continue
            
        # Determine the start and end indices of the section's content
        start_index = current_match.end()
        end_index = matches[i+1].start() if i + 1 < len(matches) else len(text)
        
        # Extract the content, strip whitespace, and remove leading punctuation
        section_content = text[start_index:end_index].strip()
        section_content = re.sub(r'^[:\s•*-]+', '', section_content).strip()
        
        # Append content if the section has been found before (e.g., multiple 'skills' sections)
        if sections.get(current_section_name):
             sections[current_section_name] += "\n" + section_content
        else:
             sections[current_section_name] = section_content

    return sections

# 🔴 REPLACE your old function with this new one for debugging 🔴

def process_resumes(folder_path):
    """
    Processes all PDF resumes in a folder, extracts the full text,
    splits it into sections, and returns a DataFrame.
    """
    resume_data = []
    
    files_to_process = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    print(f"Found {len(files_to_process)} PDF files to process.")

    # Loop through each file in the specified folder
    for filename in files_to_process:
        try:
            print(f"\n========================================================")
            print(f"--- Processing {filename} ---")
            
            experience_in_years = 0.0
            
            full_file_path = os.path.join(folder_path, filename)
            
            # 1. Extract raw text from the PDF
            raw_text = extract_text_from_pdf(full_file_path)
            
            # 2. Extract sections using the regex function
            sections = extract_sections_with_regex(raw_text)

            # 3. Get the raw experience text
            experience_text = sections.get('Experience', '')
            
            # --- NEW STEP 3.5: Pre-filter text (High-Precision Final Version) ---
            filtered_experience_text = ""
            if experience_text.strip():
                date_lines = []
                
                # --- We now use two, much stricter patterns ---

                # Pattern 1: Finds "Month YYYY ... to ... (Month YYYY | Present)"
                # This is high-confidence.
                date_range_pattern = re.compile(
                    r"""
                    ( # Start of the whole pattern
                        \b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\b # Month
                        \s*\d{4} # Year
                    ) # End of start date
                    .* # Any text in between
                    (to|–|-) # Separator
                    .* # Any text in between
                    ( # Start of the end date
                        \b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\b # Month
                        \s*\d{4} # Year
                        | # OR
                        \b(Present|Current)\b # Present/Current
                    ) # End of the end date
                    """,
                    re.IGNORECASE | re.VERBOSE
                )

                # Pattern 2: Finds "MM/YYYY ... to ... (MM/YYYY | Present)"
                # This is also high-confidence.
                mm_yyyy_range_pattern = re.compile(
                    r"""
                    ( # Start of the whole pattern
                        \b\d{1,2}/\d{4}\b # MM/YYYY
                    ) # End of start date
                    .* # Any text in between
                    (to|–|-) # Separator
                    .* # Any text in between
                    ( # Start of the end date
                        \b\d{1,2}/\d{4}\b # MM/YYYY
                        | # OR
                        \b(Present|Current)\b # Present/Current
                    ) # End of the end date
                    """,
                    re.IGNORECASE | re.VERBOSE
                )

                for line in experience_text.split('\n'):
                    # The line must match ONE of our high-precision patterns
                    if date_range_pattern.search(line) or \
                    mm_yyyy_range_pattern.search(line):
                        
                        date_lines.append(line.strip())
                        
                filtered_experience_text = "\n".join(date_lines)


            # 🔴 In process_resumes, modify Step 4 to this:

            # 4. Analyze the text if it's valid
            # We check our 'filtered_experience_text'
            
            if filtered_experience_text:
                print(f"Valid experience lines found. Calculating...")
                
                # Your debug print, showing the CLEANED text
                print(f"\n--- DEBUG: FILTERED TEXT SENT TO CALCULATOR ---\n{filtered_experience_text}\n--- END DEBUG ---\n")
                
                # --- THIS IS THE ONLY CHANGE IN THIS BLOCK ---
                # We call our reliable Python function instead of the AI
                experience_in_years = calculate_experience_python(filtered_experience_text)
                
                print(f" -> Found {experience_in_years} years.")
            else:
                print(f" -> Skipping analysis: 'Experience' section appears empty or has no valid dates.")

            # 🔴 END OF MODIFICATION 🔴

            # 5. Clean the *original* raw text for cosine similarity
            # This part remains the same. We still want the *full* text
            # for our keyword matching (cosine similarity).
            for k in sections:
                if sections[k]:
                    sections[k] = clean_text(sections[k])
            
            # 6. Append the data to our list
            resume_info = {
                "resume_id": filename.split(".")[0],
                "filename": filename,
                "total_experience_years": experience_in_years
            }
            resume_info.update(sections)
            resume_data.append(resume_info)
        except Exception as e:
            print(f"Could not process {filename}. Error: {e}")
            
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
    df.to_csv(output_path, index=False, sep=sep)
        
    print(f"\nSuccessfully processed {len(df)} resumes.")
    print(f"Output saved to: {output_path}")
    