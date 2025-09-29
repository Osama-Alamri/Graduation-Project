import os
import pdfplumber
import pandas as pd
import re
import unicodedata
import inflect
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# The path to the folder containing your resume PDF files
file_path = "./resumes"

def extract_text_from_pdf(file_path):
    """Extracts all text from a given PDF file."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

    text = text.replace('\n', ' ').replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
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
    #words = to_lowercase(words)
    #words = remove_punctuation(words)
    words = remove_non_ascii(words)
    #words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    
    return " ".join(words)



def extract_sections_with_regex(text):
    """
    Extracts content for predefined sections from resume text using regex.
    """
    # Define keywords for each section. You can add more variations here.
    section_keywords = {
        'Experience': ['experience', 'work experience', 'professional experience', 'employment history', 'relevant experience'],
        'Education': ['education', 'academic background', 'education and training'],
        'Skills': ['skills', 'technical skills', 'core competencies', 'programming languages', 'technologies'],
        'Projects': ['projects', 'personal projects', 'academic projects'],
        'Certificates / Courses': ['certifications', 'certificates', 'courses', 'professional development', 'licenses & certifications']
    }

    # Initialize a dictionary to hold the extracted text for each section
    sections = {key: "" for key in section_keywords.keys()}
    
    # Flatten the list of all keywords for the main regex pattern
    all_keywords = [item for sublist in section_keywords.values() for item in sublist]

    # Create a regex pattern to find any of the section headers
    # The pattern looks for a keyword at the start of the string or preceded by a space
    pattern_text = r'(?i)\b(' + '|'.join(all_keywords) + r')\b'
    pattern = re.compile(pattern_text)

    # Find all matches of section headers in the text
    matches = list(pattern.finditer(text))

    if not matches:
        return sections # Return empty if no headers are found

    # Iterate through matches to slice the text based on header positions
    for i in range(len(matches)):
        current_match = matches[i]
        
        # Identify which section this header belongs to
        current_header_text = current_match.group(0).lower()
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
        
        # Extract the content, strip whitespace, and remove any leading colons or bullets
        section_content = text[start_index:end_index].strip()
        section_content = re.sub(r'^[:\s•-]+', '', section_content).strip()
        
        sections[current_section_name] = section_content

    return sections

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
            print(f"--- Processing {filename} ---")
            full_file_path = os.path.join(folder_path, filename)
            
            # 1. Extract raw text from the PDF
            raw_text = extract_text_from_pdf(full_file_path)
            
            # 3. Extract sections using the regex function
            sections = extract_sections_with_regex(raw_text)

            # 2. Clean the raw text into a single string
            for k in sections:
                if sections[k]:
                    sections[k] = clean_text(sections[k])
            
            
            # 4. Append the data to our list
            resume_info = {
                "resume_id": filename.split(".")[0],
                "filename": filename
            }
            resume_info.update(sections) # Add all the extracted sections
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
    df.to_csv(output_path, index=False, sep=';')
        
    print(f"\nSuccessfully processed {len(df)} resumes.")
    print(f"Output saved to: {output_path}")
    