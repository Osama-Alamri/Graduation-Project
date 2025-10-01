from transformers import pipeline

# Load the pre-trained NER pipeline
# Note: Using the model "yashpwr/resume-ner-bert-v2" which is a popular and well-documented choice
ner_pipeline = pipeline("ner", model="yashpwr/resume-ner-bert-v2")

# --- Sample Data from our previous example ---
resume_text = """
Michael Johnson
New York, USA | michael.johnson@email.com
 | +1 (555) 123-4567 | LinkedIn: linkedin.com/in/michael-johnson

Professional Summary
Recent Information Systems graduate with strong skills in programming, data analysis, and machine learning. Experienced in developing AI-based projects and passionate about creating practical technology solutions.

Education
Bachelor of Science in Information Systems
New York University, New York, USA – 2025

Relevant coursework: Data Science, Machine Learning, Database Management, Programming

Technical Skills

Programming: Python, Java, SQL

Machine Learning & AI: Scikit-learn, TensorFlow, PyTorch, RoBERTa

Tools & Platforms: Git, Jupyter Notebook, Streamlit

Data Analysis: Pandas, NumPy, Matplotlib, Seaborn

Projects
AI-Powered Recruitment System

Developed a system to screen resumes using NLP and match candidates to job descriptions.

Implemented a chatbot for AI-based interviews and candidate scoring using ML models.

SmartList – AI Task Organizer

Built a Streamlit app that integrates a chatbot with a task management system.

Features include task/subtask creation, progress tracking, and AI-based suggestions.

Experience
Data Analysis Intern
Tech Solutions Inc., New York, USA – June 2024 to August 2024

Cleaned and analyzed large datasets to extract actionable insights.

Created visual reports to support management in decision-making.

Certifications

Machine Learning with Python – Coursera

Data Science Fundamentals – Udemy

Languages

English – Fluent

Spanish – Intermediate
"""

# Run the text through the NER pipeline
ner_results = ner_pipeline(resume_text)

# --- Process the results to make them more readable ---
def process_ner_results(results):
    """A simple function to group entity words together."""
    entities = {}
    current_entity = None
    current_text = ""

    for result in results:
        entity_label = result['entity'].split('-')[-1]  # Get the base label (e.g., 'SKILLS' from 'B-SKILLS')
        word = result['word']
        
        # If the word starts with ##, it's a sub-word, so we attach it to the previous word
        if word.startswith("##"):
            current_text += word.lstrip("##")
        else:
            # If we are starting a new entity, save the old one first
            if current_entity and current_text:
                if current_entity not in entities:
                    entities[current_entity] = []
                entities[current_entity].append(current_text)
            
            # Start a new entity
            current_entity = entity_label
            current_text = word
    
    # Add the last entity
    if current_entity and current_text:
        if current_entity not in entities:
            entities[current_entity] = []
        entities[current_entity].append(current_text)
        
    return entities

# Process and print the structured data
structured_data = process_ner_results(ner_results)
print(structured_data)