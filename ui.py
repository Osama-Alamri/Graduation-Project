import streamlit as st
import json

st.title("Job Description Input")

jd = {}
jd['Skills'] = st.text_area("Skills")
jd['Education'] = st.text_area("Education")
jd['Experience'] = st.text_area("Experience")
jd['Projects'] = st.text_area("Projects")
jd['Certificates'] = st.text_area("Certificates")

if st.button("Submit"):
    # Optional: save to JSON
    with open("job_description.json", "w", encoding="utf-8") as f:
        json.dump(jd, f, ensure_ascii=False, indent=4)
    
    st.success("Job Description saved successfully!")
