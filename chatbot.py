import streamlit as st
from openai import OpenAI
import os
import PyPDF2  # مكتبة لقراءة ملفات PDF
import io


try:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
except KeyError:
    st.error("There is No Key !!")
    st.stop()


# --- دالة لاستخلاص النص من ملفات PDF ---
def get_cv_text(uploaded_file):
    """
    تستخلص هذه الدالة النص من ملف PDF المرفوع.
    """
    text = ""
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == ".pdf":
            # استخدام BytesIO لقراءة الملف المرفوع في الذاكرة
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        else:
            # رسالة تحذير إذا كان الملف ليس PDF
            st.warning("Only PDF Files Allowed")
            return None
            
    except Exception as e:
        st.error(f" An error occurred while reading the file: {e}")
        return None
        
    return text

st.set_page_config(page_title="AI Job Interview", layout="centered")
st.title("📝 AI Job Interview")

# --- إدارة الحالة (Session State) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""



if not st.session_state.interview_started:
    st.write("Welcome to the job interview simulator. Please upload your resume (PDF) to begin the interview.")
    # تم التعديل ليقبل PDF فقط
    uploaded_cv = st.file_uploader(
        "Uploead your CV here (PDF only!)", 
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_cv is not None:
        if st.button("Start the Interview🚀 "):
            with st.spinner("Resume analysis is underway..."):
                cv_text = get_cv_text(uploaded_cv)
                if cv_text:
                    st.session_state.cv_text = cv_text
                    st.session_state.interview_started = True
                    
                    system_prompt = f"""
                    You are a professional recruiter and interviewer (HR Manager)
                    and your task is to conduct a job interview with a candidate.
                    This is the candidate's CV:.
                    ---
                    {st.session_state.cv_text}
                    ---
                    Language of the interview is English
                    
                    Your mission is as follows:
                    1. Start by welcoming the candidate and introducing yourself as an interviewer from a hypothetical company.
                    2. Ask him 10 questions in a sequential and coherent manner based on his experiences and skills mentioned in his CV.
                    3. Ask only one question at a time, and wait for the candidate's answer before asking the next question.
                    4. Make the questions varied (behavioral, technical, about his previous experiences, etc.).
                    5. After the tenth question, thank the candidate for his time and end the interview.

                    Now start with your welcome and ask the first question.
                    """
                    
                    st.session_state.messages.append({"role": "system", "content": system_prompt})

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "أنا جاهز، يمكنك البدء."}]
                    )
                    first_question = response.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": first_question})
                    
                    st.rerun()



# --- واجهة المحادثة ---
if st.session_state.interview_started:
    st.success("Your resume has been successfully analyzed. The interview has started.")

    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    if prompt := st.chat_input("Write your Answer here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("The interviewer thinks..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages
            )
            bot_reply = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})

        st.rerun()