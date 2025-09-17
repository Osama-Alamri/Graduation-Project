import streamlit as st
from openai import OpenAI
import os
import PyPDF2  # مكتبة لقراءة ملفات PDF
import io


try:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
except KeyError:
    st.error("لم يتم العثور على مفتاح OPENAI_API_KEY. يرجى تعيينه كمتغير بيئة.")
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
            st.warning("نوع الملف غير مدعوم. يرجى رفع ملف PDF فقط.")
            return None
            
    except Exception as e:
        st.error(f"حدث خطأ أثناء قراءة الملف: {e}")
        return None
        
    return text

st.set_page_config(page_title="مقابلة وظيفية بالذكاء الاصطناعي", layout="centered")
st.title("📝 مقابلة وظيفية بالذكاء الاصطناعي")

# --- إدارة الحالة (Session State) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""



if not st.session_state.interview_started:
    st.write("أهلاً بك في محاكي المقابلات الوظيفية. يرجى رفع سيرتك الذاتية (PDF) لبدء المقابلة.")
    # تم التعديل ليقبل PDF فقط
    uploaded_cv = st.file_uploader(
        "ارفع سيرتك الذاتية هنا (PDF فقط)", 
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_cv is not None:
        if st.button("🚀 ابدأ المقابلة"):
            with st.spinner("...جاري تحليل السيرة الذاتية"):
                cv_text = get_cv_text(uploaded_cv)
                if cv_text:
                    st.session_state.cv_text = cv_text
                    st.session_state.interview_started = True
                    
                    system_prompt = f"""
                    أنت خبير توظيف ومحاور محترف (HR Manager) ومهمتك هي إجراء مقابلة وظيفية مع مرشح.
                    هذه هي السيرة الذاتية للمرشح:
                    ---
                    {st.session_state.cv_text}
                    ---
                    مهمتك هي كالتالي:
                    1. ابدأ بالترحيب بالمرشح وتقديم نفسك كمحاور من شركة افتراضية.
                    2. اطرح عليه 10 أسئلة بشكل متسلسل ومترابط بناءً على خبراته ومهاراته المذكورة في سيرته الذاتية.
                    3. اطرح سؤالاً واحداً فقط في كل مرة، وانتظر إجابة المرشح قبل طرح السؤال التالي.
                    4. اجعل الأسئلة متنوعة (أسئلة سلوكية، تقنية، عن خبراته السابقة، إلخ).
                    5. بعد السؤال العاشر، اشكر المرشح على وقته وأنهِ المقابلة.

                    ابدأ الآن بترحيبك وطرح السؤال الأول.
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
    st.success("تم تحليل السيرة الذاتية بنجاح. المقابلة قد بدأت.")

    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    if prompt := st.chat_input("اكتب إجابتك هنا..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("...يفكر المحاور"):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages
            )
            bot_reply = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})

        st.rerun()



        
# prompt = st.chat_input("Say something")
# if prompt:
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=st.session_state.messages
#     )
#     bot_reply = response.choices[0].message.content
#     st.session_state.messages.append({"role": "assistant", "content": bot_reply})

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])  