import streamlit as st

st.set_page_config(page_title="Recruitment System", page_icon=":guardsman:", layout="wide")

# ---------- styles (keep your tabs centered & big) ----------
st.markdown("""
<style>
.stTabs [role="tablist"]{justify-content:center}
.stTabs [role="tab"]{font-size:50px;padding:30px 50px}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;
       background:rgba(124,58,237,.12);color:#7c3aed;font-weight:600;font-size:12px}
.meta{color:gray;font-size:13px}
</style>
""", unsafe_allow_html=True)

# ---------- simple store ----------
# if "job_posts" not in st.session_state:
#     st.session_state.job_posts = []   # list of dicts

# ---------- tabs ----------
homeTab , aboutTab , contactTab , jobsTab = st.tabs(["Home", "About us", "Contact", "Jobs"])

with homeTab:
    st.header("Home")
    st.write("Welcome to the Home page!")

with aboutTab:
    st.header("About us")
    st.write("Learn more about us on this page.")

with contactTab:
    st.header("Contact")
    st.write("Get in touch with us through this page.")

# ========================== JOBS TAB ==========================
with jobsTab:
    st.header("Jobs")
    st.caption("Design-only form to add a job post. No backend yet.")

    # with st.form("job_form", border=True):
    #     st.subheader("Create job post")

    #     col1, col2 = st.columns([2,1])
    #     with col1:
    #         title = st.text_input("Job title", placeholder="e.g., Data Scientist")
    #     with col2:
    #         total_q = st.number_input("Total questions", min_value=1, max_value=50, value=20, step=1)

    #     requirements = st.text_area(
    #         "Job requirements / description",
    #         placeholder="Write the key responsibilities and requirements here...",
    #         height=140
    #     )

    #     st.markdown("**AI question settings**")
    #     c1, c2, c3 = st.columns(3)
    #     with c1:
    #         cv_pct = st.slider("Percent from CV", 0, 100, 50)
    #     with c2:
    #         req_pct = 100 - cv_pct
    #         st.metric("Percent from requirements", f"{req_pct}%")
    #     with c3:
    #         ai_weight = st.number_input("AI scoring weight (0–100)", min_value=0, max_value=100, value=70, step=5)

    #     manual_block = st.text_area(
    #         "Manual questions (one per line)",
    #         placeholder="e.g.\nExplain your most impactful project.\nWhat KPIs would you use for churn?",
    #         height=120
    #     )

    #     submitted = st.form_submit_button("Add job post", use_container_width=True)

    # # ----------- validate + compute split -----------
    # if submitted:
    #     manual_questions = [q.strip() for q in manual_block.split("\n") if q.strip()]
    #     mcount = len(manual_questions)

    #     errors = []
    #     if not title:
    #         errors.append("Please enter a job title.")
    #     if not requirements:
    #         errors.append("Please write job requirements/description.")
    #     if mcount > total_q:
    #         errors.append(f"Manual questions ({mcount}) cannot exceed total questions ({total_q}).")

    #     if errors:
    #         for e in errors: st.error(e)
    #     else:
    #         ai_total = total_q - mcount
    #         ai_from_cv = round(ai_total * (cv_pct/100))
    #         ai_from_req = ai_total - ai_from_cv

    #         post = {
    #             "title": title,
    #             "requirements": requirements,
    #             "total_questions": total_q,
    #             "ai_total": ai_total,
    #             "ai_from_cv": ai_from_cv,
    #             "ai_from_req": ai_from_req,
    #             "ai_weight": ai_weight,
    #             "manual_questions": manual_questions,
    #         }
    #         st.session_state.job_posts.append(post)

    #         st.success("Job post added (design only).")
    #         with st.expander("Preview / details", expanded=True):
    #             st.markdown(f"### {title}  <span class='badge'>Draft</span>", unsafe_allow_html=True)
    #             st.markdown(f"<div class='meta'>AI scoring weight: {ai_weight}%</div>", unsafe_allow_html=True)
    #             st.markdown("#### Requirements")
    #             st.write(requirements)

    #             st.markdown("#### Question plan")
    #             c1, c2, c3, c4 = st.columns(4)
    #             c1.metric("Total", post["total_questions"])
    #             c2.metric("Manual", len(post["manual_questions"]))
    #             c3.metric("AI from CV", post["ai_from_cv"])
    #             c4.metric("AI from Requirements", post["ai_from_req"])

    #             if manual_questions:
    #                 st.markdown("##### Manual questions")
    #                 for i, q in enumerate(manual_questions, 1):
    #                     st.write(f"{i}. {q}")

    # # ----------- list of created posts -----------
    # if st.session_state.job_posts:
    #     st.markdown("---")
    #     st.subheader("Your job posts")
    #     for i, p in enumerate(st.session_state.job_posts, 1):
    #         with st.container(border=True):
    #             st.markdown(f"**{i}. {p['title']}**  &nbsp; <span class='meta'>(Total {p['total_questions']} • Manual {len(p['manual_questions'])} • AI {p['ai_total']} = CV {p['ai_from_cv']} + Req {p['ai_from_req']})</span>", unsafe_allow_html=True)
    #             st.caption(p["requirements"][:240] + ("..." if len(p["requirements"]) > 240 else ""))
