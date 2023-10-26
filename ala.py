import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.summarization import summarize

st.title("Candidate Selection Tool")
st.subheader("NLP Based Resume Screening")

st.caption("Aim of this project is to check whether a candidate is qualified for a role based on his or her education, experience, and other information captured on their resume. In a nutshell, it's a form of pattern matching between a job's requirements and the qualifications of a candidate based on their resume.")

uploadedJD = st.file_uploader("Upload Job Description", type="pdf")
uploadedResumes = st.file_uploader("Upload resumes", type="pdf", accept_multiple_files=True)

click = st.button("Process")

if click and uploadedJD and uploadedResumes:
    try:
        with pdfplumber.open(uploadedJD) as pdf:
            job_description = pdf.pages[0].extract_text()
    except:
        st.write("Error reading job description PDF")

    job_description = job_description.lower()

    st.write("Job Description:")
    st.write(job_description)

    cv = CountVectorizer()
    matrix = cv.fit_transform([job_description])

    matches = []

    for idx, uploadedResume in enumerate(uploadedResumes):
        try:
            with pdfplumber.open(uploadedResume) as pdf:
                resume_text = pdf.pages[0].extract_text()
        except:
            st.write(f"Error reading resume {idx + 1} PDF")

        resume_text = resume_text.lower()

        similarity_matrix = cosine_similarity(matrix, cv.transform([resume_text]))
        match = similarity_matrix[0][0] * 100
        match = round(match, 2)

        # Summarize the resume text
        summarized_resume = summarize(resume_text, ratio=0.2)  # Adjust ratio as needed

        matches.append((match, summarized_resume))

    matches.sort(key=lambda x: x[0], reverse=True)

    st.write("Top Resumes:")
    for i in range(len(matches)):
        match_percentage, summarized_resume = matches[i]
        st.write(f"Match Percentage for Resume {i + 1}: {match_percentage}%")
        st.write("Summarized Resume:")
        st.write(summarized_resume)
        st.write("-" * 50)

    # Create a bar chart to display match percentages
    chart_data = [match[0] for match in matches]
    st.bar_chart(chart_data)

st.caption(" ~ made by Team P7132")
