import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import altair as alt
import subprocess  # Import subprocess for library installation

# Install bert-extractive-summarizer within Streamlit
subprocess.call(["pip", "install", "bert-extractive-summarizer"])

# Import necessary modules for summarization
from summarizer import Summarizer

st.title("Candidate Selection Tool")
st.subheader("NLP Based Resume Screening")

# Define a list of skills to search for in the resumes
skills_to_search = ["Python", "Java", "Machine Learning", "Data Analysis", "Communication", "Teamwork", "Project Management"]

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
    skills_count = []

    for idx, uploadedResume in enumerate(uploadedResumes):
        try:
            with pdfplumber.open(uploadedResume) as pdf:
                resume_text = pdf.pages[0].extract_text()
        except:
            st.write(f"Error reading resume {idx + 1} PDF")

        resume_text = resume_text.lower()

        # Calculate similarity
        similarity_matrix = cosine_similarity(matrix, cv.transform([resume_text]))
        match = round(similarity_matrix[0][0] * 100, 2)

        matches.append((match, resume_text))

        # Count skills in the resume
        skill_count = {skill: resume_text.count(skill.lower()) for skill in skills_to_search}
        skills_count.append(skill_count)

    matches.sort(key=lambda x: x[0], reverse=True)

    st.write("Top Resumes:")
    percentages = [match[0] for match in matches]  # Extract match percentages

    for i in range(len(matches)):
        match_percentage, resume_text = matches[i]
        st.write(f"Match Percentage for Resume {i + 1}: {match_percentage}%")
        st.write("Summary of Resume:")

        # Summarize the resume text
        summarizer = Summarizer()
        summarized_text = summarizer(resume_text, ratio=0.2)  # Adjust ratio as needed
        st.write(summarized_text)

        st.write("Skills Count:")
        st.write(skills_count[i])
        st.write("-" * 50)

    # Create a bar chart using Altair for skills count
    skills_df = pd.DataFrame(skills_count)
    skills_df['Resume'] = [f"Resume {i+1}" for i in range(len(percentages))]

    skills_chart = alt.Chart(skills_df).transform_fold(
        skills_to_search,
        as_=['Skill', 'Count']
    ).mark_bar().encode(
        alt.X('Skill:N', title='Skill'),
        alt.Y('Count:Q', title='Skill Count'),
        alt.Color('Resume:N', title='Resume')
    ).properties(
        width=600  # Increase the width as needed
    )

    st.altair_chart(skills_chart, use_container_width=True)

    # Create a bar chart for match percentages
    match_percentages_df = pd.DataFrame({
        'Resume': [f"Resume {i+1}" for i in range(len(percentages))],
        'Match Percentage': percentages
    })

    match_percentages_chart = alt.Chart(match_percentages_df).mark_bar().encode(
        alt.X('Resume:N', title='Resume'),
        alt.Y('Match Percentage:Q', title='Match Percentage')
    ).properties(
        width=600  # Increase the width as needed
    )

    st.altair_chart(match_percentages_chart, use_container_width=True)

st.caption(" ~ made by Team P7132")
