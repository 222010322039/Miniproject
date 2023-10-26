import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import altair as alt
import re  # Import the regular expressions module

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
    btech_cgpa = []
    academic_percentages = []

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

        # Extract B.Tech CGPA using regular expressions
        cgpa_match = re.search(r"B\.Tech CGPA: (\d+\.\d+|\d+)", resume_text)
        if cgpa_match:
            btech_cgpa.append(float(cgpa_match.group(1)))
        else:
            btech_cgpa.append(None)

        # Extract academic percentage from CGPA
        academic_cgpa_match = re.search(r"CGPA: (\d+\.\d+|\d+)%", resume_text)
        if academic_cgpa_match:
            academic_percentages.append(float(academic_cgpa_match.group(1)))
        else:
            academic_percentages.append(None)

    matches.sort(key=lambda x: x[0], reverse=True)

    st.write("Top Resumes:")
    percentages = [match[0] for match in matches]  # Extract match percentages

    for i in range(len(matches)):
        match_percentage, resume_text = matches[i]
        st.write(f"Match Percentage for Resume {i + 1}: {match_percentage}%")

        # Extract and display a shorter description (e.g., first 5 lines)
        resume_lines = resume_text.split('\n')[:5]
        summarized_resume = "\n".join(resume_lines)
        st.write("Summary of Resume:")
        st.write(summarized_resume)

        st.write("Skills Count:")
        st.write(skills_count[i])

        # Display B.Tech CGPA
        st.write("B.Tech CGPA: {:.2f}".format(btech_cgpa[i] if btech_cgpa[i] is not None else 0.0))

        # Display academic percentage
        st.write("Academic Percentage: {:.2f}%".format(academic_percentages[i] if academic_percentages[i] is not None else 0.0))

        st.write("-" * 50)

    # Create a bar chart using Altair for B.Tech CGPA
    cgpa_df = pd.DataFrame({'Resume': [f"Resume {i + 1}" for i in range(len(btech_cgpa))], 'B.Tech CGPA': btech_cgpa})
    
    cgpa_chart = alt.Chart(cgpa_df).mark_bar().encode(
        x='Resume:N',
        y='B.Tech CGPA:Q',
        color='Resume:N'
    ).properties(
        width=600  # Increase the width as needed
    )

    st.altair_chart(cgpa_chart, use_container_width=True)

    # Create a bar chart for skills count
    skills_df = pd.DataFrame(skills_count)
    skills_df['Resume'] = [f"Resume {i+1}" for i in range(len(skills_count))]

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
        'Resume': [f"Resume {i + 1}" for i in range(len(percentages))],
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
