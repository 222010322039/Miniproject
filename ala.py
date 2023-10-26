import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import altair as alt

st.title("Candidate Selection Tool")
st.subheader("NLP Based Resume Screening")

# ... (Rest of your code)

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

        matches.append((match, resume_text))

    matches.sort(key=lambda x: x[0], reverse=True)

    st.write("Top Resumes:")
    percentages = [match[0] for match in matches]  # Extract match percentages

    for i in range(len(matches)):
        match_percentage, resume_text = matches[i]
        st.write(f"Match Percentage for Resume {i + 1}: {match_percentage}%")
        st.write(resume_text)
        st.write("-" * 50)

    # Create a bar chart using Altair
    df = pd.DataFrame({'Resume': [f"Resume {i+1}" for i in range(len(percentages)], 'Match Percentage': percentages})
    chart = alt.Chart(df).mark_bar().encode(
        x='Resume',
        y='Match Percentage'
    ).properties(
        width=400
    )

    st.altair_chart(chart, use_container_width=True)

st.caption(" ~ made by Team P7132")
