import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.title("Candidate Selection Tool")
st.subheader("NLP Based Resume Screening")

# ... (Rest of your code)

if click and uploadedJD and uploadedResumes:
    # ... (Your existing code)

    matches.sort(key=lambda x: x[0], reverse=True)

    st.write("Top Resumes:")
    percentages = [match[0] for match in matches]  # Extract match percentages

    for i in range(len(matches)):
        match_percentage, resume_text = matches[i]
        st.write(f"Match Percentage for Resume {i + 1}: {match_percentage}%")
        st.write(resume_text)
        st.write("-" * 50)

    # Create a bar graph
    plt.figure(figsize=(10, 6))
    plt.bar([f"Resume {i+1}" for i in range(len(percentages))], percentages)
    plt.xlabel("Resumes")
    plt.ylabel("Match Percentage")
    plt.title("Match Percentage for Resumes")
    st.pyplot(plt)  # Display the graph in Streamlit

st.caption(" ~ made by Team P7132")
