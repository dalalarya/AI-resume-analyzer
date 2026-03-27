import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import matplotlib.pyplot as plt

# -------------------------------
# FUNCTION: Extract text from PDF
# -------------------------------
def extract_text(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# -------------------------------
# FUNCTION: Cleaning text
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

# --------------------------------------------------------------------------------
# FUNCTION: Calculating the similarity between uploaded Resume and Job description
# --------------------------------------------------------------------------------
def calculate_similarity(resume, jd):
    texts = [resume, jd]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(similarity[0][0] * 100, 2)

# -------------------------------
# FUNCTION: Missing keywords
# -------------------------------
def missing_keywords(resume, jd):
    jd_words = set(jd.split())
    resume_words = set(resume.split())
    missing = jd_words - resume_words
    return list(missing)[:15]

# -------------------------------
# FUNCTION: Top 10 keywords
# -------------------------------
def top_keywords(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([text])
    
    words = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]
    
    word_scores = list(zip(words, scores))
    sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)
    
    return [word for word, score in sorted_words[:10]]

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("AI Resume Analyzer")
st.write("Compare your resume with a job description")

# Upload resume
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# Job description input
job_description = st.text_area("Paste Job Description")

# Button
if st.button("Analyze Resume"):

    if uploaded_file is not None and job_description != "":
        
        # Extract text
        resume_text = extract_text(uploaded_file)

        # Clean text
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(job_description)

        # Calculate similarity
        score = calculate_similarity(resume_clean, jd_clean)

        # Missing keywords
        missing = missing_keywords(resume_clean, jd_clean)

        # -------- OUTPUT --------
        st.subheader("Match Score")
        st.success(f"{score}% match")

        # Graph
        st.subheader("Match Visualization")
        labels = ['Match', 'Gap']
        values = [score, 100 - score]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        st.pyplot(fig)

        # Missing Keywords
        st.subheader("Missing Keywords")
        st.write(", ".join(missing))

        # Top Keywords
        st.subheader("Top Keywords in Resume")
        st.write(", ".join(top_keywords(resume_clean)))

        # Suggestions
        st.subheader("Suggestions")

        if score > 80:
            st.success("Excellent match! Your resume is highly aligned with the job.")
        elif score > 60:
            st.warning("Good match. Try adding more relevant keywords and measurable achievements.")
        elif score > 40:
            st.info("Moderate match. Improve skills and include more relevant experience.")
        else:
            st.error("Low match. Focus on adding required skills and optimizing your resume.")

    else:
        st.error("Please upload resume and paste job description.")