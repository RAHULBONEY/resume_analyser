import streamlit as st
import fitz  # PyMuPDF
import docx  # python-docx
import re
import spacy
from spacy.matcher import Matcher
from spacy import displacy 
import pickle
import ollama
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import json

st.set_page_config(page_title="🎓 NLP Resume Analyzer", layout="wide")

MODEL_DIR = "trained_model"
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
OLLAMA_MODEL = "llama3:latest" 

@st.cache_resource
def get_spacy_matcher(_nlp):
    SKILLS_LIST = [
        'python', 'java', 'c++', 'javascript', 'sql', 'react', 'vue', 'angular',
        'nodejs', 'express', 'django', 'flask', 'fastapi', 'spring boot',
        'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
        'pandas', 'numpy', 'nlp', 'llms', 'aws', 'azure', 'gcp', 'docker', 'kubernetes',
        'git', 'github', 'jira', 'agile', 'scrum', 'figma', 'adobe xd', 'power bi', 'tableau',
        'matplotlib', 'seaborn', 'apis', 'api', 'data preprocessing', 'feature engineering',
        'model evaluation'
    ]
    matcher = Matcher(_nlp.vocab)
    patterns = [[{"LOWER": skill_part} for skill_part in skill.split()] for skill in SKILLS_LIST]
    matcher.add("SKILLS", patterns)
    return matcher

@st.cache_resource
def load_assets():
    with st.spinner("Loading NLP assets (this happens once)..."):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        st.error(f"Model files not found. Please run 'python train_model.py' first.")
        st.stop()
        
    with open(MODEL_PATH, 'rb') as f_model:
        model = pickle.load(f_model)
    with open(VECTORIZER_PATH, 'rb') as f_vec:
        vectorizer = pickle.load(f_vec)
    
    try:
        nlp = spacy.load('en_core_web_sm')
    except IOError:
        st.info("Downloading spaCy model 'en_core_web_sm'...")
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
        
    skill_matcher = get_spacy_matcher(nlp)
    vader_analyzer = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    return model, vectorizer, nlp, vader_analyzer, stop_words, lemmatizer, skill_matcher

model, vectorizer, nlp, vader_analyzer, stop_words, lemmatizer, skill_matcher = load_assets()

def extract_text(uploaded_file):
    try:
        file_bytes = uploaded_file.getvalue()
        if uploaded_file.type == "application/pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            return text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text = "\n".join(para.text for para in doc.paragraphs)
            return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

def extract_contact_info(text):
    info = {
        "name": "Not Found",
        "email": "Not Found",
        "phone": "Not Found"
    }
    
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        info["email"] = email_match.group(0)

    phone_match = re.search(r'(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)?(\d{3}[\s.-]?)(\d{4})', text)
    if phone_match:
        info["phone"] = phone_match.group(0).strip()

    doc = nlp(text[:500])
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            info["name"] = ent.text
            break
            
    return info

def preprocess_for_ml(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_json_from_ollama(prompt, text_chunk):
    system_prompt = f"""
    You are an expert HR data extraction assistant.
    Extract the required information from the resume text provided.
    Respond ONLY with a valid JSON object.
    
    Resume Text:
    ---
    {text_chunk}
    ---
    
    JSON Request:
    {prompt}
    """
    
    try:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=system_prompt,
            format="json", 
            options={"temperature": 0.0}
        )
        response_text = response.get('response', '{}')
        match = re.search(r'\{.*\}', response_text, re.DOTALL) 
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                st.error("AI returned malformed JSON. Could not parse.")
                return {}
        else:
            st.error(f"AI response did not contain a JSON object. Response: {response_text[:100]}...")
            return {}
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}")
        return {}

def extract_skills_with_matcher(text_doc):
    matches = skill_matcher(text_doc)
    skills = set(text_doc[start:end].text.lower() for _, start, end in matches)
    return sorted(list(skills)) if skills else []

def get_tokens(text):
    return nltk.word_tokenize(text)

def get_tokens_no_stop(tokens):
    return [token for token in tokens if token.lower() not in stop_words]

def get_lemmas(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

st.title("🎓 NLP Resume Analyzer")
st.markdown("This app performs instant analysis using classical NLP (Regex, spaCy) and can optionally run slower, deep AI analysis on demand.")

if "ai_results" not in st.session_state:
    st.session_state.ai_results = None

uploaded_file = st.file_uploader("Upload your resume (.pdf or .docx)", type=["pdf", "docx"])

if uploaded_file is not None:
    raw_text = extract_text(uploaded_file)
    
    if raw_text and len(raw_text) > 50:
        doc_for_spacy = nlp(raw_text)
        
        
        contact_info = extract_contact_info(raw_text)
        matcher_resume_skills = extract_skills_with_matcher(doc_for_spacy)
        sentiment_scores = vader_analyzer.polarity_scores(raw_text)
        compound_sentiment = sentiment_scores['compound']
        
        processed_text = preprocess_for_ml(raw_text)
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]
        prediction_proba = model.predict_proba(text_vectorized).max()
        
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard (Instant)", 
            "🤖 AI Deep Dive (Slow)",
            "📈 Job Matcher", 
            "⚙️ NLP Pipeline",
            "📜 Full Text"
        ])

       
        with tab1:
            st.subheader("Candidate Overview")
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Name", contact_info['name'])
                col2.metric("Email", contact_info['email'])
                col3.metric("Phone", contact_info['phone'])
            
            st.subheader("Classical NLP & ML Analysis")
            with st.container(border=True):
                c1, c2 = st.columns(2)
                c1.metric(label="VADER Sentiment (Lexicon-Based)", value=f"{compound_sentiment:.3f}")
                c2.metric(label="Predicted Job Category (TF-IDF Model)", value=prediction, delta=f"Confidence: {prediction_proba:.1%}")

            st.subheader("Rule-Based Skills (spaCy Matcher)")
            with st.container(border=True):
                if matcher_resume_skills:
                    st.markdown(f"Found **{len(matcher_resume_skills)}** skills:")
                    st.markdown('<div style="display: flex; flex-wrap: wrap; gap: 5px;">' + 
                                ''.join(f'<span style="background-color: #E8E8E8; color: #333; padding: 5px 10px; border-radius: 15px; font-size: 14px;">{skill}</span>' for skill in matcher_resume_skills) +
                                '</div>', unsafe_allow_html=True)
                else:
                    st.warning("No skills found by the rule-based matcher.")

        
        with tab2:
            st.subheader("AI-Powered Analysis (Ollama)")
            st.markdown("This analysis is slow (1-2 mins) as it runs a large language model. Click the button to proceed.")
            
            if st.button("🚀 Run AI Analysis"):
                text_chunk_for_llm = raw_text[:4000]
                with st.spinner("Analyzing with AI... This will take a moment."):
                    master_prompt = """
                    {
                      "ai_summary": "A 3-bullet point summary of the candidate's key achievements.",
                      "ai_skills": ["skill 1", "skill 2", "technical skill 3"]
                    }
                    """
                    ai_data = get_json_from_ollama(master_prompt, text_chunk_for_llm)
                    st.session_state.ai_results = ai_data
                    st.rerun()
            
            if st.session_state.ai_results:
                st.subheader("AI-Generated Summary")
                with st.container(border=True):
                    ai_summary = st.session_state.ai_results.get("ai_summary", "AI summary generation failed.")
                    if isinstance(ai_summary, list):
                        st.markdown("\n".join(f"- {s}" for s in ai_summary))
                    else:
                        st.markdown(ai_summary)

                st.subheader("AI-Extracted Skills")
                with st.container(border=True):
                    ai_skills = st.session_state.ai_results.get("ai_skills", [])
                    if ai_skills:
                        st.markdown(f"Found **{len(ai_skills)}** skills:")
                        st.markdown('<div style="display: flex; flex-wrap: wrap; gap: 5px;">' + 
                                    ''.join(f'<span style="background-color: #0072B2; color: white; padding: 5px 10px; border-radius: 15px; font-size: 14px;">{skill}</span>' for skill in ai_skills) +
                                    '</div>', unsafe_allow_html=True)
                    else:
                        st.warning("AI skill extraction failed or found no skills.")
            
       
        with tab3:
            st.subheader("Resume vs. Job Description")
            st.markdown("This tool uses the **fast, rule-based skills** from the dashboard to find a match.")

            job_desc_text = st.text_area("Job Description", height=200, key="job_desc", value=st.session_state.get("job_desc_text", ""))
            
            if st.button("Analyze Match"):
                st.session_state["job_desc_text"] = job_desc_text 
                
                if not job_desc_text:
                    st.error("Please paste a job description first.")
                elif not matcher_resume_skills:
                     st.error("Cannot perform match: No skills were extracted from the resume.")
                else:
                    with st.spinner("Extracting skills from job description..."):
                        jd_doc = nlp(job_desc_text)
                        job_skills = set(extract_skills_with_matcher(jd_doc))
                    
                    if not job_skills:
                        st.warning("Could not extract any skills from the job description.")
                    else:
                        resume_skills_lower = set(s.lower() for s in matcher_resume_skills) 
                        matching_skills = resume_skills_lower.intersection(job_skills)
                        missing_skills = job_skills.difference(resume_skills_lower)
                        match_percentage = (len(matching_skills) / len(job_skills)) * 100 if job_skills else 0
                        
                        st.metric(label="Job Skill Match Score", value=f"{match_percentage:.1f}%")
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.success(f"✅ Matching Skills ({len(matching_skills)})")
                            st.dataframe(pd.DataFrame(list(matching_skills), columns=["Skill"]), use_container_width=True)
                        with c2:
                            st.error(f"❌ Missing Skills ({len(missing_skills)})")
                            st.dataframe(pd.DataFrame(list(missing_skills), columns=["Skill"]), use_container_width=True)
        
       
        with tab4:
            st.subheader("NLP Preprocessing Pipeline")
            st.markdown("This demonstrates the classical NLP steps used for our models.")
            
            sample_text = (raw_text.split('.')[0] + ".")
            st.code(f"Original Text: {sample_text}", language=None)
            
            tokens = get_tokens(sample_text)
            st.dataframe(pd.DataFrame(tokens, columns=["1. Tokenization"]), use_container_width=True)
            
            tokens_no_stop = get_tokens_no_stop(tokens)
            st.dataframe(pd.DataFrame(tokens_no_stop, columns=["2. Stop-Word Removal"]), use_container_width=True)
            
            lemmas = get_lemmas(tokens_no_stop)
            st.dataframe(pd.DataFrame(lemmas, columns=["3. Lemmatization (Root Word)"]), use_container_width=True)

            st.divider()

            st.subheader("Named Entity Recognition (NER)")
            st.markdown("This chart shows what the pre-trained `spaCy` model identifies as entities.")
            doc_display = nlp(raw_text[:1500])
            colors = {"PERSON": "#FFADAD", "ORG": "#ADD8E6", "GPE": "#90EE90", "DATE": "#FFD6A5"}
            options = {"ents": ["PERSON", "ORG", "GPE", "DATE", "PRODUCT", "MONEY"], "colors": colors}
            html = displacy.render(doc_display, style="ent", options=options, jupyter=False)
            st.write(html, unsafe_allow_html=True)
            
       
        with tab5:
            st.subheader("Full Extracted Text")
            st.text_area("Resume Text", raw_text, height=500, key="full_text")
            
    else:
        st.error("Failed to extract significant text from the file. It might be empty or corrupted.")