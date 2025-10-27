# 🎓 NLP-Powered Resume Analyzer

A hybrid Streamlit web application that performs high-speed "classical" NLP analysis and on-demand "modern" LLM analysis for resumes.

This project uses a multi-layered approach to demonstrate and compare different NLP techniques, from keyword matching and simple ML models to large language model (LLM)-based extraction.

## 🚀 App Showcase

Here is a gallery of the app's core features.

| Uploading a Resume | The Main Dashboard |
| :---: | :---: |
| ![Upload Page](assets/uploadpage.png) | ![Dashboard](assets/llm.png) |
| **Classical NLP & ML Analysis** | **Job Skill Matcher** |
| ![ML Analysis](assets/ml-analysis.png) | ![Job Matcher](assets/jobmatcher.png) |
| **NLP Pipeline Visualization** | **Full Text Extraction** |
| ![NLP Pipeline](assets/nlppipeline.png) | ![Full Text](assets/extractedtext.png) |

## ✨ Features

This app is built around a fast, multi-tab interface:

* **⚡ Instant Dashboard:** The app loads instantly, providing immediate analysis using "classical" NLP:
    * **Contact Info:** Extracted using Regex and spaCy's Named Entity Recognition (NER).
    * **Sentiment Analysis:** Uses VADER to score the resume's tone.
    * **Job Classification:** Predicts the job category (e.g., "Data Scientist") using a custom-trained **TF-IDF + Logistic Regression** model.
    * **Rule-Based Skills:** Extracts all skills from a predefined list using **spaCy's `Matcher`**.

* **🤖 On-Demand AI Deep Dive:** To solve the 3-4 minute LLM loading time, all slow AI tasks are moved to this tab. The user clicks a button to run the analysis on demand.
    * Uses **Ollama (llama3:latest)** to generate a professional summary.
    * Performs advanced skill extraction using the LLM's contextual understanding.
    * Uses a **single master-prompt** to get all AI data in one call, improving speed.

* **⚖️ AI vs. Rules Skill Comparison:** A dedicated tab to visually compare the skills found by the **spaCy `Matcher`** (fast, keyword-based) against the skills found by the **Ollama LLM** (slow, context-aware).

* **📈 Job Skill Matcher:**
    * Lets you paste a job description.
    * Extracts skills from the JD using the fast, rule-based spaCy `Matcher`.
    * Compares the JD skills to the resume's skills and calculates a **match percentage**.

* **⚙️ NLP Pipeline:** A visual, academic breakdown of key NLP concepts, including:
    * **Tokenization** (with `nltk`)
    * **Stop-Word Removal**
    * **Lemmatization**
    * **NER Visualization** (showing `PERSON`, `ORG`, `DATE`, etc.)

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **NLP (Classical):** spaCy, NLTK (VADER), Scikit-learn
* **NLP (Modern):** Ollama (running `llama3:latest`)
* **Text Extraction:** `PyMuPDF` (for .pdf), `python-docx` (for .docx)
* **Utilities:** Pandas

## ⚙️ How to Run Locally

**1. Prerequisites:**
* You must have **Ollama** installed and running. [Download it here](https://ollama.com/).
* Make sure you have Python 3.9+ installed.

**2. Clone & Setup:**
```bash
# Clone the repository
git clone [https://github.com/RAHULBONEY/resume_analyser.git](https://github.com/RAHULBONEY/resume_analyser.git)
cd resume_analyser

# Create and activate a virtual environment
# Note: This project uses 'venv', not '.venv'
python -m venv venv
.\venv\Scripts\activate# Install all required packages
pip install -r requirements.txt

# Download the spaCy model
python -m spacy download en_core_web_sm# Pull the LLM (this may take a while)
ollama pull llama3:latest

# Train the local ML classifier
python train_model.py# Make sure Ollama is running in the background!
streamlit run app.py
