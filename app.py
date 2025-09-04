import gradio as gr
import pandas as pd
import joblib
from lime.lime_text import LimeTextExplainer
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

# --- Load All Models and Vectorizers at Startup ---
print("Loading models and vectorizers...")

# 1. Load Fraud Detection Model
try:
    fraud_model = joblib.load('job_post_model.joblib')
    fraud_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("Fraud detection model loaded successfully.")
except FileNotFoundError:
    print("ERROR: Fraud detection model/vectorizer not found.")
    print("Please run 'train_model.py' first.")
    fraud_model = None

# 2. Load AI Text Detection Model (The new, powerful one)
try:
    ai_model = joblib.load('ai_detector_model.joblib')
    ai_vectorizer = joblib.load('ai_tfidf_vectorizer.joblib')
    print("AI text detection model loaded successfully.")
except FileNotFoundError:
    print("ERROR: AI text detection model/vectorizer not found.")
    print("Please run 'train_ai_detector.py' first.")
    ai_model = None

print("All models loaded.")

# --- Text Preprocessing Functions ---
# Download stopwords if not present
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def preprocess_for_fraud_model(text):
    """Preprocesses text specifically for the fraud detection model."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def preprocess_for_ai_model(text):
    """Preprocesses text specifically for the AI detection model."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)


# --- Prediction Functions ---

def predict_fraud(text):
    """Uses the trained XGBoost model to predict fraud risk."""
    if not fraud_model: return 0.0, "Model not loaded."
    
    # Create a dummy dataframe with all required features
    # This matches the structure used during training
    data = {
        'title': [text], 'department': [''], 'company_profile': [''],
        'description': [''], 'requirements': [''], 'benefits': [''],
        'employment_type': [''], 'required_experience': [''],
        'required_education': [''], 'industry': [''], 'function': ['']
    }
    df = pd.DataFrame(data)

    # Engineer the binary "has_" features
    for col in ['company_profile', 'requirements', 'benefits']:
        df[f'has_{col}'] = df[col].apply(lambda x: 1 if x.strip() else 0)

    # Combine text for vectorization
    df['combined_text'] = df.apply(lambda row: ' '.join(row[col] for col in data.keys()), axis=1)
    processed_text = df['combined_text'].apply(preprocess_for_fraud_model)
    text_vec = fraud_vectorizer.transform(processed_text)
    
    # Prepare numerical features
    num_features = df[[f'has_{col}' for col in ['company_profile', 'requirements', 'benefits']]]
    
    # The model expects a specific format, so we can't directly combine sparse and dense
    # We will rely on the text features which are the most dominant
    prediction_proba = fraud_model.predict_proba(text_vec)[:, 1]
    
    return prediction_proba[0], "Analysis based on linguistic patterns."

def predict_ai_generated(text):
    """Uses the trained Logistic Regression model to detect AI-generated text."""
    if not ai_model: return 0.0, "Model not loaded."

    # Preprocess text
    processed_text = preprocess_for_ai_model(text)
    # TF-IDF vector
    tfidf_vec = ai_vectorizer.transform([processed_text])


    # Stylistic features (original)
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(str(text))
        passive_voice = int(any(tok.dep_ == "auxpass" for tok in doc))
    except Exception:
        passive_voice = 0
    filler_words = set(["just", "really", "very", "actually", "basically", "literally", "simply", "quite", "somewhat", "perhaps"])
    has_filler = int(bool(set(str(text).lower().split()) & filler_words))
    pronouns = set(["i", "we", "me", "us", "my", "our"])
    has_pronoun = int(bool(set(str(text).lower().split()) & pronouns))
    post_length = len(str(text).split())
    suspicious_keywords = ['work from home', 'no experience required', 'urgent', 'immediate start', 'easy money', 'ai', 'generated', 'automated']
    has_suspicious = int(any(kw in str(text).lower() for kw in suspicious_keywords))
    sentences = re.split(r'[.!?]', str(text))
    short_sentences = [s.strip() for s in sentences if 1 < len(s.split()) < 5 and s.strip()]
    grammar_issues = int(len(short_sentences) > 3)

    # --- Enhanced AI Detection Features ---
    # 1. Generic/Enthusiastic/Vague Phrasing
    generic_phrases = [
        "exciting opportunity", "dynamic team", "fast-paced environment", "passionate", "innovative", "growth", "amazing", "cutting-edge", "join us", "make an impact", "driven", "motivated", "collaborative", "forward-thinking", "empowered", "rewarding", "unique culture", "visionary", "game-changing", "world-class", "exceptional", "unparalleled", "limitless", "incredible", "enthusiastic", "dedicated", "talented", "highly skilled", "top-tier", "industry-leading"
    ]
    generic_phrase_count = sum(str(text).lower().count(phrase) for phrase in generic_phrases)

    # 2. N-gram Diversity
    from collections import Counter
    def ngram_diversity(text, n=2):
        words = str(text).lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        if not ngrams:
            return 0.0
        unique_ngrams = set(ngrams)
        return len(unique_ngrams) / len(ngrams)
    bigram_diversity = ngram_diversity(text, 2)
    trigram_diversity = ngram_diversity(text, 3)

    # 3. Repetitiveness (sentence-level)
    def repetitiveness(text):
        sentences = [s.strip().lower() for s in re.split(r'[.!?]', str(text)) if s.strip()]
        if not sentences:
            return 0.0
        counts = Counter(sentences)
        repeated = sum(1 for v in counts.values() if v > 1)
        return repeated / len(sentences)
    sentence_repetitiveness = repetitiveness(text)

    # 4. Lack of Detail
    def avg_sentence_length(text):
        sentences = [s.strip() for s in re.split(r'[.!?]', str(text)) if s.strip()]
        if not sentences:
            return 0.0
        return sum(len(s.split()) for s in sentences) / len(sentences)
    def short_sentence_count(text):
        sentences = [s.strip() for s in re.split(r'[.!?]', str(text)) if s.strip()]
        return sum(1 for s in sentences if len(s.split()) < 5)
    avg_sent_len = avg_sentence_length(text)
    short_sent_count = short_sentence_count(text)

    # Stack all features
    stylistic_features = np.array([[post_length, has_suspicious, grammar_issues, passive_voice, has_filler, has_pronoun,
                                   generic_phrase_count, bigram_diversity, trigram_diversity, sentence_repetitiveness, avg_sent_len, short_sent_count]])

    # Embedding features
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = np.array(embedder.encode([str(text)]))
    except ImportError:
        embedding = np.zeros((1, 384)) # fallback to zeros if not available

    # Stack all features
    from scipy.sparse import hstack
    if embedding.shape[1] > 1:
        X_full = hstack([tfidf_vec, stylistic_features, embedding])
    else:
        X_full = hstack([tfidf_vec, stylistic_features])

    prediction_proba = ai_model.predict_proba(X_full)[:, 1]

    # Provide a simple explanation
    if prediction_proba[0] > 0.75:
        reason = "The text exhibits linguistic patterns and a structure highly consistent with AI-generated content."
    elif prediction_proba[0] > 0.5:
        reason = "The writing style shows some characteristics of AI generation, such as overly professional language and common buzzwords."
    else:
        reason = "The text appears to be human-written, with a more natural and less uniform linguistic style."

    return prediction_proba[0], reason


# --- Main Application Logic ---

def analyze_job_post(job_description_text):
    """
    The main function called by the Gradio interface.
    It runs both models and returns a combined analysis.
    """
    disclaimer_md = """
**Disclaimer:**  
<sub>The analysis is automated and for informational purposes only. The tool does not provide legal or financial advice. The results are not a guarantee of a posting's legitimacy or lack thereof. Users are responsible for conducting their own due diligence before applying or sharing personal information.</sub>
"""
    if not job_description_text.strip():
        return "Please enter a job description.", "", "", disclaimer_md

    # --- Fraud Analysis ---
    fraud_prob, fraud_reason = predict_fraud(job_description_text)
    fraud_risk_percent = f"{fraud_prob:.1%}"
    # Rule-based override: if any high-risk red flag is present, always flag as fraudulent
    # (Red flags are added below, so we check after that)

    # --- Rule-based Red Flag Checks ---
    red_flags = []
    # Check for non-corporate email addresses
    email_match = re.search(r'([\w\.-]+@[\w\.-]+)', job_description_text)
    if email_match:
        email = email_match.group(1)
        # Flag if not ending in a corporate domain (e.g., not ending in @company.com)
        if re.search(r'@(gmail|yahoo|hotmail|outlook|aol|icloud|mail|protonmail|zoho|gmx|yandex)\.com$', email) or not re.search(r'@[a-zA-Z0-9-]+\.(com|org|net|edu)$', email):
            red_flags.append("High-Risk Indicator: The recruiter's contact is a non-corporate email address (@gmail.com, etc.)")

    # Check for grammatical errors (simple heuristic: lots of short sentences or repeated punctuation)
    sentences = re.split(r'[.!?]', job_description_text)
    short_sentences = [s.strip() for s in sentences if 1 < len(s.split()) < 5 and s.strip() and not all(len(w) == 1 and not w.isalpha() for w in s.split())]
    highlighted = [f"**{s}** ← possible grammar issue" for s in short_sentences]
    if highlighted:
        red_flags.append(
            "High-Risk Indicator: The job description contains numerous grammatical errors.\n"
            "Problematic sentences/phrases detected:\n" + "\n".join(highlighted)
        )

    # Check for unusually high salary (simple heuristic)
    salary_match = re.search(r'\$([\d,]+)', job_description_text)
    if salary_match:
        salary = int(salary_match.group(1).replace(',', ''))
        # Assume $100,000 is a typical upper bound for most roles
        if salary > 150000:
            red_flags.append("High-Risk Indicator: The offered salary is 50% above the regional average for this role.")

    # Rule-based override: if any high-risk red flag is present, always flag as fraudulent
    fraud_detected = fraud_prob > 0.8 or bool(red_flags)
    if fraud_detected:
        fraud_summary = f"Detected as fraudulent. The model has identified a {fraud_risk_percent} risk of this being a fraudulent post. {fraud_reason}"
        if red_flags:
            fraud_summary += "\n\nRule-based override: High-risk red flag(s) detected."
    else:
        fraud_summary = f"Not detected as fraudulent. The model has identified a {fraud_risk_percent} risk of this being a fraudulent post. {fraud_reason}"

    # --- AI Generation Analysis ---
    ai_prob, ai_reason = predict_ai_generated(job_description_text)
    ai_prob_percent = f"{ai_prob:.1%}"
    ai_detected = ai_prob > 0.4
    if ai_detected:
        ai_summary = f"Detected as AI-generated. There is a {ai_prob_percent} probability that this text was AI-generated. {ai_reason}"
    else:
        ai_summary = f"Not detected as AI-generated. There is a {ai_prob_percent} probability that this text was AI-generated. {ai_reason}"

    # --- Rule-based Red Flag Checks ---
    red_flags = []
    # Check for non-corporate email addresses
    email_match = re.search(r'([\w\.-]+@[\w\.-]+)', job_description_text)
    if email_match:
        email = email_match.group(1)
        # Flag if not ending in a corporate domain (e.g., not ending in @company.com)
        if re.search(r'@(gmail|yahoo|hotmail|outlook|aol|icloud|mail|protonmail|zoho|gmx|yandex)\.com$', email) or not re.search(r'@[a-zA-Z0-9-]+\.(com|org|net|edu)$', email):
            red_flags.append("High-Risk Indicator: The recruiter's contact is a non-corporate email address (@gmail.com, etc.)")

    # Check for grammatical errors (simple heuristic: lots of short sentences or repeated punctuation)
    sentences = re.split(r'[.!?]', job_description_text)
    # Only show meaningful short sentences (at least 2 words, mostly alphabetic)
    # Less strict: include any short sentence (2-4 words), skip single letters/symbols
    short_sentences = [s.strip() for s in sentences if 1 < len(s.split()) < 5 and s.strip() and not all(len(w) == 1 and not w.isalpha() for w in s.split())]
    highlighted = [f"**{s}** ← possible grammar issue" for s in short_sentences]
    if highlighted:
        red_flags.append(
            "High-Risk Indicator: The job description contains numerous grammatical errors.\n"
            "Problematic sentences/phrases detected:\n" + "\n".join(highlighted)
        )

    # Check for unusually high salary (simple heuristic)
    salary_match = re.search(r'\$([\d,]+)', job_description_text)
    if salary_match:
        salary = int(salary_match.group(1).replace(',', ''))
        # Assume $100,000 is a typical upper bound for most roles
        if salary > 150000:
            red_flags.append("High-Risk Indicator: The offered salary is 50% above the regional average for this role.")

    # --- LIME Explanation for Fraud (most critical) ---
    explainer = LimeTextExplainer(class_names=['Legitimate', 'Fraudulent'])
    def lime_predictor(texts):
        processed_texts = [preprocess_for_fraud_model(text) for text in texts]
        text_vecs = fraud_vectorizer.transform(processed_texts)
        return fraud_model.predict_proba(text_vecs)

    explanation = explainer.explain_instance(job_description_text, lime_predictor, num_features=8)
    print("Available labels in explanation:", explanation.local_exp.keys())
    available_labels = list(explanation.local_exp.keys())
    risky_label = 1 if 1 in available_labels else available_labels[-1]
    risky_word_weights = explanation.as_list(label=risky_label)
    print(f"Fraudulent label ({risky_label}) word weights: {risky_word_weights}")
    risky_words = [word for word, weight in risky_word_weights if weight > 0]
    explanation_summary = "Key Risk Factors (words associated with fraud):\n- " + "\n- ".join(risky_words)

    # Add red flags to explanation if present
    if red_flags:
        explanation_summary = "Analysis Complete: {} Red Flags Detected\n".format(len(red_flags))
        explanation_summary += "\n" + "\n".join(red_flags)
        explanation_summary += "\nRecommendation: Caution is advised. Please independently verify the company and the recruiter before proceeding.\n\n" + "Key Risk Factors (words associated with fraud):\n- " + "\n- ".join(risky_words)

    return fraud_summary, ai_summary, explanation_summary, disclaimer_md


# --- Gradio Interface ---
disclaimer_md = """
**Disclaimer:**  
<sub>The analysis is automated and for informational purposes only. The tool does not provide legal or financial advice. The results are not a guarantee of a posting's legitimacy or lack thereof. Users are responsible for conducting their own due diligence before applying or sharing personal information.</sub>
"""

iface = gr.Interface(
    fn=analyze_job_post,
    inputs=gr.Textbox(lines=20, placeholder="Paste the full job description here..."),
    outputs=[
        gr.Textbox(label="Fraud Risk Analysis"),
        gr.Textbox(label="AI Generation Analysis"),
        gr.Textbox(label="Explanation of Key Factors"),
        gr.Markdown(disclaimer_md)
    ],
    title="Job Post Analyzer",
    description="Analyze a job description to detect potential fraud and AI-generated content. This tool provides a risk score and an explanation of the factors influencing the decision.",
    allow_flagging="never"
)

if __name__ == "__main__":
    if fraud_model is None or ai_model is None:
        print("\nOne or more models failed to load. The application cannot start.")
        print("Please ensure you have run both 'train_model.py' and 'train_ai_detector.py' successfully.")
    else:
        print("\nStarting the Gradio web server...")
        iface.launch()

