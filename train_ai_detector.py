import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords if you haven't already
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- Configuration ---
INPUT_FILE = 'data/ai_vs_human_jobs_prod.csv'
MODEL_OUTPUT_PATH = 'ai_detector_model.joblib'
VECTORIZER_OUTPUT_PATH = 'ai_tfidf_vectorizer.joblib'

# --- Text Preprocessing Function ---
def preprocess_text(text):
    """A simple preprocessing function to clean the text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\@\w+|\#','', text) # Remove mentions and hashtags
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

# --- Main Training Logic ---
if __name__ == "__main__":
    print(f"Loading and preprocessing data from '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)


    # Clean the text data
    df.dropna(subset=['text'], inplace=True)
    df['clean_text'] = df['text'].apply(preprocess_text)

    # Data balancing: undersample majority class if needed
    ai_count = df['label'].sum()
    human_count = len(df) - ai_count
    if human_count > 2 * ai_count:
        human_df = df[df['label'] == 0].sample(n=2*ai_count, random_state=42)
        ai_df = df[df['label'] == 1]
        df = pd.concat([human_df, ai_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Balanced dataset: {df['label'].value_counts().to_dict()}")

    # Feature: post_length
    df['post_length'] = df['text'].apply(lambda x: len(str(x).split()))

    # Feature: suspicious_keywords
    suspicious_keywords = ['work from home', 'no experience required', 'urgent', 'immediate start', 'easy money', 'AI', 'generated', 'automated']
    def has_suspicious_keywords(text):
        text = str(text).lower()
        return int(any(kw in text for kw in suspicious_keywords))
    df['suspicious_keywords'] = df['text'].apply(has_suspicious_keywords)

    # Feature: grammar_issues
    def grammar_issues(text):
        import re
        sentences = re.split(r'[.!?]', str(text))
        short_sentences = [s.strip() for s in sentences if 1 < len(s.split()) < 5 and s.strip()]
        return int(len(short_sentences) > 3)
    df['grammar_issues'] = df['text'].apply(grammar_issues)

    # Stylistic features
    # Passive voice detection (simple heuristic)
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        import os
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    def passive_voice(text):
        doc = nlp(str(text))
        return int(any(tok.dep_ == "auxpass" for tok in doc))
    df['passive_voice'] = df['text'].apply(passive_voice)

    # Filler words
    filler_words = set(["just", "really", "very", "actually", "basically", "literally", "simply", "quite", "somewhat", "perhaps"])
    def has_filler_words(text):
        words = set(str(text).lower().split())
        return int(bool(words & filler_words))
    df['filler_words'] = df['text'].apply(has_filler_words)

    # Personal pronouns
    pronouns = set(["i", "we", "me", "us", "my", "our"])
    def has_pronouns(text):
        words = set(str(text).lower().split())
        return int(bool(words & pronouns))
    df['personal_pronouns'] = df['text'].apply(has_pronouns)

    # --- Enhanced AI Detection Features ---
    # 1. Generic/Enthusiastic/Vague Phrasing
    generic_phrases = [
        "exciting opportunity", "dynamic team", "fast-paced environment", "passionate", "innovative", "growth", "amazing", "cutting-edge", "join us", "make an impact", "driven", "motivated", "collaborative", "forward-thinking", "empowered", "rewarding", "unique culture", "visionary", "game-changing", "world-class", "exceptional", "unparalleled", "limitless", "incredible", "enthusiastic", "dedicated", "talented", "highly skilled", "top-tier", "industry-leading"
    ]
    def count_generic_phrases(text):
        text = str(text).lower()
        return sum(text.count(phrase) for phrase in generic_phrases)
    df['generic_phrase_count'] = df['text'].apply(count_generic_phrases)

    # 2. N-gram Diversity
    from collections import Counter
    def ngram_diversity(text, n=2):
        words = str(text).lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        if not ngrams:
            return 0.0
        unique_ngrams = set(ngrams)
        return len(unique_ngrams) / len(ngrams)
    df['bigram_diversity'] = df['text'].apply(lambda x: ngram_diversity(x, 2))
    df['trigram_diversity'] = df['text'].apply(lambda x: ngram_diversity(x, 3))

    # 3. Repetitiveness (sentence-level)
    def repetitiveness(text):
        sentences = [s.strip().lower() for s in re.split(r'[.!?]', str(text)) if s.strip()]
        if not sentences:
            return 0.0
        counts = Counter(sentences)
        repeated = sum(1 for v in counts.values() if v > 1)
        return repeated / len(sentences)
    df['sentence_repetitiveness'] = df['text'].apply(repetitiveness)

    # 4. Lack of Detail
    def avg_sentence_length(text):
        sentences = [s.strip() for s in re.split(r'[.!?]', str(text)) if s.strip()]
        if not sentences:
            return 0.0
        return sum(len(s.split()) for s in sentences) / len(sentences)
    def short_sentence_count(text):
        sentences = [s.strip() for s in re.split(r'[.!?]', str(text)) if s.strip()]
        return sum(1 for s in sentences if len(s.split()) < 5)
    df['avg_sentence_length'] = df['text'].apply(avg_sentence_length)
    df['short_sentence_count'] = df['text'].apply(short_sentence_count)

    # Advanced embeddings (Sentence Transformers)
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        df['embedding'] = list(embedder.encode(df['text'].tolist(), show_progress_bar=True))
        embedding_dim = len(df['embedding'][0])
    except ImportError:
        print("Sentence Transformers not installed. Skipping embedding features.")
        df['embedding'] = [[] for _ in range(len(df))]
        embedding_dim = 0

    # Define features (X) and target (y)
    # Stack TF-IDF, stylistic, and embedding features
    X_tfidf = df['clean_text']
    y = df['label'] # 0 for human, 1 for AI

    # Vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf_vec = vectorizer.fit_transform(X_tfidf)

    # Stylistic features matrix
    stylistic_features = df[[
        'post_length', 'suspicious_keywords', 'grammar_issues', 'passive_voice', 'filler_words', 'personal_pronouns',
        'generic_phrase_count', 'bigram_diversity', 'trigram_diversity', 'sentence_repetitiveness', 'avg_sentence_length', 'short_sentence_count'
    ]].values

    # Embedding features matrix
    import numpy as np
    if embedding_dim > 0:
        embedding_features = np.array(df['embedding'].tolist())
        from scipy.sparse import hstack
        X_full = hstack([X_tfidf_vec, stylistic_features, embedding_features])
    else:
        from scipy.sparse import hstack
        X_full = hstack([X_tfidf_vec, stylistic_features])

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training the AI detection model (Logistic Regression with calibration)...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    base_model = LogisticRegression(solver='liblinear', random_state=42)
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    model.fit(X_train, y_train)

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF vectorization already done above; do not repeat here

    print("Training the AI detection model (Logistic Regression)...")
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human-Written', 'AI-Generated']))

    print("\nTop 20 most influential features (words) for AI detection:")
    feature_names = vectorizer.get_feature_names_out()
    from sklearn.calibration import CalibratedClassifierCV
    if isinstance(model, CalibratedClassifierCV):
        coefs = model.calibrated_classifiers_[0].base_estimator.coef_[0]
    else:
        coefs = model.coef_[0]
    tfidf_size = len(feature_names)
    top_pos_idx = coefs.argsort()[-20:][::-1]
    top_neg_idx = coefs.argsort()[:20]
    print("Most AI-indicative words:")
    for idx in top_pos_idx:
        if idx < tfidf_size:
            print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")
        else:
            print(f"  [extra feature {idx-tfidf_size+1}]: {coefs[idx]:.4f}")
    print("\nMost human-indicative words:")
    for idx in top_neg_idx:
        if idx < tfidf_size:
            print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")
        else:
            print(f"  [extra feature {idx-tfidf_size+1}]: {coefs[idx]:.4f}")

    print("\nSaving the trained model and vectorizer...")
    joblib.dump(model, MODEL_OUTPUT_PATH)
    joblib.dump(vectorizer, VECTORIZER_OUTPUT_PATH)

    print(f"Successfully saved model to '{MODEL_OUTPUT_PATH}'")
    print(f"Successfully saved vectorizer to '{VECTORIZER_OUTPUT_PATH}'")
    print("\nTraining complete. You are now ready for the final integration step.")
