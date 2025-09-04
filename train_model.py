import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import xgboost as xgb
import joblib
import logging
import warnings
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("train_model.log")
    ]
)
logger = logging.getLogger(__name__)

# Log warnings to file
def log_warning(message, category, filename, lineno, file=None, line=None):
    logger.warning(f'{category.__name__}: {message} ({filename}:{lineno})')
warnings.showwarning = log_warning

# --- 1. Download NLTK data (only needs to be done once) ---
logger.info("Downloading NLTK stopwords...")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
logger.info("Download complete.")

# --- 2. Text Preprocessing Function ---
def preprocess_text(text):
    """Cleans and preprocesses the input text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words] # Remove stopwords
    return " ".join(tokens)


# --- 3. Load and Feature Engineer Data ---
logger.info("Loading and preparing data...")
df = pd.read_csv('data/fake_job_postings.csv')

# Data balancing: undersample legitimate if needed
fraud_count = df['fraudulent'].sum()
legit_count = len(df) - fraud_count
if legit_count > 2 * fraud_count:
    legit_df = df[df['fraudulent'] == 0].sample(n=2*fraud_count, random_state=42)
    fraud_df = df[df['fraudulent'] == 1]
    df = pd.concat([legit_df, fraud_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Balanced dataset: {df['fraudulent'].value_counts().to_dict()}")

# Feature: has_corporate_email
def is_corporate_email(text):
    import re
    match = re.search(r'([\w\.-]+@([\w\.-]+))', str(text))
    if match:
        domain = match.group(2)
        return int(not re.search(r'(gmail|yahoo|hotmail|outlook|aol|icloud|mail|protonmail|zoho|gmx|yandex)\.com$', domain))
    return 0
df['has_corporate_email'] = df['description'].apply(is_corporate_email)

# Feature: high_salary
def is_high_salary(text):
    import re
    match = re.search(r'\$([\d,]+)', str(text))
    if match:
        salary = int(match.group(1).replace(',', ''))
        return int(salary > 150000)
    return 0
df['high_salary'] = df['description'].apply(is_high_salary)

# Feature: grammar_issues
def grammar_issues(text):
    import re
    sentences = re.split(r'[.!?]', str(text))
    short_sentences = [s.strip() for s in sentences if 1 < len(s.split()) < 5 and s.strip()]
    return int(len(short_sentences) > 3)
df['grammar_issues'] = df['description'].apply(grammar_issues)

# Feature: suspicious_keywords
suspicious_keywords = ['work from home', 'no experience required', 'urgent', 'immediate start', 'easy money']
def has_suspicious_keywords(text):
    text = str(text).lower()
    return int(any(kw in text for kw in suspicious_keywords))
df['suspicious_keywords'] = df['description'].apply(has_suspicious_keywords)

# Feature: post_length
df['post_length'] = df['description'].apply(lambda x: len(str(x).split()))

# Existing binary "has_" features
for col in ['company_profile', 'requirements', 'benefits']:
    df[f'has_{col}'] = df[col].notna().astype(int)

# Combine all text fields into a single, comprehensive text feature
text_cols = ['title', 'department', 'company_profile', 'description', 'requirements', 'benefits', 'required_experience', 'required_education', 'industry', 'function']
df['full_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)

# Apply preprocessing
df['clean_text'] = df['full_text'].apply(preprocess_text)

# Define features (X) and target (y)
X_text = df['clean_text']
y = df['fraudulent']
logger.info("Data preparation complete.")

# --- 4. Split Data and Vectorize Text ---
logger.info("Splitting data and vectorizing text...")
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
logger.info("Vectorization complete.")


# --- 5. Hyperparameter Tuning with Grid Search ---
logger.info("Starting hyperparameter tuning with GridSearchCV...")
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
param_grid = {
    'n_estimators': [100],
    'learning_rate': [0.1],
    'max_depth': [5, 7],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'scale_pos_weight': [scale_pos_weight]
}
base_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(base_model, param_grid, scoring='f1', cv=cv, verbose=1, n_jobs=1)
grid_search.fit(X_train_vec, y_train)
logger.info(f"Best parameters found: {grid_search.best_params_}")
model = grid_search.best_estimator_
logger.info("Model training complete.")

# --- 6. Evaluate the Model ---
logger.info("Evaluating model performance...")
y_pred = model.predict(X_test_vec)

logger.info("--- Classification Report ---\n" + str(classification_report(y_test, y_pred)))
logger.info("--- Confusion Matrix ---\n" + str(confusion_matrix(y_test, y_pred)))
logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
logger.info("Note: For this problem, 'recall' for class 1 (fraudulent) is the most important metric!")


# --- 7. Save the Model and Vectorizer ---
logger.info("Saving model and vectorizer to disk...")
joblib.dump(model, 'job_post_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
logger.info("Artifacts saved successfully. You are now ready to run the chatbot app.")

