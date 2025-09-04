import pandas as pd
from tqdm import tqdm
import time
import requests
import json
import os

# --- Configuration ---
NUM_SAMPLES = 500  # Reduced for faster API calls during development
OUTPUT_FILE = 'data/ai_vs_human_jobs_prod.csv'
INPUT_FILE = 'data/fake_job_postings.csv'

# IMPORTANT: Set up your Gemini API Key in config.json
with open('config.json') as f:
    config = json.load(f)
API_KEY = config.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"

# --- Real AI Generation Function ---
def generate_ai_version(job_title, job_description, retries=3, backoff_factor=2):
    """
    Makes a real API call to the Gemini model to rewrite a job description.
    Includes error handling and exponential backoff.
    """
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        raise ValueError("GEMINI_API_KEY not set in config.json. Please add your actual Gemini API key.")

    prompt = f"""
    Rewrite the following job description for a '{job_title}' position to be more
    engaging and professional for a corporate careers page. Use strong, dynamic
    language and highlight the key responsibilities and qualifications.
    Do not change the core meaning or requirements of the job.

    Original Description:
    {job_description}
    """
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    for i in range(retries):
        try:
            response = requests.post(API_URL, json=payload, headers={'Content-Type': 'application/json'})
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            # Navigate the JSON response to get the text
            candidate = result.get('candidates', [{}])[0]
            content = candidate.get('content', {})
            parts = content.get('parts', [{}])[0]
            ai_text = parts.get('text', '')

            if ai_text:
                return ai_text
            else:
                # If the response structure is unexpected, log it and return a fallback
                print(f"Warning: Unexpected API response format: {result}")
                return f"AI Generated version for {job_title}" # Fallback

        except requests.exceptions.RequestException as e:
            print(f"API request failed (attempt {i+1}/{retries}): {e}")
            if i < retries - 1:
                time.sleep(backoff_factor ** i) # Exponential backoff
            else:
                print("Error: Max retries reached. Skipping this entry.")
                return f"Error generating AI version for {job_title}" # Final fallback

# --- Main Script Logic ---

print(f"Loading job posts from '{OUTPUT_FILE}'...")
df = pd.read_csv(OUTPUT_FILE)

# Only process rows with 'Error generating AI version for '

# Precompute indices to process
prefix = 'Error generating AI version for '
error_indices = [i for i, txt in enumerate(df['text']) if isinstance(txt, str) and txt.startswith(prefix)]
print(f"Found {len(error_indices)} entries to re-fetch.")

for idx in tqdm(error_indices, desc="Re-fetching AI versions"):
    error_text = df.at[idx, 'text']
    job_title = error_text[len(prefix):].strip()
    # Find a human-written post for this job title if possible
    human_row = df[(df['label'] == 0) & (df['text'].str.contains(job_title, case=False, na=False, regex=False))]
    if not human_row.empty:
        human_text = human_row.iloc[0]['text']
    else:
        human_text = job_title
    ai_text = generate_ai_version(job_title, human_text)
    # Only replace if a new version is successfully generated
    if ai_text and not ai_text.startswith(prefix):
        df.at[idx, 'text'] = ai_text
    # else: keep as is for future reruns
    time.sleep(1)

print(f"Saving updated dataset to '{OUTPUT_FILE}'...")
df.to_csv(OUTPUT_FILE, index=False)
print(f"Done. You can rerun this script to retry any remaining errors.")

