

# Job Post Analyzer


This project uses machine learning and AI to detect fraudulent and AI-generated job postings. It features:
- Advanced feature engineering for robust AI detection (generic/vague phrasing, n-gram diversity, repetitiveness, lack of detail)
- Model training for fraud and AI detection
- Gemini API integration for generating and regenerating synthetic job posts (with error handling and efficient retry logic)
- LIME explanations (risk factors only)
- Gradio web app for interactive analysis


## Features
- Fraud detection using XGBoost (threshold: 0.8, plus rule-based red flags)
- AI-generated text detection using Logistic Regression (threshold: 0.4)
- Enhanced feature engineering for AI detection:
	- Generic/enthusiastic/vague phrasing
	- N-gram diversity (bigram/trigram)
	- Sentence repetitiveness
	- Lack of detail (short sentences, average sentence length)
- LIME explanations (shows only risk factors for fraud)
- Gemini API-powered dataset generation and error entry regeneration (`generate_ai_dataset.py`)
	- Only processes rows with failed AI generations ("Error generating AI version for ...")
	- Efficient in-place updates, skips non-error rows, rerunnable for remaining errors
- Automated test runner (`test/test_app.py`) for validating model and rule-based logic
- Gradio chatbot interface for interactive analysis


## Workflow & Usage
1. Place your initial dataset in `data/fake_job_postings.csv`.
2. (Optional) Generate or regenerate AI/human mixed dataset:
	- Add your Gemini API key to `config.json`.
	- Run `python generate_ai_dataset.py`.
	  - Only error entries ("Error generating AI version for ...") will be retried and updated in-place.
	  - Script is rerunnable: skips non-error rows, only processes failed generations.
3. Train the fraud detection model:
	- `python train_model.py`
4. Train the AI detector model (with enhanced features):
	- `python train_ai_detector.py`
5. Run the Gradio app for interactive analysis:
	- `python app.py`
6. (Optional) Run automated tests:
	- `python test/test_app.py`


## Requirements
- Python 3.10+
- pandas, numpy, nltk, scikit-learn, xgboost, joblib
- lime (for explanations)
- gradio (for web app)
- tqdm (for progress bars)
- requests (for Gemini API)
- spacy (for passive voice detection)
- sentence-transformers (for embedding features, optional)


## Gemini API Setup & Dataset Regeneration
1. Get your Gemini API key and add it to `config.json`:
	```json
	{ "GEMINI_API_KEY": "YOUR_API_KEY_HERE" }
	```
2. Run `generate_ai_dataset.py` to create or regenerate synthetic job posts.
	- The script will only process rows with failed AI generations ("Error generating AI version for ...").
	- Updates are made in-place and the script can be rerun to retry remaining errors.


## Model Training
```bash
python train_model.py         # Fraud detection model
python train_ai_detector.py   # AI detection model (with enhanced features)
```

## Run the App
```bash
python app.py
```

## Automated Testing
```bash
python test/test_app.py
```
Tests validate:
- AI detection threshold (0.4)
- Fraud detection threshold (0.8 + red flags)
- Red flag explanations


## Output
The app displays:
- Fraud risk analysis (with probability and rule-based red flags)
- AI generation analysis (with probability and explanation)
- Key risk factors (words associated with fraud)



## Disclaimer
Every analysis your AI provides must be accompanied by a clear, visible disclaimer. This is your most important legal safeguard. It should state that:
- The analysis is automated and for informational purposes only.
- The tool does not provide legal or financial advice.
- The results are not a guarantee of a posting's legitimacy or lack thereof.
- Users are responsible for conducting their own due diligence before applying or sharing personal information.


## License
MIT
