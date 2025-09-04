
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from app import analyze_job_post

def run_tests(csv_path):
    df = pd.read_csv(csv_path)
    results = []
    from app import predict_fraud, predict_ai_generated
    for idx, row in df.iterrows():
        desc = row['description']
        expected_ai = row['label_ai']
        expected_fraud = row['label_fraud']
        expected_flags = str(row['expected_red_flags']).split(';') if pd.notna(row['expected_red_flags']) else []
        fraud_out, ai_out, explanation, _ = analyze_job_post(desc)
        # Get raw probabilities
        fraud_prob, _ = predict_fraud(desc)
        ai_prob, _ = predict_ai_generated(desc)
        # Check red flags
        flags_found = [flag for flag in expected_flags if flag and flag in explanation]
        # Check AI label using threshold
        ai_detected = ai_prob > 0.4
        # Rule-based override for fraud: flag as fraud if probability > 0.8 or any high-risk red flag is present
        fraud_detected = fraud_prob > 0.8 or len(flags_found) > 0
        results.append({
            'description': desc,
            'expected_ai': expected_ai,
            'ai_detected': ai_detected,
            'expected_fraud': expected_fraud,
            'fraud_detected': fraud_detected,
            'expected_flags': expected_flags,
            'flags_found': flags_found,
            'fraud_prob': fraud_prob,
            'ai_prob': ai_prob,
            'pass': (ai_detected == bool(expected_ai)) and (fraud_detected == bool(expected_fraud)) and (len(flags_found) == len(expected_flags))
        })
    return results

if __name__ == "__main__":
    test_results = run_tests("test/test_cases.csv")
    for i, res in enumerate(test_results):
        print(f"Test {i+1}: {'PASS' if res['pass'] else 'FAIL'}")
        print(f"  Description: {res['description']}")
        print(f"  Expected AI: {res['expected_ai']} | Detected: {res['ai_detected']}")
        print(f"  Expected Fraud: {res['expected_fraud']} | Detected: {res['fraud_detected']}")
        print(f"  Expected Flags: {res['expected_flags']} | Found: {res['flags_found']}")
        print(f"  Fraud Probability: {res['fraud_prob']:.4f}")
        print(f"  AI Probability: {res['ai_prob']:.4f}")
        print()
