"""
Train the ToolFunctionSelector model.
RUN using python scripts/train_tool_selector.py
Results Saved to data/tool_selector/selector.pkl.
"""
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

TRAINING_DATA = [
    ("What are the indications for metformin?", "search_drug_label"),
    ("Show me the drug label for lisinopril", "search_drug_label"),
    ("What does the FDA label say about atorvastatin?", "search_drug_label"),
    ("Dosage information for amoxicillin", "search_drug_label"),
    ("What are the contraindications of warfarin?", "search_drug_label"),
    ("Give me the prescribing information for ibuprofen", "search_drug_label"),
    ("What is sertraline used for?", "search_drug_label"),
    ("Tell me about the warnings on the label for prednisone", "search_drug_label"),
    ("What are the approved uses of omeprazole?", "search_drug_label"),
    ("FDA drug label information for furosemide", "search_drug_label"),
    ("Full prescribing details for levothyroxine", "search_drug_label"),

    ("What adverse events have been reported for aspirin?", "search_drug_adverse_events"),
    ("Show me side effects filed with the FDA for metformin", "search_drug_adverse_events"),
    ("FDA adverse event reports for clopidogrel", "search_drug_adverse_events"),
    ("What are the reported reactions for simvastatin?", "search_drug_adverse_events"),
    ("List adverse event reports for ibuprofen", "search_drug_adverse_events"),
    ("Search for FDA FAERS reports on atorvastatin", "search_drug_adverse_events"),
    ("What harmful reactions have been reported for warfarin?", "search_drug_adverse_events"),
    ("Drug safety reports for acetaminophen", "search_drug_adverse_events"),
    ("What adverse reactions are reported for lisinopril?", "search_drug_adverse_events"),
    ("Give me FAERS data on sertraline", "search_drug_adverse_events"),
    ("Patient safety events related to furosemide", "search_drug_adverse_events"),

    ("Calculate BMI for a patient weighing 70 kg and 1.75 m tall", "calculate_bmi"),
    ("What is the BMI of someone who is 80 kg and 1.80 m?", "calculate_bmi"),
    ("BMI calculation for 55 kg weight and height of 1.60 m", "calculate_bmi"),
    ("Compute body mass index: weight 90 kg, height 1.85 m", "calculate_bmi"),
    ("Is a person 65 kg and 1.70 m overweight?", "calculate_bmi"),
    ("Body mass index for 100 kg and 1.65 m tall", "calculate_bmi"),
    ("Find BMI given 72 kg and 1.78 m", "calculate_bmi"),
    ("What BMI category is 50 kg at 1.55 m?", "calculate_bmi"),
    ("Calculate body mass index for weight 85 kg height 1.72 m", "calculate_bmi"),
    ("BMI for 68 kg 1.68 m patient", "calculate_bmi"),
    ("Weight 95 kg height 1.90 m what is BMI?", "calculate_bmi"),

    ("Calculate creatinine clearance for a 65 year old male, 80 kg, creatinine 1.2", "calculate_creatinine_clearance"),
    ("Cockcroft-Gault CrCl: age 70, weight 75 kg, serum creatinine 1.5", "calculate_creatinine_clearance"),
    ("What is the creatinine clearance for a 55 yo woman weighing 60 kg with creatinine 0.9?", "calculate_creatinine_clearance"),
    ("Estimate kidney function: 80 year old male 70 kg creatinine 2.0", "calculate_creatinine_clearance"),
    ("CrCl calculation for 45 year old female 65 kg serum Cr 1.1", "calculate_creatinine_clearance"),
    ("Renal function estimate using Cockcroft-Gault: 72 yr male 85 kg Cr 1.8", "calculate_creatinine_clearance"),
    ("Calculate renal clearance for patient age 60 weight 90 kg creatinine 1.4", "calculate_creatinine_clearance"),
    ("How well are the kidneys functioning? Age 50, 55 kg female, serum creatinine 0.8", "calculate_creatinine_clearance"),
    ("CKD staging via GFR: 68 year old 78 kg male creatinine 2.5", "calculate_creatinine_clearance"),
    ("Glomerular filtration for 75 yo woman 62 kg creatinine 1.0", "calculate_creatinine_clearance"),
    ("Compute creatinine clearance age 40 male weight 100 kg Cr 1.3", "calculate_creatinine_clearance"),

    ("What is the ideal body weight for a male patient 175 cm tall?", "calculate_ideal_body_weight"),
    ("Calculate IBW for a female 160 cm", "calculate_ideal_body_weight"),
    ("Devine formula for ideal weight: 180 cm male", "calculate_ideal_body_weight"),
    ("What should a woman at 165 cm weigh ideally?", "calculate_ideal_body_weight"),
    ("Ideal body weight for 170 cm man", "calculate_ideal_body_weight"),
    ("IBW calculation for a 155 cm female patient", "calculate_ideal_body_weight"),
    ("Compute ideal weight using Devine for 190 cm male", "calculate_ideal_body_weight"),
    ("What is the target body weight for a 162 cm woman?", "calculate_ideal_body_weight"),
    ("Determine ideal body weight for height 178 cm male", "calculate_ideal_body_weight"),
    ("IBW for 168 cm female", "calculate_ideal_body_weight"),
    ("Expected body weight for 185 cm man", "calculate_ideal_body_weight"),

    ("Calculate pediatric dose for a child weighing 20 kg, adult dose is 400 mg", "calculate_pediatric_dose"),
    ("What is the Clark's rule dose for a 25 kg child if adult dose is 500 mg?", "calculate_pediatric_dose"),
    ("Pediatric dosing for 15 kg child with adult dose of 250 mg", "calculate_pediatric_dose"),
    ("Child dose calculation: adult dose 300 mg, child weight 30 kg", "calculate_pediatric_dose"),
    ("Dose for a 10 kg pediatric patient when the standard adult dose is 200 mg", "calculate_pediatric_dose"),
    ("Compute pediatric dose using Clark's rule: child 22 kg, adult 600 mg", "calculate_pediatric_dose"),
    ("How much medication for a 35 kg child if adult dose is 1000 mg?", "calculate_pediatric_dose"),
    ("Weight-based dose for child 18 kg adult dose 450 mg", "calculate_pediatric_dose"),
    ("Pediatric weight-adjusted dose: 12 kg child, adult dose 150 mg", "calculate_pediatric_dose"),
    ("Child weighing 28 kg, adult dose 800 mg, what is pediatric dose?", "calculate_pediatric_dose"),
    ("Dosing calculation for 40 kg pediatric patient adult dose 1200 mg", "calculate_pediatric_dose"),
]

TEST_QUERIES = [
    ("Prescribing information for aspirin", "search_drug_label"),
    ("FDA safety events for digoxin", "search_drug_adverse_events"),
    ("BMI for 75 kg and 1.80 m", "calculate_bmi"),
    ("CrCl for 72 year old female 58 kg creatinine 1.3", "calculate_creatinine_clearance"),
    ("IBW for 168 cm man", "calculate_ideal_body_weight"),
    ("Child dose for 32 kg kid adult dose 500 mg", "calculate_pediatric_dose"),
]

def train(output_path: str = "data/tool_selector/selector.pkl"):
    print("Loading embedding model...")
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    queries = [q for q, _ in TRAINING_DATA]
    labels  = [l for _, l in TRAINING_DATA]
    print(f"Encoding {len(queries)} training examples...")
    
    X = encoder.encode(queries, show_progress_bar=True)
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    clf = LogisticRegression(solver='lbfgs', max_iter=1000, C=4.0)
    clf.fit(X, y)

    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"\nCV accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
    
    print("\nTest predictions:")
    for query, expected in TEST_QUERIES:
        vec  = encoder.encode([query], show_progress_bar=False)
        pred = le.inverse_transform(clf.predict(vec))[0]
        status = "OK" if pred == expected else "FAIL"
        print(f"  [{status}] {query[:50]} -> {pred}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({'clf': clf, 'le': le}, f)
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    train()
