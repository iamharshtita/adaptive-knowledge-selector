# Evaluate RL Model with ML-style Metrics (Accuracy, Precision, Recall, F1)
# Compares 3 evaluation approaches

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from models.adaptive_selector import AdaptiveSelector
from scripts.train_rl_agent import TrainingSystem
from utils.llm_judge import LLMJudge


class ModelEvaluator:
    """Evaluate RL model using ML-style classification metrics"""

    def __init__(self):
        print("\n" + "=" * 80)
        print("  MODEL EVALUATION - ML METRICS")
        print("=" * 80)
        print("\n  Loading components...")

        # Load model
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.agent = AdaptiveSelector(input_dim=384, num_sources=4, lr=0.001)

        model_path = "data/rl_selector/adaptive_dqn.pth"
        if not self.agent.load(model_path):
            print(f"ERROR: Could not load model from {model_path}")
            sys.exit(1)

        print("  ✓ Model loaded")

        # Initialize sources
        self.system = TrainingSystem()
        print("  ✓ Sources ready")

        # Initialize LLM judge
        self.judge = LLMJudge()
        print("  ✓ LLM Judge ready")

        # Explicitly set source names
        self.sources = ["KnowledgeGraphSource", "ToolAPISource", "LLMSource", "PDFKnowledgeSource"]

        print("\n" + "=" * 80)

    # ========================================================================
    # APPROACH 1: Manual Labeled Test Set
    # ========================================================================

    def get_manual_test_set(self):
        """Manually labeled test set with 250 queries (62-63 per source)"""
        return [
            # ============================================================
            # KnowledgeGraphSource (50 queries)
            # Drug interactions, side effects, treatments, relationships
            # ============================================================
            {"query": "What drugs interact with warfarin?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Can I take aspirin with ibuprofen?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Does metformin interact with alcohol?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Drug interactions with lisinopril", "correct_source": "KnowledgeGraphSource"},
            {"query": "What are side effects of metformin?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Side effects of atorvastatin", "correct_source": "KnowledgeGraphSource"},
            {"query": "Can I take clopidogrel with omeprazole?", "correct_source": "KnowledgeGraphSource"},
            {"query": "What medications interact with simvastatin?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Does amoxicillin interact with birth control?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Drug interactions for amlodipine", "correct_source": "KnowledgeGraphSource"},
            {"query": "What are the side effects of prednisone?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Side effects of hydrochlorothiazide", "correct_source": "KnowledgeGraphSource"},
            {"query": "Can I take acetaminophen with naproxen?", "correct_source": "KnowledgeGraphSource"},
            {"query": "What drugs interact with levothyroxine?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Does gabapentin interact with tramadol?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Side effects of losartan", "correct_source": "KnowledgeGraphSource"},
            {"query": "What medications treat hypertension?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Drugs used to treat diabetes", "correct_source": "KnowledgeGraphSource"},
            {"query": "What treats atrial fibrillation?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Medications for heart failure", "correct_source": "KnowledgeGraphSource"},
            {"query": "What drugs interact with duloxetine?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Can I take metoprolol with albuterol?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Side effects of sertraline", "correct_source": "KnowledgeGraphSource"},
            {"query": "What interacts with fluoxetine?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Does pantoprazole interact with iron?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Side effects of montelukast", "correct_source": "KnowledgeGraphSource"},
            {"query": "What drugs interact with diazepam?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Can you take lorazepam with oxycodone?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Side effects of alprazolam", "correct_source": "KnowledgeGraphSource"},
            {"query": "What medications treat depression?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Drugs for anxiety disorders", "correct_source": "KnowledgeGraphSource"},
            {"query": "What treats rheumatoid arthritis?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Medications for osteoporosis", "correct_source": "KnowledgeGraphSource"},
            {"query": "What drugs interact with cyclosporine?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Does azithromycin interact with statins?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Side effects of carvedilol", "correct_source": "KnowledgeGraphSource"},
            {"query": "What interacts with digoxin?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Can I take spironolactone with ACE inhibitors?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Side effects of furosemide", "correct_source": "KnowledgeGraphSource"},
            {"query": "What drugs interact with diltiazem?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Does verapamil interact with beta blockers?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Side effects of ramipril", "correct_source": "KnowledgeGraphSource"},
            {"query": "What medications treat COPD?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Drugs for asthma treatment", "correct_source": "KnowledgeGraphSource"},
            {"query": "What treats Parkinson's disease?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Medications for seizures", "correct_source": "KnowledgeGraphSource"},
            {"query": "What drugs interact with phenytoin?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Side effects of valproic acid", "correct_source": "KnowledgeGraphSource"},
            {"query": "Can I take carbamazepine with lamotrigine?", "correct_source": "KnowledgeGraphSource"},
            {"query": "What interacts with levetiracetam?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Side effects of topiramate", "correct_source": "KnowledgeGraphSource"},
            {"query": "What drugs interact with clonazepam?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Can I take buspirone with SSRIs?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Side effects of trazodone", "correct_source": "KnowledgeGraphSource"},
            {"query": "What treats bipolar disorder?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Medications for schizophrenia", "correct_source": "KnowledgeGraphSource"},
            {"query": "What drugs interact with lithium?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Side effects of quetiapine", "correct_source": "KnowledgeGraphSource"},
            {"query": "What interacts with olanzapine?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Can I take aripiprazole with other antipsychotics?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Side effects of risperidone", "correct_source": "KnowledgeGraphSource"},
            {"query": "What drugs interact with clozapine?", "correct_source": "KnowledgeGraphSource"},
            {"query": "Medications for migraine prevention", "correct_source": "KnowledgeGraphSource"},

            # ============================================================
            # ToolAPISource (63 queries)
            # Calculations, FDA labels, drug information lookups
            # ============================================================
            {"query": "Calculate BMI for 70 kg and 1.75 m", "correct_source": "ToolAPISource"},
            {"query": "What is creatinine clearance for 60 year old, 70 kg, Cr 1.0?", "correct_source": "ToolAPISource"},
            {"query": "Ideal body weight for 175 cm male", "correct_source": "ToolAPISource"},
            {"query": "BMI for 85 kg and 1.80 m", "correct_source": "ToolAPISource"},
            {"query": "What are FDA approved indications for metformin?", "correct_source": "ToolAPISource"},
            {"query": "FDA label for lisinopril", "correct_source": "ToolAPISource"},
            {"query": "Calculate BMI for 55 kg and 1.60 m", "correct_source": "ToolAPISource"},
            {"query": "Creatinine clearance for 75 year old female, 60 kg, Cr 1.5", "correct_source": "ToolAPISource"},
            {"query": "What is ideal body weight for 160 cm female?", "correct_source": "ToolAPISource"},
            {"query": "BMI calculation for 90 kg, 1.85 m", "correct_source": "ToolAPISource"},
            {"query": "FDA indications for atorvastatin", "correct_source": "ToolAPISource"},
            {"query": "Prescribing information for amlodipine", "correct_source": "ToolAPISource"},
            {"query": "Calculate BMI 100 kg 1.70 m", "correct_source": "ToolAPISource"},
            {"query": "Creatinine clearance 50 year old male 80 kg Cr 0.9", "correct_source": "ToolAPISource"},
            {"query": "Ideal weight for 180 cm male", "correct_source": "ToolAPISource"},
            {"query": "What is BMI for 65 kg and 1.68 m?", "correct_source": "ToolAPISource"},
            {"query": "FDA label information for omeprazole", "correct_source": "ToolAPISource"},
            {"query": "Approved uses for gabapentin", "correct_source": "ToolAPISource"},
            {"query": "Calculate body mass index 75 kg 1.72 m", "correct_source": "ToolAPISource"},
            {"query": "CrCl for 65 year old woman 55 kg Cr 1.2", "correct_source": "ToolAPISource"},
            {"query": "IBW for 165 cm female", "correct_source": "ToolAPISource"},
            {"query": "BMI 80 kg 1.78 m", "correct_source": "ToolAPISource"},
            {"query": "FDA warnings for metformin", "correct_source": "ToolAPISource"},
            {"query": "Prescribing info for levothyroxine", "correct_source": "ToolAPISource"},
            {"query": "Calculate BMI 95 kg 1.82 m", "correct_source": "ToolAPISource"},
            {"query": "Creatinine clearance 70 year old 65 kg Cr 1.3", "correct_source": "ToolAPISource"},
            {"query": "Ideal body weight 170 cm male", "correct_source": "ToolAPISource"},
            {"query": "What is my BMI if I'm 68 kg and 1.65 m?", "correct_source": "ToolAPISource"},
            {"query": "FDA approved indications for sertraline", "correct_source": "ToolAPISource"},
            {"query": "Drug label for losartan", "correct_source": "ToolAPISource"},
            {"query": "BMI calculation 72 kg 1.76 m", "correct_source": "ToolAPISource"},
            {"query": "CrCl 55 year old male 85 kg Cr 1.1", "correct_source": "ToolAPISource"},
            {"query": "Ideal weight 158 cm female", "correct_source": "ToolAPISource"},
            {"query": "Calculate BMI 62 kg 1.69 m", "correct_source": "ToolAPISource"},
            {"query": "FDA information for hydrochlorothiazide", "correct_source": "ToolAPISource"},
            {"query": "Prescribing details for duloxetine", "correct_source": "ToolAPISource"},
            {"query": "BMI for 88 kg and 1.81 m", "correct_source": "ToolAPISource"},
            {"query": "Creatinine clearance 80 year old 58 kg Cr 1.4", "correct_source": "ToolAPISource"},
            {"query": "IBW for 172 cm male", "correct_source": "ToolAPISource"},
            {"query": "What's BMI for 77 kg 1.74 m?", "correct_source": "ToolAPISource"},
            {"query": "FDA label for pantoprazole", "correct_source": "ToolAPISource"},
            {"query": "Approved indications for montelukast", "correct_source": "ToolAPISource"},
            {"query": "Calculate BMI 58 kg 1.63 m", "correct_source": "ToolAPISource"},
            {"query": "CrCl 68 year old female 62 kg Cr 1.0", "correct_source": "ToolAPISource"},
            {"query": "Ideal body weight 168 cm female", "correct_source": "ToolAPISource"},
            {"query": "BMI 83 kg 1.79 m", "correct_source": "ToolAPISource"},
            {"query": "FDA warnings for albuterol", "correct_source": "ToolAPISource"},
            {"query": "Prescribing information for carvedilol", "correct_source": "ToolAPISource"},
            {"query": "Calculate BMI 71 kg 1.71 m", "correct_source": "ToolAPISource"},
            {"query": "Creatinine clearance 72 year old 67 kg Cr 1.2", "correct_source": "ToolAPISource"},
            {"query": "BMI for 66 kg and 1.67 m", "correct_source": "ToolAPISource"},
            {"query": "FDA label for spironolactone", "correct_source": "ToolAPISource"},
            {"query": "Approved indications for furosemide", "correct_source": "ToolAPISource"},
            {"query": "Calculate BMI 79 kg 1.77 m", "correct_source": "ToolAPISource"},
            {"query": "CrCl 63 year old male 73 kg Cr 1.1", "correct_source": "ToolAPISource"},
            {"query": "Ideal body weight 166 cm female", "correct_source": "ToolAPISource"},
            {"query": "BMI 92 kg 1.83 m", "correct_source": "ToolAPISource"},
            {"query": "FDA warnings for ramipril", "correct_source": "ToolAPISource"},
            {"query": "Prescribing information for diltiazem", "correct_source": "ToolAPISource"},
            {"query": "Calculate BMI 74 kg 1.73 m", "correct_source": "ToolAPISource"},
            {"query": "Creatinine clearance 58 year old 69 kg Cr 0.8", "correct_source": "ToolAPISource"},
            {"query": "IBW for 174 cm male", "correct_source": "ToolAPISource"},
            {"query": "What is BMI for 81 kg 1.80 m?", "correct_source": "ToolAPISource"},

            # ============================================================
            # LLMSource (63 queries)
            # Concepts, mechanisms, explanations, general knowledge
            # ============================================================
            {"query": "How does insulin work?", "correct_source": "LLMSource"},
            {"query": "Explain the mechanism of ACE inhibitors", "correct_source": "LLMSource"},
            {"query": "What is the difference between Type 1 and Type 2 diabetes?", "correct_source": "LLMSource"},
            {"query": "How do beta blockers lower blood pressure?", "correct_source": "LLMSource"},
            {"query": "What is hypertension?", "correct_source": "LLMSource"},
            {"query": "Explain how statins work", "correct_source": "LLMSource"},
            {"query": "What causes heart failure?", "correct_source": "LLMSource"},
            {"query": "How do diuretics work?", "correct_source": "LLMSource"},
            {"query": "What is atrial fibrillation?", "correct_source": "LLMSource"},
            {"query": "Explain the mechanism of action of metformin", "correct_source": "LLMSource"},
            {"query": "What is the difference between angina and heart attack?", "correct_source": "LLMSource"},
            {"query": "How do calcium channel blockers work?", "correct_source": "LLMSource"},
            {"query": "What is chronic kidney disease?", "correct_source": "LLMSource"},
            {"query": "Explain how antibiotics work", "correct_source": "LLMSource"},
            {"query": "What is the difference between bacteria and viruses?", "correct_source": "LLMSource"},
            {"query": "How does the immune system fight infections?", "correct_source": "LLMSource"},
            {"query": "What is inflammation?", "correct_source": "LLMSource"},
            {"query": "Explain how NSAIDs reduce pain", "correct_source": "LLMSource"},
            {"query": "What causes asthma?", "correct_source": "LLMSource"},
            {"query": "How do bronchodilators work?", "correct_source": "LLMSource"},
            {"query": "What is COPD?", "correct_source": "LLMSource"},
            {"query": "Explain the pathophysiology of pneumonia", "correct_source": "LLMSource"},
            {"query": "What is sepsis?", "correct_source": "LLMSource"},
            {"query": "How does blood clotting work?", "correct_source": "LLMSource"},
            {"query": "What is the difference between warfarin and heparin?", "correct_source": "LLMSource"},
            {"query": "Explain how anticoagulants work", "correct_source": "LLMSource"},
            {"query": "What causes stroke?", "correct_source": "LLMSource"},
            {"query": "How do antiplatelet drugs work?", "correct_source": "LLMSource"},
            {"query": "What is atherosclerosis?", "correct_source": "LLMSource"},
            {"query": "Explain cholesterol metabolism", "correct_source": "LLMSource"},
            {"query": "What is the difference between HDL and LDL?", "correct_source": "LLMSource"},
            {"query": "How does the kidney filter blood?", "correct_source": "LLMSource"},
            {"query": "What is diabetic nephropathy?", "correct_source": "LLMSource"},
            {"query": "Explain how loop diuretics work", "correct_source": "LLMSource"},
            {"query": "What causes edema?", "correct_source": "LLMSource"},
            {"query": "How do thiazide diuretics work?", "correct_source": "LLMSource"},
            {"query": "What is congestive heart failure?", "correct_source": "LLMSource"},
            {"query": "Explain the mechanism of digoxin", "correct_source": "LLMSource"},
            {"query": "What is the Frank-Starling mechanism?", "correct_source": "LLMSource"},
            {"query": "How does the heart pump blood?", "correct_source": "LLMSource"},
            {"query": "What is systolic vs diastolic blood pressure?", "correct_source": "LLMSource"},
            {"query": "Explain cardiac output", "correct_source": "LLMSource"},
            {"query": "What causes arrhythmias?", "correct_source": "LLMSource"},
            {"query": "How do antiarrhythmic drugs work?", "correct_source": "LLMSource"},
            {"query": "What is ventricular fibrillation?", "correct_source": "LLMSource"},
            {"query": "Explain how defibrillation works", "correct_source": "LLMSource"},
            {"query": "What is myocardial infarction?", "correct_source": "LLMSource"},
            {"query": "How do thrombolytics work?", "correct_source": "LLMSource"},
            {"query": "What is the difference between ischemia and infarction?", "correct_source": "LLMSource"},
            {"query": "Explain oxygen delivery to tissues", "correct_source": "LLMSource"},
            {"query": "What is hemoglobin?", "correct_source": "LLMSource"},
            {"query": "How does anemia affect oxygen transport?", "correct_source": "LLMSource"},
            {"query": "What is the difference between anemia types?", "correct_source": "LLMSource"},
            {"query": "Explain how erythropoietin works", "correct_source": "LLMSource"},
            {"query": "What causes iron deficiency anemia?", "correct_source": "LLMSource"},
            {"query": "How do iron supplements work?", "correct_source": "LLMSource"},
            {"query": "What is pernicious anemia?", "correct_source": "LLMSource"},
            {"query": "Explain vitamin B12 metabolism", "correct_source": "LLMSource"},
            {"query": "What is folate deficiency?", "correct_source": "LLMSource"},
            {"query": "How does the body absorb iron?", "correct_source": "LLMSource"},
            {"query": "What is sickle cell disease?", "correct_source": "LLMSource"},
            {"query": "Explain thalassemia pathophysiology", "correct_source": "LLMSource"},

            # ============================================================
            # PDFKnowledgeSource (61 queries)
            # Document-specific queries about papers, research, KRR
            # ============================================================
            {"query": "What do the KRR papers say about ontologies?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Knowledge representation in clinical decision support", "correct_source": "PDFKnowledgeSource"},
            {"query": "Medical reasoning approaches in the documents", "correct_source": "PDFKnowledgeSource"},
            {"query": "Semantic web technologies for healthcare", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do the papers discuss about SNOMED CT?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Description logics in medical informatics", "correct_source": "PDFKnowledgeSource"},
            {"query": "What is mentioned about RDF in the documents?", "correct_source": "PDFKnowledgeSource"},
            {"query": "OWL ontology applications in medicine", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do the KRR papers say about reasoning?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Clinical terminologies discussed in papers", "correct_source": "PDFKnowledgeSource"},
            {"query": "What is said about ICD-10 in the documents?", "correct_source": "PDFKnowledgeSource"},
            {"query": "UMLS discussion in the papers", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do the documents say about medical coding?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Health information exchange in the papers", "correct_source": "PDFKnowledgeSource"},
            {"query": "What is mentioned about FHIR?", "correct_source": "PDFKnowledgeSource"},
            {"query": "HL7 standards discussed in documents", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do papers say about interoperability?", "correct_source": "PDFKnowledgeSource"},
            {"query": "EHR systems mentioned in the documents", "correct_source": "PDFKnowledgeSource"},
            {"query": "What is said about clinical workflows?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Decision support systems in the papers", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do documents discuss about CDSS?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Guideline representation in the papers", "correct_source": "PDFKnowledgeSource"},
            {"query": "What is mentioned about Arden Syntax?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Rule-based systems in medical AI", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do papers say about expert systems?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Machine learning applications in the documents", "correct_source": "PDFKnowledgeSource"},
            {"query": "What is discussed about natural language processing?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Clinical NLP mentioned in papers", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do documents say about text mining?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Information extraction from clinical notes", "correct_source": "PDFKnowledgeSource"},
            {"query": "What is said about named entity recognition?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Temporal reasoning in the papers", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do documents discuss about time series?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Event detection in clinical data", "correct_source": "PDFKnowledgeSource"},
            {"query": "What is mentioned about data quality?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Privacy and security in the papers", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do documents say about HIPAA?", "correct_source": "PDFKnowledgeSource"},
            {"query": "De-identification techniques discussed", "correct_source": "PDFKnowledgeSource"},
            {"query": "What is said about data governance?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Clinical data warehousing in papers", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do documents discuss about i2b2?", "correct_source": "PDFKnowledgeSource"},
            {"query": "OMOP common data model mentioned", "correct_source": "PDFKnowledgeSource"},
            {"query": "What is said about PCORnet?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Phenotyping algorithms in the papers", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do documents say about computable phenotypes?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Clinical research informatics discussed", "correct_source": "PDFKnowledgeSource"},
            {"query": "What is mentioned about registry systems?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Cohort identification in the papers", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do documents discuss about trial recruitment?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Precision medicine approaches mentioned", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do papers say about genomic medicine?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Pharmacogenomics discussed in documents", "correct_source": "PDFKnowledgeSource"},
            {"query": "What is mentioned about biomarkers?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Clinical trials informatics in the papers", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do documents discuss about translational research?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Learning health systems mentioned", "correct_source": "PDFKnowledgeSource"},
            {"query": "What is said about population health?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Public health informatics in papers", "correct_source": "PDFKnowledgeSource"},
            {"query": "What do documents say about syndromic surveillance?", "correct_source": "PDFKnowledgeSource"},
            {"query": "Epidemiological modeling discussed", "correct_source": "PDFKnowledgeSource"},
        ]

    # ========================================================================
    # APPROACH 2: LLM Judge as Ground Truth
    # ========================================================================

    def evaluate_with_llm_judge(self, queries):
        """Query all sources, use LLM judge to determine best source"""
        print("\n  Querying all sources for each query...")

        ground_truth = []

        for i, query in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] {query[:50]}...")

            best_source = None
            best_quality = -1

            # Query all 4 sources
            for source_name in self.sources:
                try:
                    results = self.system.query_source(source_name, query)

                    if isinstance(results, dict):
                        answer = results.get('answer', str(results))
                    else:
                        answer = str(results)

                    # Skip empty answers
                    if not answer or len(answer.strip()) < 10:
                        continue

                    # Evaluate quality
                    evaluation = self.judge.evaluate_quality(query, answer, source_name)
                    quality = evaluation.get('quality', 0)

                    if quality > best_quality:
                        best_quality = quality
                        best_source = source_name

                except Exception as e:
                    print(f"    ⚠ Error with {source_name}: {str(e)[:50]}")
                    continue

            # Only add if we found a valid source
            if best_source:
                ground_truth.append({
                    "query": query,
                    "correct_source": best_source,
                    "quality": best_quality
                })
            else:
                print(f"    ⚠ No valid source found, skipping query")

        return ground_truth

    # ========================================================================
    # APPROACH 3: Use Existing Training Data as Test Set
    # ========================================================================

    def load_training_data_test_set(self, samples_per_source=10):
        """Use stratified sampling from training dataset - balanced across all sources"""
        training_file = "data/training_dataset_600.json"

        if not os.path.exists(training_file):
            print(f"  ⚠ Training data not found: {training_file}")
            return []

        with open(training_file, 'r') as f:
            data = json.load(f)

        # Group by source
        source_groups = defaultdict(list)
        for item in data:
            source = item.get('best_source')
            if source:
                source_groups[source].append(item)

        print(f"\n  Training data distribution:")
        for source in self.sources:
            count = len(source_groups.get(source, []))
            print(f"    {source.replace('Source', ''):<25s}: {count} samples")

        # Stratified sampling - take equal samples from each source
        test_set = []
        for source in self.sources:
            source_samples = source_groups.get(source, [])

            # Take last N samples from this source (or all if fewer)
            num_to_take = min(samples_per_source, len(source_samples))
            selected = source_samples[-num_to_take:]

            for item in selected:
                test_set.append({
                    "query": item['query'],
                    "correct_source": item['best_source']
                })

        print(f"\n  Stratified test set created: {len(test_set)} samples")
        print(f"  ({samples_per_source} per source × 4 sources)")

        return test_set

    # ========================================================================
    # Evaluation Logic
    # ========================================================================

    def evaluate_test_set(self, test_set, approach_name):
        """Evaluate model on test set and compute metrics"""
        print(f"\n" + "=" * 80)
        print(f"  EVALUATING: {approach_name}")
        print("=" * 80)
        print(f"\n  Test set size: {len(test_set)}")

        predictions = []
        ground_truth = []

        # Get model predictions
        for item in test_set:
            query = item['query']
            correct_source = item.get('correct_source')

            # Skip if no ground truth
            if not correct_source:
                continue

            # Encode and predict
            emb = self.encoder.encode([query])[0]
            action_idx = self.agent.select_action(emb, epsilon=0.0)
            predicted_source = self.sources[action_idx]

            predictions.append(predicted_source)
            ground_truth.append(correct_source)

        # Compute metrics
        metrics = self.compute_metrics(predictions, ground_truth)

        # Print results
        self.print_metrics(metrics, approach_name)

        return metrics

    def compute_metrics(self, predictions, ground_truth):
        """Compute accuracy, precision, recall, F1"""

        # Overall accuracy
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        accuracy = correct / len(predictions) if predictions else 0

        # Per-source metrics
        source_metrics = {}

        for source in self.sources:
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p == source and g == source)
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p == source and g != source)
            fn = sum(1 for p, g in zip(predictions, ground_truth) if p != source and g == source)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            source_metrics[source] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': sum(1 for g in ground_truth if g == source)
            }

        # Confusion matrix
        confusion = defaultdict(lambda: defaultdict(int))
        for p, g in zip(predictions, ground_truth):
            confusion[g][p] += 1

        return {
            'accuracy': accuracy,
            'source_metrics': source_metrics,
            'confusion_matrix': confusion,
            'predictions': predictions,
            'ground_truth': ground_truth
        }

    def print_metrics(self, metrics, approach_name):
        """Pretty print metrics"""

        print(f"\n  OVERALL ACCURACY: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")

        print(f"\n  PER-SOURCE METRICS:")
        print(f"  {'Source':<25s} {'Precision':<12s} {'Recall':<12s} {'F1':<12s} {'Support'}")
        print(f"  {'-'*75}")

        for source in self.sources:
            m = metrics['source_metrics'][source]
            src_short = source.replace('Source', '')
            print(f"  {src_short:<25s} {m['precision']:>8.3f}     {m['recall']:>8.3f}     {m['f1']:>8.3f}     {m['support']:>4d}")

        # Macro averages
        avg_precision = np.mean([m['precision'] for m in metrics['source_metrics'].values()])
        avg_recall = np.mean([m['recall'] for m in metrics['source_metrics'].values()])
        avg_f1 = np.mean([m['f1'] for m in metrics['source_metrics'].values()])

        print(f"  {'-'*75}")
        print(f"  {'Macro Average':<25s} {avg_precision:>8.3f}     {avg_recall:>8.3f}     {avg_f1:>8.3f}")

        # Confusion matrix
        print(f"\n  CONFUSION MATRIX:")
        print(f"  (Rows: Ground Truth, Columns: Predicted)")

        # Header
        src_abbr = {s: s.replace('Source', '')[:8] for s in self.sources}
        print(f"\n  {'Ground Truth':<15s}", end="")
        for source in self.sources:
            print(f"  {src_abbr[source]:>8s}", end="")
        print()
        print(f"  {'-'*70}")

        for true_source in self.sources:
            print(f"  {src_abbr[true_source]:<15s}", end="")
            for pred_source in self.sources:
                count = metrics['confusion_matrix'][true_source][pred_source]
                print(f"  {count:>8d}", end="")
            print()

    # ========================================================================
    # Main Evaluation Runner
    # ========================================================================

    def run_all_evaluations(self):
        """Run manual evaluation only"""

        results = {}

        # ONLY Manual Test Set
        print("\n" + "=" * 80)
        print("  EVALUATION: MANUAL LABELED TEST SET (250 queries)")
        print("=" * 80)

        manual_test_set = self.get_manual_test_set()
        results['manual'] = self.evaluate_test_set(manual_test_set, "Manual Labeled Test Set")

        return results

    def compare_approaches(self, results):
        """Compare the 3 evaluation approaches"""

        print("\n" + "=" * 80)
        print("  COMPARISON OF EVALUATION APPROACHES")
        print("=" * 80)

        print(f"\n  {'Approach':<30s} {'Accuracy':<12s} {'Macro F1':<12s} {'Test Size'}")
        print(f"  {'-'*70}")

        for approach_name, metrics in results.items():
            if metrics is None or not metrics:
                continue

            # Skip if no predictions
            if len(metrics.get('predictions', [])) == 0:
                continue

            accuracy = metrics['accuracy']
            avg_f1 = np.mean([m['f1'] for m in metrics['source_metrics'].values()])
            test_size = len(metrics['predictions'])

            display_name = {
                'manual': 'Manual Labeled',
                'llm_judge': 'LLM Judge',
                'training': 'Training Data'
            }.get(approach_name, approach_name)

            print(f"  {display_name:<30s} {accuracy:>8.3f}     {avg_f1:>8.3f}     {test_size:>4d}")

        print(f"\n  ANALYSIS:")
        print(f"  • Manual Labeled: High quality, small size, labor intensive")
        print(f"  • LLM Judge: Automated, but expensive and may have noise")
        print(f"  • Training Data: Large size, but may be biased (model trained on it)")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.run_all_evaluations()

    # Save results
    output_file = "data/rl_selector/evaluation_metrics.json"

    # Convert numpy/defaultdict to JSON-serializable format
    json_results = {}
    for approach, metrics in results.items():
        if metrics:
            json_results[approach] = {
                'accuracy': float(metrics['accuracy']),
                'source_metrics': {k: {
                    'precision': float(v['precision']),
                    'recall': float(v['recall']),
                    'f1': float(v['f1']),
                    'support': int(v['support'])
                } for k, v in metrics['source_metrics'].items()},
                'confusion_matrix': {k: dict(v) for k, v in metrics['confusion_matrix'].items()}
            }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}\n")