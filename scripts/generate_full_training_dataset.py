"""
Generate comprehensive training dataset with 600 medically accurate queries
Ensures balanced distribution across sources and query types
"""

import json
import os

# Target distribution for 600 queries
TARGET_TOTAL = 600
TARGET_BY_SOURCE = {
    "KnowledgeGraphSource": 180,   # 30% - structured data
    "ToolAPISource": 160,            # 27% - calculations and labels
    "LLMSource": 200,                # 33% - explanations
    "PDFKnowledgeSource": 60         # 10% - documents
}

# Load existing dataset
existing_file = "data/training_dataset.json"
if os.path.exists(existing_file):
    with open(existing_file, 'r') as f:
        existing_data = json.load(f)
    print(f"Loaded {len(existing_data)} existing queries")
else:
    existing_data = []
    print("Starting fresh dataset")

# Additional queries to reach 600 total
# These are generated programmatically to ensure coverage

additional_queries = []

# ============================================================================
# KNOWLEDGE GRAPH SOURCE QUERIES (Need 120 more to reach 180)
# ============================================================================

# More interaction queries (variations and additional drugs) - need 40 total
kg_interaction_queries = [
    ("amlodipine and simvastatin interaction", "Common combination query"),
    ("losartan drug interactions", "ARB interactions"),
    ("enalapril interactions with other drugs", "ACE inhibitor interactions"),
    ("what interacts with hydrochlorothiazide", "Diuretic interactions"),
    ("atenolol drug-drug interactions", "Beta blocker interactions"),
    ("interactions of diltiazem", "Calcium channel blocker"),
    ("verapamil interactions", "CCB with many interactions"),
    ("carvedilol drug interactions", "Alpha-beta blocker"),
    ("ramipril interaction profile", "ACE inhibitor"),
    ("bisoprolol interactions with other medications", "Selective beta blocker"),
    ("spironolactone drug interactions", "Potassium-sparing diuretic"),
    ("amlodipine interactions", "Common CCB"),
    ("hydrochlorothiazide and lisinopril interaction", "Common combination"),
    ("atorvastatin drug interactions", "Most prescribed statin"),
    ("rosuvastatin interactions", "Potent statin"),
    ("pravastatin drug-drug interactions", "Renal-cleared statin"),
    ("clopidogrel interactions with PPIs", "Documented interaction"),
    ("prasugrel drug interactions", "P2Y12 inhibitor"),
    ("ticagrelor interactions", "Another P2Y12 inhibitor"),
    ("dabigatran drug interactions", "Direct thrombin inhibitor"),
    ("apixaban interactions", "Factor Xa inhibitor"),
    ("rivaroxaban drug-drug interactions", "Common DOAC"),
    ("edoxaban interactions", "DOAC"),
    ("sertraline drug interactions", "SSRI"),
    ("fluoxetine interactions with other drugs", "Long half-life SSRI"),
    ("escitalopram drug interactions", "Common SSRI"),
    ("venlafaxine interactions", "SNRI"),
    ("duloxetine drug-drug interactions", "SNRI for pain"),
    ("bupropion interactions", "NDRI antidepressant"),
    ("mirtazapine drug interactions", "Atypical antidepressant"),
    ("aripiprazole interactions", "Atypical antipsychotic"),
    ("quetiapine drug interactions", "Common antipsychotic"),
    ("risperidone interactions", "Antipsychotic"),
    ("olanzapine drug-drug interactions", "Metabolic effects"),
    ("alprazolam interactions", "Benzodiazepine"),
    ("lorazepam drug interactions", "Short-acting benzo"),
    ("diazepam interactions", "Long-acting benzo"),
    ("zolpidem drug interactions", "Sleep medication"),
    ("eszopiclone interactions", "Non-benzo sleep aid"),
    ("trazodone drug interactions", "Sleep/depression"),
    ("levothyroxine interactions", "Thyroid medication interactions"),
    ("metoprolol drug interactions", "Selective beta-1 blocker"),
    ("propranolol interactions with other drugs", "Non-selective beta blocker"),
    ("candesartan drug interactions", "ARB interactions"),
    ("valsartan interactions", "Common ARB"),
    ("telmisartan drug-drug interactions", "Long-acting ARB"),
]

for query, reasoning in kg_interaction_queries:
    additional_queries.append({
        "query": query,
        "best_source": "KnowledgeGraphSource",
        "alternative_sources": ["ToolAPISource"],
        "reasoning": f"{reasoning} - structured relationships in knowledge graph",
        "query_type": "interaction",
        "expected_result_type": "list_of_drugs"
    })

# Treatment queries for KG (need 80 total for 40+40=80 treatment queries)
kg_treatment_queries = [
    ("medications for chronic kidney disease", "CKD treatment options"),
    ("drugs for diabetic neuropathy", "Neuropathy treatments"),
    ("treatments for acute coronary syndrome", "ACS medications"),
    ("medications for peripheral artery disease", "PAD treatment"),
    ("drugs for venous thromboembolism", "VTE treatment options"),
    ("treatments for pulmonary embolism", "PE medications"),
    ("medications for stroke prevention", "Stroke prophylaxis"),
    ("drugs for coronary artery disease", "CAD treatment"),
    ("treatments for angina pectoris", "Angina medications"),
    ("medications for chronic pain", "Pain management drugs"),
    ("drugs for fibromyalgia", "Fibromyalgia treatment"),
    ("treatments for neuropathic pain", "Neuropathic pain drugs"),
    ("medications for inflammatory bowel disease", "IBD treatment"),
    ("drugs for Crohn's disease", "Crohn's medications"),
    ("treatments for ulcerative colitis", "UC drugs"),
    ("medications for multiple sclerosis", "MS treatment options"),
    ("drugs for schizophrenia", "Antipsychotic options"),
    ("treatments for bipolar disorder", "Mood stabilizers"),
    ("medications for ADHD", "ADHD treatment drugs"),
    ("drugs for generalized anxiety disorder", "GAD medications"),
    ("treatments for panic disorder", "Panic disorder drugs"),
    ("medications for obsessive compulsive disorder", "OCD treatment"),
    ("drugs for post-traumatic stress disorder", "PTSD medications"),
    ("treatments for insomnia", "Sleep disorder drugs"),
    ("medications for hypothyroidism", "Thyroid replacement"),
    ("drugs for hyperthyroidism", "Hyperthyroid treatment"),
    ("treatments for Graves disease", "Graves disease drugs"),
    ("medications for Cushing syndrome", "Cushing's treatment"),
    ("drugs for Addison disease", "Addison's medications"),
    ("treatments for pheochromocytoma", "Pheochromocytoma drugs"),
    ("medications for benign prostatic hyperplasia", "BPH treatment"),
    ("drugs for erectile dysfunction", "ED medications"),
    ("treatments for overactive bladder", "OAB drugs"),
    ("medications for urinary incontinence", "Incontinence treatment"),
    ("drugs for glaucoma", "Glaucoma medications"),
    ("treatments for macular degeneration", "AMD treatment"),
    ("medications for seasonal allergies", "Allergy drugs"),
    ("drugs for allergic rhinitis", "Rhinitis treatment"),
    ("treatments for chronic urticaria", "Hives medications"),
    ("medications for psoriasis", "Psoriasis treatment"),
    ("drugs for rheumatoid arthritis", "RA medications"),
    ("treatments for osteoarthritis", "OA drugs"),
    ("medications for gout", "Gout treatment"),
    ("drugs for osteoporosis", "Bone health medications"),
    ("treatments for Parkinson's disease", "PD drugs"),
    ("medications for Alzheimer's disease", "Dementia treatment"),
    ("drugs for epilepsy", "Anticonvulsant options"),
    ("treatments for migraine prophylaxis", "Migraine prevention"),
    ("medications for cluster headaches", "Cluster headache drugs"),
    ("drugs for tension headaches", "Tension headache treatment"),
    ("treatments for trigeminal neuralgia", "Nerve pain medication"),
    ("medications for restless leg syndrome", "RLS treatment"),
    ("drugs for narcolepsy", "Narcolepsy medications"),
    ("treatments for myasthenia gravis", "MG drugs"),
    ("medications for polymyalgia rheumatica", "PMR treatment"),
    ("drugs for temporal arteritis", "Giant cell arteritis"),
    ("treatments for systemic lupus erythematosus", "SLE medications"),
    ("medications for scleroderma", "Scleroderma treatment"),
    ("drugs for dermatomyositis", "Dermatomyositis treatment"),
    ("treatments for Sjogren's syndrome", "Sjogren's drugs"),
    ("medications for ankylosing spondylitis", "AS treatment"),
    ("drugs for reactive arthritis", "Reactive arthritis meds"),
    ("treatments for septic arthritis", "Septic arthritis antibiotics"),
    ("medications for celiac disease", "Celiac treatment"),
    ("drugs for gastroesophageal reflux disease", "GERD medications"),
    ("treatments for peptic ulcer disease", "PUD drugs"),
    ("medications for irritable bowel syndrome", "IBS treatment"),
    ("drugs for diverticulitis", "Diverticulitis antibiotics"),
    ("treatments for hepatitis B", "HBV antivirals"),
    ("medications for hepatitis C", "HCV direct antivirals"),
    ("drugs for cirrhosis complications", "Cirrhosis management"),
    ("treatments for ascites", "Ascites medications"),
    ("medications for hepatic encephalopathy", "HE treatment"),
    ("drugs for variceal bleeding prophylaxis", "Variceal bleed prevention"),
]

for query, reasoning in kg_treatment_queries:
    additional_queries.append({
        "query": query,
        "best_source": "KnowledgeGraphSource",
        "alternative_sources": ["LLMSource"],
        "reasoning": f"{reasoning} - disease-drug relationships in knowledge graph",
        "query_type": "treatment",
        "expected_result_type": "list_of_drugs"
    })

# ============================================================================
# TOOL API SOURCE QUERIES (Need 119 more to reach 160)
# ============================================================================

# More dosage calculations
tool_dosage_queries = [
    ("BMI calculation for 55 kg and 1.60 m", "BMI formula"),
    ("ideal body weight for 160 cm female", "IBW for females"),
    ("creatinine clearance for 70 year old male 75 kg Cr 1.5", "CrCl calculation"),
    ("body surface area for 165 cm 60 kg", "BSA calculation"),
    ("adjusted body weight for 120 kg 165 cm", "AdjBW for obesity"),
    ("pediatric dose for 12 kg child adult dose 400 mg", "Clark's rule"),
    ("BMI for 68 kg 1.70 m", "Standard BMI"),
    ("creatinine clearance 55 year old female 65 kg Cr 1.0", "Female CrCl"),
    ("ideal body weight 180 cm male", "IBW calculation"),
    ("body mass index 95 kg 1.75 m", "BMI"),
    ("Young's rule dose for 3 year old adult 600 mg", "Pediatric dosing"),
    ("BSA calculation for 170 cm 70 kg", "Mosteller formula"),
    ("adjusted body weight 140 kg 170 cm", "Obesity adjustment"),
    ("pediatric dosing 18 kg child adult 500 mg", "Weight-based pediatric"),
    ("CrCl for 60 yo male 80 kg creatinine 1.3", "Cockcroft-Gault"),
    ("BMI for 105 kg 1.82 m", "Overweight BMI"),
    ("IBW for 175 cm male patient", "Male IBW"),
    ("body surface area 155 cm 50 kg", "Small patient BSA"),
    ("creatinine clearance 45 yo female 55 kg Cr 0.8", "Normal renal function"),
    ("pediatric dose 25 kg child adult dose 750 mg", "Clark's rule pediatric"),
]

for query, reasoning in tool_dosage_queries:
    additional_queries.append({
        "query": query,
        "best_source": "ToolAPISource",
        "alternative_sources": [],
        "reasoning": f"{reasoning} - deterministic calculation",
        "query_type": "dosage",
        "expected_result_type": "calculation"
    })

# Drug label queries
tool_label_queries = [
    ("FDA label for metoprolol", "Beta blocker label"),
    ("prescribing information for amlodipine", "CCB prescribing info"),
    ("losartan drug label", "ARB label"),
    ("FDA indications for sertraline", "SSRI indications"),
    ("atorvastatin prescribing information", "Statin label"),
    ("metformin drug label details", "Diabetes drug label"),
    ("lisinopril FDA label", "ACE inhibitor label"),
    ("omeprazole prescribing information", "PPI label"),
    ("levothyroxine drug label", "Thyroid hormone label"),
    ("albuterol FDA indications", "Beta agonist label"),
    ("gabapentin prescribing information", "Anticonvulsant label"),
    ("hydrochlorothiazide drug label", "Diuretic label"),
    ("pantoprazole FDA label", "PPI label"),
    ("clopidogrel prescribing information", "Antiplatelet label"),
    ("simvastatin drug label", "Statin label"),
    ("escitalopram FDA indications", "SSRI label"),
    ("valsartan prescribing information", "ARB label"),
    ("atenolol drug label", "Beta blocker label"),
    ("duloxetine FDA label", "SNRI label"),
    ("rosuvastatin prescribing information", "Statin label"),
    ("fluoxetine drug label", "SSRI label"),
    ("losartan FDA indications", "ARB indications"),
    ("pravastatin prescribing information", "Statin label"),
    ("venlafaxine drug label", "SNRI label"),
    ("carvedilol FDA label", "Alpha-beta blocker label"),
    ("bupropion prescribing information", "NDRI label"),
    ("ramipril drug label", "ACE inhibitor label"),
    ("aripiprazole FDA indications", "Antipsychotic label"),
    ("quetiapine prescribing information", "Antipsychotic label"),
    ("risperidone drug label", "Antipsychotic label"),
    ("olanzapine FDA label", "Antipsychotic label"),
    ("lamotrigine prescribing information", "Anticonvulsant label"),
    ("topiramate drug label", "Anticonvulsant label"),
    ("levetiracetam FDA indications", "Anticonvulsant label"),
    ("phenytoin prescribing information", "Anticonvulsant label"),
    ("valproic acid drug label", "Mood stabilizer label"),
    ("lithium carbonate FDA label", "Mood stabilizer label"),
    ("buspirone prescribing information", "Anxiolytic label"),
    ("hydroxyzine drug label", "Anxiolytic label"),
    ("trazodone FDA indications", "Antidepressant/sleep label"),
    ("methotrexate FDA label", "DMARD label"),
    ("adalimumab prescribing information", "Biologic label"),
    ("etanercept drug label", "TNF inhibitor label"),
    ("infliximab FDA indications", "Biologic label"),
    ("rituximab prescribing information", "Monoclonal antibody label"),
    ("insulin glargine drug label", "Long-acting insulin label"),
    ("insulin lispro FDA indications", "Rapid-acting insulin label"),
    ("semaglutide prescribing information", "GLP-1 agonist label"),
    ("empagliflozin drug label", "SGLT2 inhibitor label"),
    ("canagliflozin FDA label", "SGLT2 inhibitor label"),
]

for query, reasoning in tool_label_queries:
    additional_queries.append({
        "query": query,
        "best_source": "ToolAPISource",
        "alternative_sources": [],
        "reasoning": f"{reasoning} - OpenFDA database",
        "query_type": "label",
        "expected_result_type": "fda_label"
    })

# Specific dosing queries
tool_specific_dosing = [
    ("enoxaparin dose for DVT prophylaxis 80 kg", "VTE prophylaxis dosing"),
    ("heparin infusion for PE 90 kg patient", "Anticoagulation protocol"),
    ("vancomycin trough goal for MRSA bacteremia", "Vancomycin TDM"),
    ("gentamicin dose for UTI CrCl 50", "Aminoglycoside renal dosing"),
    ("levothyroxine starting dose for hypothyroidism", "Thyroid replacement start"),
    ("warfarin INR goal for atrial fibrillation", "Anticoagulation target"),
    ("metformin maximum daily dose", "Maximum safe dose"),
    ("insulin sliding scale for type 2 diabetes", "Insulin dosing"),
    ("prednisone taper schedule", "Steroid taper"),
    ("amiodarone loading regimen", "Antiarrhythmic loading"),
    ("digoxin loading dose for heart failure", "Cardiac glycoside loading"),
    ("phenytoin loading dose status epilepticus", "Seizure emergency dosing"),
    ("magnesium sulfate for eclampsia protocol", "Eclampsia treatment"),
    ("norepinephrine starting dose for shock", "Vasopressor initiation"),
    ("dopamine infusion for hypotension", "Vasopressor dosing"),
    ("insulin drip protocol for DKA", "DKA management"),
    ("nitroglycerin infusion for acute MI", "ACS protocol"),
    ("alteplase dose for stroke", "Thrombolytic dosing"),
    ("epinephrine dose for anaphylaxis", "Emergency dosing"),
    ("naloxone dose for opioid overdose", "Overdose reversal"),
    ("adenosine dose for SVT", "Arrhythmia conversion"),
    ("amiodarone dose for ventricular tachycardia", "VT treatment"),
    ("atropine dose for bradycardia", "Bradycardia treatment"),
    ("calcium gluconate for hyperkalemia", "Electrolyte emergency"),
    ("sodium bicarbonate for metabolic acidosis", "Acidosis correction"),
    ("dextrose 50% for hypoglycemia", "Glucose replacement"),
    ("flumazenil dose for benzodiazepine overdose", "Benzo reversal"),
    ("glucagon for beta blocker overdose", "BB overdose treatment"),
    ("fomepizole for methanol poisoning", "Toxin antidote"),
    ("N-acetylcysteine for acetaminophen overdose", "Tylenol overdose protocol"),
    ("labetalol drip for hypertensive emergency", "HTN emergency"),
    ("nicardipine infusion for blood pressure control", "IV CCB"),
    ("esmolol infusion for tachycardia", "Short-acting beta blocker"),
    ("propofol sedation dose ICU", "ICU sedation"),
    ("fentanyl PCA settings postoperative", "Pain pump settings"),
    ("morphine IV push for acute pain", "IV opioid"),
    ("hydromorphone IV dose severe pain", "Dilaudid dosing"),
    ("ketamine dose for procedural sedation", "Dissociative sedation"),
    ("etomidate dose for rapid sequence intubation", "RSI induction"),
    ("succinylcholine dose for intubation", "Paralytic dosing"),
    ("rocuronium dose for intubation", "Non-depolarizing paralytic"),
    ("vecuronium maintenance dose", "Paralytic maintenance"),
    ("cisatracurium infusion in ICU", "Long-term paralysis"),
    ("dexmedetomidine infusion for sedation", "Alpha-2 agonist sedation"),
    ("midazolam IV sedation dose", "Benzo sedation"),
    ("lorazepam IV for status epilepticus", "Seizure termination"),
    ("diazepam IV for active seizure", "Acute seizure"),
    ("levetiracetam IV loading dose", "IV anticonvulsant"),
    ("valproic acid IV for seizure", "IV valproate"),
]

for query, reasoning in tool_specific_dosing:
    additional_queries.append({
        "query": query,
        "best_source": "ToolAPISource",
        "alternative_sources": ["LLMSource"],
        "reasoning": f"{reasoning} - specific protocol or calculation",
        "query_type": "dosage",
        "expected_result_type": "specific_dose"
    })

# ============================================================================
# LLM SOURCE QUERIES (Need 147 more to reach 200)
# ============================================================================

# More mechanism of action queries
llm_mechanism_queries = [
    ("mechanism of action of loop diuretics", "Diuretic mechanism"),
    ("how do thiazide diuretics work", "Thiazide mechanism"),
    ("ARB mechanism of action", "Angiotensin receptor blocker"),
    ("how do direct renin inhibitors work", "Aliskiren mechanism"),
    ("aldosterone antagonist mechanism", "Spironolactone action"),
    ("alpha blocker mechanism of action", "Alpha-1 blocker"),
    ("how do nitrates work for angina", "Nitrate mechanism"),
    ("mechanism of antiplatelet drugs", "Antiplatelet action"),
    ("how do direct thrombin inhibitors work", "Dabigatran mechanism"),
    ("factor Xa inhibitor mechanism", "Apixaban/rivaroxaban"),
    ("SSRI mechanism of action", "Serotonin reuptake inhibition"),
    ("SNRI mechanism of action", "Dual reuptake inhibition"),
    ("tricyclic antidepressant mechanism", "TCA action"),
    ("MAOI mechanism of action", "Monoamine oxidase inhibition"),
    ("atypical antipsychotic mechanism", "D2/5HT2A antagonism"),
    ("typical antipsychotic mechanism", "D2 antagonism"),
    ("benzodiazepine mechanism of action", "GABA-A modulation"),
    ("how do Z-drugs work for sleep", "Zolpidem mechanism"),
    ("melatonin receptor agonist mechanism", "Ramelteon action"),
    ("orexin antagonist mechanism", "Suvorexant action"),
    ("stimulant mechanism for ADHD", "Amphetamine mechanism"),
    ("non-stimulant ADHD drug mechanism", "Atomoxetine action"),
    ("mood stabilizer mechanism of action", "Lithium/valproate"),
    ("anticonvulsant mechanism of action", "Multiple mechanisms"),
    ("GABA analogue mechanism", "Gabapentin/pregabalin"),
    ("sodium channel blocker mechanism", "Phenytoin action"),
    ("carbonic anhydrase inhibitor mechanism", "Acetazolamide"),
    ("anticholinergic mechanism of action", "Muscarinic antagonism"),
    ("antimuscarinic for overactive bladder", "Oxybutynin mechanism"),
    ("alpha-1 blocker for BPH mechanism", "Tamsulosin action"),
    ("5-alpha reductase inhibitor mechanism", "Finasteride action"),
    ("PDE5 inhibitor mechanism", "Sildenafil action"),
    ("H2 receptor antagonist mechanism", "Ranitidine action"),
    ("prokinetic agent mechanism", "Metoclopramide action"),
    ("5-HT3 antagonist mechanism", "Ondansetron action"),
    ("laxative mechanisms", "Different types"),
    ("antidiarrheal mechanism", "Loperamide action"),
    ("bile acid sequestrant mechanism", "Cholestyramine"),
    ("PCSK9 inhibitor mechanism", "Evolocumab action"),
    ("fibrate mechanism of action", "Gemfibrozil action"),
]

for query, reasoning in llm_mechanism_queries:
    additional_queries.append({
        "query": query,
        "best_source": "LLMSource",
        "alternative_sources": [],
        "reasoning": f"{reasoning} - requires detailed mechanistic explanation",
        "query_type": "concept",
        "expected_result_type": "mechanism_explanation"
    })

# Clinical concepts and comparisons
llm_clinical_queries = [
    ("difference between heart failure with reduced vs preserved ejection fraction", "HFrEF vs HFpEF"),
    ("STEMI vs NSTEMI differences", "ACS types"),
    ("stable vs unstable angina", "Angina classification"),
    ("systolic vs diastolic heart failure", "HF types"),
    ("primary vs secondary hypertension", "HTN classification"),
    ("type 1 vs type 2 myocardial infarction", "MI types"),
    ("acute vs chronic kidney disease", "Kidney disease types"),
    ("prerenal vs intrinsic vs postrenal AKI", "AKI classification"),
    ("nephrotic vs nephritic syndrome", "Glomerular diseases"),
    ("upper vs lower motor neuron lesions", "Neurological distinction"),
    ("ischemic vs hemorrhagic stroke", "Stroke types"),
    ("Alzheimer's vs vascular dementia", "Dementia types"),
    ("Parkinson's disease vs parkinsonism", "Movement disorder distinction"),
    ("generalized vs focal seizures", "Seizure classification"),
    ("obstructive vs restrictive lung disease", "Pulmonary disease types"),
    ("asthma vs COPD differences", "Obstructive lung diseases"),
    ("community acquired vs hospital acquired pneumonia", "Pneumonia types"),
    ("exudative vs transudative pleural effusion", "Pleural fluid classification"),
    ("Crohn's disease vs ulcerative colitis", "IBD types"),
    ("upper vs lower GI bleeding", "GI bleed location"),
    ("acute vs chronic pancreatitis", "Pancreatitis types"),
    ("viral vs bacterial meningitis", "Meningitis types"),
    ("type 1 vs type 2 diabetes pathophysiology", "Diabetes mechanisms"),
    ("DKA vs HHS differences", "Hyperglycemic emergencies"),
    ("hypothyroidism vs hyperthyroidism", "Thyroid disorders"),
    ("Graves disease vs toxic multinodular goiter", "Hyperthyroid causes"),
    ("primary vs secondary vs tertiary hypothyroidism", "Hypothyroid classification"),
    ("osteoarthritis vs rheumatoid arthritis", "Arthritis types"),
    ("gout vs pseudogout", "Crystal arthropathies"),
    ("SLE vs drug-induced lupus", "Lupus types"),
    ("Hodgkin vs non-Hodgkin lymphoma", "Lymphoma classification"),
    ("acute vs chronic leukemia", "Leukemia types"),
    ("AML vs ALL differences", "Acute leukemia"),
    ("CML vs CLL differences", "Chronic leukemia"),
    ("iron deficiency vs anemia of chronic disease", "Anemia types"),
    ("megaloblastic vs non-megaloblastic anemia", "Anemia classification"),
    ("hemolytic vs non-hemolytic anemia", "Anemia mechanisms"),
    ("thrombocytopenia causes", "Low platelet mechanisms"),
    ("primary vs secondary immunodeficiency", "Immune deficiency types"),
    ("cellulitis vs erysipelas", "Skin infections"),
]

for query, reasoning in llm_clinical_queries:
    additional_queries.append({
        "query": query,
        "best_source": "LLMSource",
        "alternative_sources": ["PDFKnowledgeSource"],
        "reasoning": f"{reasoning} - requires clinical comparison and explanation",
        "query_type": "concept",
        "expected_result_type": "clinical_comparison"
    })

# Pharmacology concepts
llm_pharmacology_queries = [
    ("explain drug clearance", "Clearance concept"),
    ("what is area under the curve in pharmacokinetics", "AUC explanation"),
    ("loading dose vs maintenance dose", "Dosing concepts"),
    ("explain zero order vs first order kinetics", "Elimination kinetics"),
    ("what is minimum effective concentration", "MEC concept"),
    ("therapeutic index explanation", "Safety margin"),
    ("explain prodrug concept", "Prodrug activation"),
    ("what is drug accumulation", "Accumulation mechanism"),
    ("dose-response relationship", "Pharmacodynamic concept"),
    ("explain potency vs efficacy", "Drug comparison"),
    ("what is an agonist vs antagonist", "Receptor activity"),
    ("partial agonist concept", "Partial activity"),
    ("inverse agonist explanation", "Negative activity"),
    ("competitive vs non-competitive antagonism", "Antagonist types"),
    ("explain allosteric modulation", "Indirect binding"),
    ("what is desensitization and downregulation", "Receptor adaptation"),
    ("explain tachyphylaxis", "Rapid tolerance"),
    ("drug-receptor binding", "Molecular interaction"),
    ("explain affinity and selectivity", "Binding properties"),
    ("what is a therapeutic window", "Safe dose range"),
    ("narrow vs wide therapeutic index", "Safety comparison"),
    ("explain peak and trough levels", "TDM concepts"),
    ("what is time to steady state", "Steady state concept"),
    ("five half-lives to steady state rule", "Pharmacokinetic principle"),
    ("explain context-sensitive half-time", "Anesthetic concept"),
    ("enterohepatic recirculation", "Drug recycling"),
    ("explain active metabolites", "Metabolite activity"),
    ("what is phase 1 vs phase 2 metabolism", "Metabolism phases"),
    ("explain conjugation reactions", "Phase 2 metabolism"),
    ("oxidation reduction hydrolysis in drug metabolism", "Phase 1 reactions"),
]

for query, reasoning in llm_pharmacology_queries:
    additional_queries.append({
        "query": query,
        "best_source": "LLMSource",
        "alternative_sources": [],
        "reasoning": f"{reasoning} - fundamental pharmacology concept",
        "query_type": "concept",
        "expected_result_type": "pharmacology_explanation"
    })

# General medical questions
llm_general_queries = [
    ("when should antibiotics be used", "Antibiotic stewardship"),
    ("how to prevent medication errors", "Patient safety"),
    ("what is polypharmacy", "Multiple medication use"),
    ("drug allergies vs side effects", "Adverse reaction types"),
    ("how to assess medication adherence", "Compliance evaluation"),
    ("what is deprescribing", "Medication reduction"),
    ("generic vs brand name drugs", "Drug naming"),
    ("biosimilar drugs explanation", "Biologic equivalents"),
    ("how to counsel patients on medications", "Patient education"),
    ("medication reconciliation importance", "Med rec process"),
    ("over-the-counter vs prescription drugs", "OTC vs Rx"),
    ("controlled substance scheduling", "DEA schedules"),
    ("drug approval process", "FDA approval"),
    ("clinical trial phases", "Drug development"),
    ("orphan drugs explanation", "Rare disease drugs"),
    ("medication storage requirements", "Storage conditions"),
    ("drug stability concerns", "Stability factors"),
    ("medication disposal guidelines", "Safe disposal"),
    ("pregnancy drug categories", "Pregnancy classification"),
    ("breastfeeding and medications", "Lactation considerations"),
    ("pediatric drug dosing principles", "Pediatric pharmacology"),
    ("geriatric prescribing considerations", "Elderly pharmacology"),
    ("renal dose adjustment principles", "Renal dosing"),
    ("hepatic dose adjustment principles", "Hepatic dosing"),
    ("Drug-food interactions", "Food effects"),
    ("alcohol and medication interactions", "Ethanol interactions"),
    ("smoking effects on drug metabolism", "Tobacco interactions"),
    ("herbal supplements and drug interactions", "Supplement interactions"),
    ("medication adherence barriers", "Non-compliance causes"),
    ("cost as barrier to medication adherence", "Affordability issues"),
    ("antibiotic resistance mechanisms", "Resistance"),
    ("vaccination principles", "Immunization"),
    ("biologic drugs vs small molecules", "Drug types"),
    ("personalized medicine", "Precision medicine"),
    ("pharmacogenomics application", "Genetic testing"),
    ("drug repurposing", "New indications"),
    ("right to try laws", "Experimental drugs"),
]

for query, reasoning in llm_general_queries:
    additional_queries.append({
        "query": query,
        "best_source": "LLMSource",
        "alternative_sources": [],
        "reasoning": f"{reasoning} - requires broad medical knowledge",
        "query_type": "general",
        "expected_result_type": "general_explanation"
    })

# ============================================================================
# PDF KNOWLEDGE SOURCE QUERIES (Need 35 more to reach 60)
# ============================================================================

pdf_document_queries = [
    ("WHO essential drugs selection criteria explained", "WHO criteria"),
    ("essential medicines concept", "Essential drug concept"),
    ("Model List of Essential Drugs purpose", "List purpose"),
    ("rational drug use principles", "Rational use"),
    ("drug procurement guidelines", "Procurement"),
    ("pharmaceutical supply management", "Supply chain"),
    ("quality assurance of pharmaceuticals", "Quality standards"),
    ("drug information resources", "Information sources"),
    ("pharmacovigilance principles", "Safety monitoring"),
    ("adverse drug reaction reporting", "ADR reporting"),
    ("medication safety systems", "Safety systems"),
    ("behavioral health medication guidelines", "Mental health meds"),
    ("substance use disorder pharmacotherapy", "Addiction treatment"),
    ("opioid use disorder medications", "OUD treatment"),
    ("alcohol use disorder pharmacotherapy", "AUD treatment"),
    ("stimulant use disorder treatment", "Stimulant addiction"),
    ("tobacco cessation medications", "Smoking cessation"),
    ("harm reduction strategies", "Harm reduction"),
    ("medication-assisted treatment principles", "MAT principles"),
    ("psychiatric medication management", "Psych med management"),
    ("commonly abused drug categories", "Abuse potential"),
    ("signs of substance intoxication", "Intoxication signs"),
    ("withdrawal syndrome management", "Withdrawal treatment"),
    ("overdose prevention strategies", "Overdose prevention"),
    ("naloxone distribution programs", "Overdose reversal programs"),
    ("drug scheduling rationale", "DEA scheduling"),
    ("prescription drug monitoring programs", "PDMP systems"),
    ("cardiovascular drug classification", "CV drug categories"),
    ("antihypertensive drug classes", "HTN drug classes"),
    ("antianginal medication options", "Angina treatment"),
    ("heart failure medication classes", "HF drug classes"),
    ("antiarrhythmic drug classification", "Vaughan Williams"),
    ("anticoagulant drug categories", "Anticoagulant types"),
    ("antiplatelet therapy options", "Antiplatelet drugs"),
    ("lipid-lowering medication classes", "Dyslipidemia treatment"),
]

for query, reasoning in pdf_document_queries:
    additional_queries.append({
        "query": query,
        "best_source": "PDFKnowledgeSource",
        "alternative_sources": ["LLMSource"],
        "reasoning": f"{reasoning} - content from provided PDF documents",
        "query_type": "document",
        "expected_result_type": "document_content"
    })

# Combine existing and additional queries
print(f"\nGenerating {len(additional_queries)} additional queries...")
all_queries = existing_data + additional_queries

# Verify distribution
from collections import Counter
final_by_source = Counter([q['best_source'] for q in all_queries])
final_by_type = Counter([q['query_type'] for q in all_queries])

print(f"\n{'='*70}")
print(f"FINAL DATASET: {len(all_queries)} queries")
print(f"{'='*70}")
print(f"\nBy Source:")
for source in sorted(final_by_source.keys()):
    count = final_by_source[source]
    target = TARGET_BY_SOURCE.get(source, 0)
    status = "✓" if count >= target else f"(need {target - count} more)"
    print(f"  {source:30s}: {count:3d} / {target:3d} {status}")

print(f"\nBy Query Type:")
for qtype in sorted(final_by_type.keys()):
    print(f"  {qtype:20s}: {final_by_type[qtype]:3d}")

# Save to file
output_file = "data/training_dataset_600.json"
with open(output_file, 'w') as f:
    json.dump(all_queries, f, indent=2)

print(f"\n✓ Saved to {output_file}")
print(f"{'='*70}\n")

print("Dataset ready for training!")
print("Next steps:")
print("  1. Review queries for medical accuracy")
print("  2. Adjust alternative_sources if needed")
print("  3. Run Phase 1 supervised training")