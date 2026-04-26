import os
import pickle
import re
import requests
from typing import Dict, Any, Optional, List
from sentence_transformers import SentenceTransformer

_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tool_selector", "selector.pkl")

class ToolFunctionSelector:
    def __init__(self, model_path: str = _MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found. Run python scripts/train_tool_selector.py"
            )
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        self.clf = bundle['clf']
        self.le = bundle['le']
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def predict(self, query: str) -> str:
        vec = self.encoder.encode([query], show_progress_bar=False)
        idx = self.clf.predict(vec)[0]
        return self.le.inverse_transform([idx])[0]

_DRUG_STOPWORDS = {
    'the', 'a', 'an', 'of', 'drug', 'medication', 'medicine', 'label',
    'information', 'data', 'report', 'reports', 'details', 'profile',
    'properties', 'class', 'classification', 'type', 'about', 'on',
    'my', 'your', 'this', 'that', 'any', 'all', 'me', 'give', 'show',
    'check', 'list', 'search', 'find', 'get', 'compute', 'calculate',
}

def fetch_drug_name(text: str) -> Optional[str]:
    pattern = r'\b(?:for|of|about|on|with|regarding|check|lookup|search|is)\b\s+([A-Za-z][A-Za-z0-9\-]{1,30}(?:\s+[A-Za-z][A-Za-z0-9\-]{1,30}){0,2})'
    for m in re.finditer(pattern, text, re.IGNORECASE):
        match = m.group(1).strip()
        if match.split()[0].lower() not in _DRUG_STOPWORDS:
            return match
    # fallback
    for word in reversed(re.findall(r'\b[A-Za-z]{3,}\b', text)):
        if word.lower() not in _DRUG_STOPWORDS:
            return word
    return None

def fetch_weight_kg(text: str) -> Optional[float]:
    # skip weights that belong to a child
    child_match = re.search(
        r'(\d+(?:\.\d+)?)\s*kg\b.{0,30}(?:child|pediatric|infant|baby)|'
        r'(?:child|pediatric|infant|baby).{0,30}(\d+(?:\.\d+)?)\s*kg\b',
        text, re.IGNORECASE
    )
    child_val = float(child_match.group(1) or child_match.group(2)) if child_match else None
    for m in re.findall(r'(\d+(?:\.\d+)?)\s*kg', text, re.IGNORECASE):
        val = float(m)
        if child_val is None or val != child_val:
            return val
    return None

def fetch_height(text: str) -> Optional[float]:
    # returns height in metres regardless of input unit
    for m in re.findall(r'(\d+(?:\.\d+)?)\s*m(?!g|l|m)', text, re.IGNORECASE):
        val = float(m)
        if 1.0 <= val <= 3.0:
            return val
    for m in re.findall(r'(\d+(?:\.\d+)?)\s*cm', text, re.IGNORECASE):
        val = float(m)
        if 100.0 <= val <= 250.0:
            return round(val / 100, 4)
    return None

def fetch_age(text: str) -> Optional[int]:
    for pat in [
        r'(\d+)\s*(?:years?[\s\-]old|yo\b|yrs?\b|y/o)',
        r'age[d]?\s+(\d+)',
        r'(\d+)\s*years?\b',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None

def fetch_serum_creatinine(text: str) -> Optional[float]:
    m = re.search(
        r'(?:creatinine\s*(?:of\s*|level[s]?\s*|=\s*)?(\d+(?:\.\d+)?)'
        r'|(\d+(?:\.\d+)?)\s*(?:mg/dl\s*)?(?:serum\s*)?creatinine)',
        text, re.IGNORECASE
    )
    if m:
        return float(m.group(1) or m.group(2))
    m2 = re.search(r'\bCr\s+(\d+(?:\.\d+)?)', text)
    if m2:
        return float(m2.group(1))
    return None

def fetch_is_female(text: str) -> Optional[bool]:
    if re.search(r'\b(?:female|woman|women|girl|she|her)\b', text, re.IGNORECASE):
        return True
    if re.search(r'\b(?:male|man|men|boy|he|his)\b', text, re.IGNORECASE):
        return False
    return None

def fetch_adult_dose_mg(text: str) -> Optional[float]:
    m = re.search(r'adult\s+dose\s+(?:is\s+|of\s+)?(\d+(?:\.\d+)?)\s*mg', text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    all_mg = re.findall(r'(\d+(?:\.\d+)?)\s*mg', text, re.IGNORECASE)
    return float(all_mg[0]) if all_mg else None

def fetch_child_weight_kg(text: str) -> Optional[float]:
    m = re.search(
        r'(?:child|pediatric|infant|baby|kid)\s+(?:weighing\s+|weight\s+)?(\d+(?:\.\d+)?)\s*kg',
        text, re.IGNORECASE
    )
    if m:
        return float(m.group(1))
    m2 = re.search(r'(\d+(?:\.\d+)?)\s*kg\s+(?:child|pediatric|infant|baby|kid)', text, re.IGNORECASE)
    if m2:
        return float(m2.group(1))
    if re.search(r'\b(?:child|pediatric|infant|baby|kid)\b', text, re.IGNORECASE):
        m3 = re.search(r'(\d+(?:\.\d+)?)\s*kg', text, re.IGNORECASE)
        if m3:
            return float(m3.group(1))
    return None

class ToolAPISource:

    def __init__(self):
        self.openfda_base = "https://api.fda.gov/drug"
        self.rxnorm_base = "https://rxnav.nlm.nih.gov/REST"
        self.selector = ToolFunctionSelector()

    def _normalize_drug_name(self, drug_name: str) -> str:
        # tries to resolve brand names to generic via RxNorm approximateTerm
        try:
            r = requests.get(
                f"{self.rxnorm_base}/approximateTerm.json",
                params={'term': drug_name, 'maxEntries': 1},
                timeout=10
            )
            r.raise_for_status()
            candidates = r.json().get('approximateGroup', {}).get('candidate', [])
            if not candidates:
                return drug_name

            top = candidates[0]
            resolved = top.get('name')

            if not resolved and top.get('rxcui'):
                prop_r = requests.get(
                    f"{self.rxnorm_base}/rxcui/{top['rxcui']}/properties.json", timeout=10
                )
                if prop_r.ok:
                    resolved = prop_r.json().get('properties', {}).get('name')

            if resolved and resolved.lower() != drug_name.lower():
                return resolved

        except Exception:
            pass

        return drug_name

    def search_drug_label(self, drug_name: str) -> Optional[Dict[str, Any]]:
        drug_name = self._normalize_drug_name(drug_name)
        params = {
            'search': f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
            'limit': 1
        }
        try:
            r = requests.get(f"{self.openfda_base}/label.json", params=params, timeout=10)
            r.raise_for_status()
            results = r.json().get('results')
            if not results:
                return None
            res = results[0]
            return {
                'brand_name': res.get('openfda', {}).get('brand_name', [''])[0],
                'generic_name': res.get('openfda', {}).get('generic_name', [''])[0],
                'indications': res.get('indications_and_usage', [''])[0] if res.get('indications_and_usage') else '',
                'dosage': res.get('dosage_and_administration', [''])[0] if res.get('dosage_and_administration') else '',
                'warnings': res.get('warnings', [''])[0] if res.get('warnings') else '',
                'adverse_reactions': res.get('adverse_reactions', [''])[0] if res.get('adverse_reactions') else ''
            }
        except Exception:
            return None

    def search_drug_adverse_events(self, drug_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        drug_name = self._normalize_drug_name(drug_name)
        params = {
            'search': f'patient.drug.medicinalproduct:"{drug_name}"',
            'limit': limit
        }
        try:
            r = requests.get(f"{self.openfda_base}/event.json", params=params, timeout=10)
            r.raise_for_status()
            events = []
            for result in r.json().get('results', []):
                patient = result.get('patient', {})
                events.append({
                    'reactions': [rx.get('reactionmeddrapt', 'Unknown') for rx in patient.get('reaction', [])],
                    'serious': result.get('serious', 0),
                    'outcome': patient.get('patientoutcome', 'Unknown')
                })
            return events
        except Exception:
            return []

    def calculate_bmi(self, weight_kg: float, height_m: float) -> Dict[str, Any]:
        bmi = weight_kg / (height_m ** 2)
        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal weight"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
        return {'bmi': round(bmi, 2), 'category': category}

    def calculate_creatinine_clearance(self, age: int, weight_kg: float,
                                       serum_creatinine: float, is_female: bool = False) -> Dict[str, Any]:
        crcl = ((140 - age) * weight_kg) / (72 * serum_creatinine)
        if is_female:
            crcl *= 0.85

        if crcl >= 90:
            stage = "Normal"
        elif crcl >= 60:
            stage = "Mild CKD (Stage 2)"
        elif crcl >= 30:
            stage = "Moderate CKD (Stage 3)"
        elif crcl >= 15:
            stage = "Severe CKD (Stage 4)"
        else:
            stage = "Kidney Failure (Stage 5)"

        return {
            'creatinine_clearance': round(crcl, 2),
            'unit': 'mL/min',
            'kidney_function': stage
        }

    def calculate_ideal_body_weight(self, height_cm: float, is_female: bool = False) -> Dict[str, float]:
        height_inches = height_cm / 2.54
        ibw = (45.5 if is_female else 50) + 2.3 * (height_inches - 60)
        return {'ideal_body_weight': round(ibw, 2), 'unit': 'kg'}

    def calculate_pediatric_dose(self, adult_dose_mg: float, child_weight_kg: float, method: str = 'clark') -> Dict[str, Any]:
        if method != 'clark':
            raise ValueError(f"Unknown method: {method}")
        return {
            'pediatric_dose': round((child_weight_kg / 70) * adult_dose_mg, 2),
            'unit': 'mg',
            'method': method,
            'warning': 'Always verify with clinical guidelines'
        }

    def query(self, text: str) -> Dict[str, Any]:
        fn_name = self.selector.predict(text)
        args: Dict[str, Any] = {}
        result: Any = None

        def _missing(*fields):
            return {
                'function': fn_name, 'args': {}, 'result': None,
                'answer': f"Could not extract: {', '.join(fields)}."
            }

        if fn_name == 'search_drug_label':
            drug = fetch_drug_name(text)
            if not drug:
                return _missing("drug name")
            args = {'drug_name': drug}
            result = self.search_drug_label(**args)
            if result:
                answer = (
                    f"Drug Label — {result.get('brand_name') or drug} "
                    f"({result.get('generic_name', '')}):\n"
                    f"Indications: {result.get('indications', '')[:300]}\n"
                    f"Dosage: {result.get('dosage', '')[:200]}\n"
                    f"Warnings: {result.get('warnings', '')[:200]}"
                )
            else:
                answer = f"No label found for '{drug}'."

        elif fn_name == 'search_drug_adverse_events':
            drug = fetch_drug_name(text)
            if not drug:
                return _missing("drug name")
            args = {'drug_name': drug}
            result = self.search_drug_adverse_events(**args)
            if result:
                lines = [
                    f"{i}. Reactions: {', '.join(ev.get('reactions', [])[:3])} | Serious: {ev.get('serious', '?')}"
                    for i, ev in enumerate(result[:5], 1)
                ]
                answer = f"Adverse events for {drug}:\n" + "\n".join(lines)
            else:
                answer = f"No adverse event reports found for '{drug}'."

        elif fn_name == 'calculate_bmi':
            weight = fetch_weight_kg(text)
            height = fetch_height(text)
            missing = [f for f, v in [("weight (kg)", weight), ("height (m)", height)] if v is None]
            if missing:
                return _missing(*missing)
            args = {'weight_kg': weight, 'height_m': height}
            result = self.calculate_bmi(**args)
            answer = f"BMI = {result['bmi']} ({result['category']}) for {weight} kg, {height} m."

        elif fn_name == 'calculate_creatinine_clearance':
            age    = fetch_age(text)
            weight = fetch_weight_kg(text)
            cr     = fetch_serum_creatinine(text)
            is_f   = fetch_is_female(text)
            missing = [f for f, v in [("age", age), ("weight (kg)", weight), ("serum creatinine", cr)] if v is None]
            if missing:
                return _missing(*missing)
            args = {
                'age': age, 'weight_kg': weight,
                'serum_creatinine': cr,
                'is_female': bool(is_f) if is_f is not None else False
            }
            result = self.calculate_creatinine_clearance(**args)
            answer = f"CrCl = {result['creatinine_clearance']} {result['unit']} ({result['kidney_function']})"

        elif fn_name == 'calculate_ideal_body_weight':
            height = fetch_height(text)
            is_f   = fetch_is_female(text)
            if height is None:
                return _missing("height (cm)")
            height_cm = round(height * 100, 2)
            args = {'height_cm': height_cm, 'is_female': bool(is_f) if is_f is not None else False}
            result = self.calculate_ideal_body_weight(**args)
            sex = "female" if args['is_female'] else "male"
            answer = f"Ideal Body Weight ({sex}, {height_cm} cm) = {result['ideal_body_weight']} {result['unit']}"

        elif fn_name == 'calculate_pediatric_dose':
            adult_dose   = fetch_adult_dose_mg(text)
            child_weight = fetch_child_weight_kg(text)
            missing = [f for f, v in [("adult dose (mg)", adult_dose), ("child weight (kg)", child_weight)] if v is None]
            if missing:
                return _missing(*missing)
            args = {'adult_dose_mg': adult_dose, 'child_weight_kg': child_weight}
            result = self.calculate_pediatric_dose(**args)
            answer = (
                f"Pediatric dose (Clark's Rule) = {result['pediatric_dose']} {result['unit']} "
                f"for {child_weight} kg child, adult dose {adult_dose} mg. {result['warning']}"
            )

        else:
            answer = f"Unknown function: {fn_name}"

        return {'function': fn_name, 'args': args, 'result': result, 'answer': answer}

if __name__ == "__main__":
    print("=" * 70)
    print("  TOOL API SOURCE — Standalone Test")
    print("=" * 70)

    tool = ToolAPISource()

    queries = [
        "What are the indications for aspirin?",
        "What adverse events have been reported for ibuprofen?",
        "Calculate BMI for 70 kg and 1.75 m",
        "Creatinine clearance for 65 year old male 80 kg creatinine 1.2",
        "Ideal body weight for 175 cm male",
        "Pediatric dose for a child weighing 20 kg adult dose 400 mg",
    ]

    for q in queries:
        print(f"\n{'─' * 60}")
        print(f"🔎 Query: {q}")
        print(f"{'─' * 60}")
        out = tool.query(q)
        print(f"Function: {out['function']}")
        print(f"Answer:   {out['answer'][:200]}")

    print(f"\n{'=' * 70}")
    print("✅ Done")
    print(f"{'=' * 70}")

