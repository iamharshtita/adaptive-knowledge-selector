"""
Tool/API Source - Deterministic Computations
Integrates OpenFDA, RxNorm, and custom calculators for drug dosing and interactions
"""

import requests
from typing import Dict, Any, Optional, List
import json


class ToolAPISource:
    """
    Provides access to medical calculation tools and drug information APIs
    """

    def __init__(self):
        """Initialize the tool/API source"""
        self.openfda_base = "https://api.fda.gov/drug"
        self.rxnorm_base = "https://rxnav.nlm.nih.gov/REST"

    # ==================== OpenFDA APIs ====================

    def search_drug_label(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        Search OpenFDA for drug label information

        Args:
            drug_name: Name of the drug

        Returns:
            Drug label information including indications, dosage, warnings
        """
        url = f"{self.openfda_base}/label.json"
        params = {
            'search': f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
            'limit': 1
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('results'):
                result = data['results'][0]
                return {
                    'brand_name': result.get('openfda', {}).get('brand_name', [''])[0],
                    'generic_name': result.get('openfda', {}).get('generic_name', [''])[0],
                    'indications': result.get('indications_and_usage', [''])[0] if result.get('indications_and_usage') else '',
                    'dosage': result.get('dosage_and_administration', [''])[0] if result.get('dosage_and_administration') else '',
                    'warnings': result.get('warnings', [''])[0] if result.get('warnings') else '',
                    'adverse_reactions': result.get('adverse_reactions', [''])[0] if result.get('adverse_reactions') else ''
                }

            return None

        except Exception as e:
            print(f"Error searching OpenFDA: {e}")
            return None

    def search_drug_adverse_events(self, drug_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for adverse events reported for a drug

        Args:
            drug_name: Name of the drug
            limit: Maximum number of results

        Returns:
            List of adverse event reports
        """
        url = f"{self.openfda_base}/event.json"
        params = {
            'search': f'patient.drug.medicinalproduct:"{drug_name}"',
            'limit': limit
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            events = []
            for result in data.get('results', []):
                patient = result.get('patient', {})
                reactions = patient.get('reaction', [])

                events.append({
                    'reactions': [r.get('reactionmeddrapt', 'Unknown') for r in reactions],
                    'serious': result.get('serious', 0),
                    'outcome': result.get('patient', {}).get('patientoutcome', 'Unknown')
                })

            return events

        except Exception as e:
            print(f"Error searching adverse events: {e}")
            return []

    # ==================== RxNorm APIs ====================

    def get_rxnorm_id(self, drug_name: str) -> Optional[str]:
        """
        Get RxNorm Concept Unique Identifier (RXCUI) for a drug

        Args:
            drug_name: Name of the drug

        Returns:
            RXCUI or None
        """
        url = f"{self.rxnorm_base}/rxcui.json"
        params = {'name': drug_name}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            rxcui_list = data.get('idGroup', {}).get('rxnormId', [])
            return rxcui_list[0] if rxcui_list else None

        except Exception as e:
            print(f"Error getting RxNorm ID: {e}")
            return None

    def get_drug_interactions(self, drug_name: str) -> List[Dict[str, str]]:
        """
        Get drug interactions using RxNorm

        Args:
            drug_name: Name of the drug

        Returns:
            List of drug interactions
        """
        # First get RXCUI
        rxcui = self.get_rxnorm_id(drug_name)
        if not rxcui:
            return []

        url = f"{self.rxnorm_base}/interaction/interaction.json"
        params = {'rxcui': rxcui}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            interactions = []
            for group in data.get('interactionTypeGroup', []):
                for interaction_type in group.get('interactionType', []):
                    for pair in interaction_type.get('interactionPair', []):
                        interactions.append({
                            'drug': pair.get('interactionConcept', [{}])[0].get('minConceptItem', {}).get('name', 'Unknown'),
                            'severity': pair.get('severity', 'Unknown'),
                            'description': pair.get('description', 'No description available')
                        })

            return interactions

        except Exception as e:
            print(f"Error getting drug interactions: {e}")
            return []

    def get_drug_properties(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        Get drug properties from RxNorm

        Args:
            drug_name: Name of the drug

        Returns:
            Drug properties including strength, dose forms
        """
        rxcui = self.get_rxnorm_id(drug_name)
        if not rxcui:
            return None

        url = f"{self.rxnorm_base}/rxcui/{rxcui}/properties.json"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            props = data.get('properties', {})
            return {
                'name': props.get('name', drug_name),
                'synonym': props.get('synonym', ''),
                'tty': props.get('tty', ''),  # Term type
                'language': props.get('language', 'ENG')
            }

        except Exception as e:
            print(f"Error getting drug properties: {e}")
            return None

    # ==================== Medical Calculators ====================

    def calculate_bmi(self, weight_kg: float, height_m: float) -> Dict[str, Any]:
        """
        Calculate Body Mass Index

        Args:
            weight_kg: Weight in kilograms
            height_m: Height in meters

        Returns:
            BMI value and category
        """
        bmi = weight_kg / (height_m ** 2)

        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal weight"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"

        return {
            'bmi': round(bmi, 2),
            'category': category
        }

    def calculate_creatinine_clearance(self, age: int, weight_kg: float,
                                       serum_creatinine: float, is_female: bool = False) -> Dict[str, Any]:
        """
        Calculate Creatinine Clearance using Cockcroft-Gault equation

        Args:
            age: Age in years
            weight_kg: Weight in kilograms
            serum_creatinine: Serum creatinine in mg/dL
            is_female: Whether patient is female

        Returns:
            Creatinine clearance in mL/min
        """
        # Cockcroft-Gault formula
        crcl = ((140 - age) * weight_kg) / (72 * serum_creatinine)

        if is_female:
            crcl *= 0.85

        # Interpretation
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
        """
        Calculate Ideal Body Weight using Devine formula

        Args:
            height_cm: Height in centimeters
            is_female: Whether patient is female

        Returns:
            Ideal body weight in kg
        """
        height_inches = height_cm / 2.54

        if is_female:
            ibw = 45.5 + 2.3 * (height_inches - 60)
        else:
            ibw = 50 + 2.3 * (height_inches - 60)

        return {
            'ideal_body_weight': round(ibw, 2),
            'unit': 'kg'
        }

    def calculate_pediatric_dose(self, adult_dose_mg: float, child_weight_kg: float,
                                  method: str = 'clark') -> Dict[str, Any]:
        """
        Calculate pediatric drug dose

        Args:
            adult_dose_mg: Adult dose in mg
            child_weight_kg: Child's weight in kg
            method: 'clark' or 'young' rule

        Returns:
            Pediatric dose
        """
        if method == 'clark':
            # Clark's Rule: (Weight in kg / 70) × Adult Dose
            pediatric_dose = (child_weight_kg / 70) * adult_dose_mg
        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            'pediatric_dose': round(pediatric_dose, 2),
            'unit': 'mg',
            'method': method,
            'warning': 'Always verify with clinical guidelines'
        }

    def check_drug_pregnancy_category(self, drug_name: str) -> Optional[Dict[str, str]]:
        """
        Check pregnancy category for a drug (using FDA data)

        Note: This is a simplified version. Real implementation would query comprehensive database.

        Args:
            drug_name: Name of the drug

        Returns:
            Pregnancy category information
        """
        # In production, this would query a real database
        # For now, return structure only
        categories = {
            'A': 'Adequate and well-controlled studies show no risk',
            'B': 'Animal studies show no risk, human studies inadequate',
            'C': 'Animal studies show adverse effect, human studies inadequate',
            'D': 'Evidence of human fetal risk, but benefits may warrant use',
            'X': 'Fetal abnormalities demonstrated, contraindicated in pregnancy'
        }

        return {
            'drug': drug_name,
            'category': 'Query pregnancy database',
            'description': 'Implementation needed with real database'
        }


# Example usage
if __name__ == "__main__":
    # Initialize
    tool_api = ToolAPISource()

    # Test OpenFDA drug label
    print("\n=== OpenFDA Drug Label Search ===")
    label = tool_api.search_drug_label("aspirin")
    if label:
        print(f"Brand: {label['brand_name']}")
        print(f"Generic: {label['generic_name']}")
        print(f"Indications: {label['indications'][:200]}...")

    # Test RxNorm drug interactions
    print("\n\n=== RxNorm Drug Interactions ===")
    interactions = tool_api.get_drug_interactions("warfarin")
    for i, interaction in enumerate(interactions[:5], 1):
        print(f"{i}. {interaction['drug']}: {interaction['severity']}")
        print(f"   {interaction['description'][:100]}...")

    # Test medical calculators
    print("\n\n=== Medical Calculators ===")

    # BMI
    bmi_result = tool_api.calculate_bmi(weight_kg=70, height_m=1.75)
    print(f"\nBMI: {bmi_result['bmi']} - {bmi_result['category']}")

    # Creatinine Clearance
    crcl_result = tool_api.calculate_creatinine_clearance(
        age=65,
        weight_kg=80,
        serum_creatinine=1.2,
        is_female=False
    )
    print(f"\nCreatinine Clearance: {crcl_result['creatinine_clearance']} {crcl_result['unit']}")
    print(f"Kidney Function: {crcl_result['kidney_function']}")

    # Ideal Body Weight
    ibw_result = tool_api.calculate_ideal_body_weight(height_cm=175, is_female=False)
    print(f"\nIdeal Body Weight: {ibw_result['ideal_body_weight']} {ibw_result['unit']}")

    # Pediatric Dose
    ped_dose = tool_api.calculate_pediatric_dose(
        adult_dose_mg=400,
        child_weight_kg=20,
        method='clark'
    )
    print(f"\nPediatric Dose: {ped_dose['pediatric_dose']} {ped_dose['unit']}")
    print(f"Method: {ped_dose['method']}")
    print(f"Warning: {ped_dose['warning']}")
