# Figures out if the agent picked the right source and assigns a reward

import re


# Which sources are good for which types of queries
SOURCE_PREFERENCES = {
    "KnowledgeGraphSource": ["interaction", "treatment"],
    "ToolAPISource": ["dosage", "label", "interaction"],
    "LLMSource": ["concept", "general"],
    "PDFKnowledgeSource": ["document", "research", "concept"],
}


def classify_query(query):
    """
    Look at the query text and figure out what type of question it is.
    Returns one of: interaction, dosage, label, treatment, document, research, concept, etc.
    """
    q = query.lower()

    # Drug interaction questions
    if any(word in q for word in ['interact', 'combination', 'together', 'contraindication', 'drug-drug']):
        return 'interaction'

    # Calculation or dosage questions
    if any(word in q for word in ['bmi', 'dose', 'dosage', 'calculate', 'creatinine clearance', 'crcl',
                                    'weight', 'kg', 'pediatric dose', 'body mass', 'infusion', 'mg', 'protocol']):
        return 'dosage'

    # FDA label or prescribing info
    if any(word in q for word in ['fda', 'label', 'prescribing information', 'indication', 'boxed warning',
                                    'black box', 'approved use']):
        return 'label'

    # Treatment recommendations
    if any(word in q for word in ['medications for', 'drugs for', 'treatments for', 'treatment of',
                                    'medication for', 'drug for', 'treat']):
        return 'treatment'

    # Asking about documents or papers
    if any(word in q for word in ['paper', 'document', 'pdf', 'krr', 'ontolog', 'who essential',
                                    'bhmeds', 'nida', 'clincalc']):
        return 'document'

    # Research or literature questions
    if any(word in q for word in ['research', 'study', 'studies', 'trial', 'evidence', 'pubmed',
                                    'clinical trial', 'meta-analysis', 'systematic review', 'recent findings',
                                    'latest', 'journal']):
        return 'research'

    # Mechanism or explanation questions
    if any(word in q for word in ['how does', 'mechanism', 'explain', 'what is', 'why', 'difference between',
                                    'pathophysiology', 'work', 'action', 'concept', 'principle']):
        return 'concept'

    # Side effects
    if any(word in q for word in ['side effect', 'adverse', 'reaction', 'toxicity', 'safety']):
        return 'side_effects'

    return 'general'


class RewardEvaluator:
    """Decides how good the agent's source selection was"""

    @staticmethod
    def compute_reward(query, source_name, results):
        """
        Give the agent a reward based on whether it picked a good source.

        Reward scheme:
          +1.0 = perfect choice, got good results
          +0.5 = acceptable alternative source, got results
          +0.3 = right source but nothing came back (weird query maybe)
          +0.2 = wrong source but somehow got something
          -0.3 = wrong source and no results (bad pick)
        """
        qtype = classify_query(query)
        got_results = RewardEvaluator._check_if_useful(results)

        # Is this source a primary choice for this query type?
        is_good_match = qtype in SOURCE_PREFERENCES.get(source_name, [])

        # Is it at least an acceptable backup?
        is_ok_match = RewardEvaluator._is_acceptable_backup(qtype, source_name)

        # Assign reward based on match + result quality
        if is_good_match and got_results:
            return 1.0

        if is_ok_match and got_results:
            return 0.5

        if is_good_match and not got_results:
            return 0.3

        if not is_good_match and got_results:
            return 0.2

        return -0.3

    @staticmethod
    def _check_if_useful(results):
        """Did we actually get useful results back?"""
        if results is None:
            return False

        if isinstance(results, list):
            return len(results) > 0

        if isinstance(results, dict):
            # Tool API and others return dicts with 'answer' or 'result'
            if 'answer' in results:
                return bool(results['answer'])
            if 'result' in results:
                return results['result'] is not None
            return len(results) > 0

        if isinstance(results, str):
            return len(results.strip()) > 10

        return bool(results)

    @staticmethod
    def _is_acceptable_backup(qtype, source_name):
        """Is this source an OK fallback for this query type?"""
        # Some query types can be answered by multiple sources
        backups = {
            'interaction': ['KnowledgeGraphSource', 'ToolAPISource'],
            'dosage': ['ToolAPISource'],
            'label': ['ToolAPISource'],
            'treatment': ['KnowledgeGraphSource', 'LLMSource'],
            'concept': ['LLMSource', 'PDFKnowledgeSource'],
            'document': ['PDFKnowledgeSource'],
            'research': ['PDFKnowledgeSource', 'LLMSource'],
            'general': ['LLMSource'],
            'side_effects': ['ToolAPISource', 'LLMSource'],
        }

        return source_name in backups.get(qtype, [])