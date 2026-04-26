"""
LLM Source - Parametric Knowledge
Integrates various language models for conceptual explanations and reasoning

MODEL COMPARISON FOR MEDICAL TASKS:
====================================

1. **Commercial Models (API-based)**
   - GPT-4 / GPT-4 Turbo (OpenAI): Best overall, excellent medical reasoning
   - Claude 3 Opus/Sonnet (Anthropic): Strong medical knowledge, safety-focused
   - Pros: High quality, easy to use, constantly updated
   - Cons: Cost per query, API dependency, data privacy concerns

2. **Medical-Specific Models**
   - BioGPT: 1.5B params, trained on PubMed
     * Pros: Medical domain-specific, good for generation
     * Cons: Smaller than general models, may lack broad reasoning

   - MedPaLM 2 (Google): State-of-art medical QA
     * Pros: Best medical accuracy, physician-level performance
     * Cons: Not publicly available yet

   - BioClinicalBERT: Encoder-only (not for generation)
     * Use for: Classification, NER, embeddings only

3. **Open-Source General Models**
   - **Llama 2 (7B-70B)**:
     * Pros: Free, can run locally, decent general knowledge
     * Cons: Not specialized for medical, hallucination risk, needs GPU
     * Verdict: OK for general explanations, NOT ideal for medical facts

   - Llama 2 Medical Fine-tuned versions:
     * Med-Llama: Llama 2 fine-tuned on medical data
     * Pros: Better medical knowledge than base Llama 2
     * Still not as good as commercial or BioGPT for medical

   - Mistral 7B: Strong performance for size
     * Similar to Llama 2 but more efficient
     * Same limitations for medical domain

4. **Google Gemini (API-based)**
   - Gemini 2.0 Flash: Fast, free tier (15 RPM, 1M tokens/day)
     * Pros: Free, strong medical reasoning, easy API
     * Cons: Rate limits on free tier

RECOMMENDATION FOR THIS PROJECT:
================================
Best Approach: Hybrid
- Query Encoding: sentence-transformers/all-MiniLM-L6-v2 (as specified)
- Generation:
  * Primary: Gemini (free tier) or GPT-4 (paid) - best quality
  * Alternative: BioGPT - medical-specific, can run locally
  * Budget/Offline: Llama 2 7B/13B - but expect lower accuracy

For Llama 2: If you must use it, fine-tune on medical QA dataset first!
"""

import os
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class LLMSource:
    """
    Provides access to Language Model for conceptual explanations
    """

    def __init__(self, model_type: str = "groq", api_key: str = None):
        """
        Initialize LLM source

        Args:
            model_type: 'groq', 'gemini', 'gpt4', 'claude', 'biogpt', 'llama2', 'mistral'
            api_key: API key for commercial models
        """
        self.model_type = model_type
        # Resolve API key based on model type
        if api_key:
            self.api_key = api_key
        elif model_type == "groq":
            self.api_key = os.getenv("GROQ_API_KEY")
        elif model_type == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY")
        elif model_type == "claude":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        # Groq client (primary or fallback)
        self.groq_client = None
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        self._initialize_model()
        # Setup Groq as fallback for non-groq models too
        if self.model_type != "groq":
            self._setup_groq_fallback()

    def _initialize_model(self):
        """Initialize the selected model"""
        if self.model_type == "groq":
            self._setup_groq_primary()
        elif self.model_type == "gemini":
            self._setup_gemini()
        elif self.model_type == "gpt4":
            self._setup_gpt4()
        elif self.model_type == "claude":
            self._setup_claude()
        elif self.model_type == "biogpt":
            self._setup_biogpt()
        elif self.model_type == "llama2":
            self._setup_llama2()
        elif self.model_type == "mistral":
            self._setup_mistral()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _setup_groq_primary(self):
        """Setup Groq as the primary LLM"""
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=self.api_key or self.groq_api_key)
            print(f"Groq initialized as PRIMARY LLM (model: llama-3.3-70b-versatile)")
        except ImportError:
            print("Error: Install groq package: pip install groq")
        except Exception as e:
            print(f"Error initializing Groq: {e}")

    def _setup_gemini(self):
        """Setup Google Gemini"""
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self.gemini_model = "gemini-2.0-flash"
            print(f"Gemini initialized successfully (model: {self.gemini_model})")
        except ImportError:
            print("Error: Install google-genai package: pip install google-genai")

    def _setup_groq_fallback(self):
        """Setup Groq as automatic fallback for when primary LLM hits quota"""
        if not self.groq_api_key:
            return
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=self.groq_api_key)
            print("Groq fallback initialized (model: llama-3.3-70b-versatile)")
        except ImportError:
            print("Groq fallback not available (install: pip install groq)")
        except Exception as e:
            print(f"Groq fallback setup failed: {e}")

    def _setup_gpt4(self):
        """Setup OpenAI GPT-4"""
        try:
            import openai
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
            print("GPT-4 initialized successfully")
        except ImportError:
            print("Error: Install openai package: pip install openai")

    def _setup_claude(self):
        """Setup Anthropic Claude"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
            print("Claude initialized successfully")
        except ImportError:
            print("Error: Install anthropic package: pip install anthropic")

    def _setup_biogpt(self):
        """Setup BioGPT - Medical-specific model"""
        try:
            print("Loading BioGPT... (this may take a moment)")
            model_name = "microsoft/biogpt"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)

            print(f"BioGPT initialized on {device}")
        except Exception as e:
            print(f"Error loading BioGPT: {e}")

    def _setup_llama2(self):
        """
        Setup Llama 2

        Note: Requires HuggingFace token and model access approval
        """
        try:
            print("Loading Llama 2... (this may take several minutes)")
            # Using 7B model (smallest). For better quality, use 13B or 70B
            model_name = "meta-llama/Llama-2-7b-chat-hf"

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=os.getenv("HUGGINGFACE_TOKEN"),
                torch_dtype=torch.float16,
                device_map="auto"
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )

            print("Llama 2 initialized successfully")
        except Exception as e:
            print(f"Error loading Llama 2: {e}")
            print("Note: You need HuggingFace token and model access approval")

    def _setup_mistral(self):
        """Setup Mistral 7B"""
        try:
            print("Loading Mistral... (this may take several minutes)")
            model_name = "mistralai/Mistral-7B-Instruct-v0.1"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )

            print("Mistral initialized successfully")
        except Exception as e:
            print(f"Error loading Mistral: {e}")

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate text using the selected model

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            Generated text
        """
        if self.model_type == "groq":
            return self._generate_groq(prompt, max_tokens, temperature)
        elif self.model_type == "gemini":
            return self._generate_gemini(prompt, max_tokens, temperature)
        elif self.model_type == "gpt4":
            return self._generate_gpt4(prompt, max_tokens, temperature)
        elif self.model_type == "claude":
            return self._generate_claude(prompt, max_tokens, temperature)
        elif self.model_type in ["biogpt", "llama2", "mistral"]:
            return self._generate_huggingface(prompt, max_tokens, temperature)
        else:
            return "Model not initialized"

    def _generate_gemini(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Google Gemini, with automatic Groq fallback on quota errors"""
        try:
            from google.genai import types
            medical_prompt = (
                "You are a medical expert assistant. "
                "Provide accurate, evidence-based information.\n\n" + prompt
            )
            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=medical_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            return response.text
        except Exception as e:
            # Fall back to Groq for ANY Gemini error (quota, timeout, 500, etc.)
            if self.groq_client:
                reason = "quota hit" if ('429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e)) else str(type(e).__name__)
                print(f"  ⚡ Gemini {reason} — falling back to Groq")
                return self._generate_groq(prompt, max_tokens, temperature)
            return f"Error generating with Gemini: {e}"

    def _generate_groq(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Groq (Llama 3.3 70B)"""
        if not self.groq_client:
            return "Error: Groq client not initialized. Set GROQ_API_KEY in your environment."
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a medical expert assistant. Provide accurate, evidence-based information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating with Groq: {e}"

    def _generate_gpt4(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using GPT-4"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",  # or "gpt-4"
                messages=[
                    {"role": "system", "content": "You are a medical expert assistant. Provide accurate, evidence-based information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating with GPT-4: {e}"

    def _generate_claude(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Claude"""
        try:
            message = self.client.messages.create(
                model="claude-3-opus-20240229",  # or "claude-3-sonnet-20240229"
                max_tokens=max_tokens,
                temperature=temperature,
                system="You are a medical expert assistant. Provide accurate, evidence-based information.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error generating with Claude: {e}"

    def _generate_huggingface(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using HuggingFace models (BioGPT, Llama 2, Mistral)"""
        try:
            if self.model_type == "llama2":
                # Llama 2 chat format
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            else:
                formatted_prompt = prompt

            if self.pipeline:
                # Use pipeline for Llama 2 and Mistral
                outputs = self.pipeline(
                    formatted_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=1
                )
                return outputs[0]['generated_text'].replace(formatted_prompt, '').strip()
            else:
                # Use model directly for BioGPT
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return generated_text.replace(formatted_prompt, '').strip()

        except Exception as e:
            return f"Error generating: {e}"

    def explain_concept(self, concept: str) -> str:
        """
        Explain a medical concept

        Args:
            concept: The concept to explain

        Returns:
            Explanation
        """
        prompt = f"""Explain the following medical concept in clear, accurate terms:

Concept: {concept}

Provide a concise but comprehensive explanation suitable for healthcare professionals."""

        return self.generate(prompt, max_tokens=400, temperature=0.3)

    def answer_question(self, question: str) -> str:
        """
        Answer a medical question

        Args:
            question: The question to answer

        Returns:
            Answer
        """
        prompt = f"""Answer the following medical question accurately and concisely:

Question: {question}

Answer:"""

        return self.generate(prompt, max_tokens=300, temperature=0.3)

    def summarize_text(self, text: str) -> str:
        """
        Summarize medical text

        Args:
            text: Text to summarize

        Returns:
            Summary
        """
        prompt = f"""Summarize the following medical text, focusing on key points:

Text: {text}

Summary:"""

        return self.generate(prompt, max_tokens=200, temperature=0.3)

    def query(self, text: str) -> Dict[str, Any]:
        """
        Unified query interface — takes any text query, generates an
        answer using the configured LLM, and returns a standardised dict.

        Args:
            text: Free-text medical question

        Returns:
            Dict with 'answer' (str) and 'source' name
        """
        answer = self.answer_question(text)
        return {
            "source": "LLMSource",
            "answer": answer,
        }


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  LLM SOURCE — Standalone Test")
    print("=" * 70)

    llm = LLMSource(model_type="groq")

    queries = [
        "How does insulin regulate blood sugar?",
        "What are the side effects of metformin?",
        "Explain the mechanism of action of aspirin",
    ]

    for q in queries:
        print(f"\n{'─' * 60}")
        print(f"🔎 Query: {q}")
        print(f"{'─' * 60}")
        out = llm.query(q)
        print(out["answer"][:300])

    print(f"\n{'=' * 70}")
    print("✅ Done")
    print(f"{'=' * 70}")

