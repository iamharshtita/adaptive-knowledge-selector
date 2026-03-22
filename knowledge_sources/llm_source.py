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

RECOMMENDATION FOR THIS PROJECT:
================================
Best Approach: Hybrid
- Query Encoding: sentence-transformers/all-MiniLM-L6-v2 (as specified)
- Generation:
  * Primary: GPT-4 or Claude (via API) - best quality
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

    def __init__(self, model_type: str = "gpt4", api_key: str = None):
        """
        Initialize LLM source

        Args:
            model_type: 'gpt4', 'claude', 'biogpt', 'llama2', 'mistral'
            api_key: API key for commercial models
        """
        self.model_type = model_type
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the selected model"""
        if self.model_type == "gpt4":
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
        if self.model_type == "gpt4":
            return self._generate_gpt4(prompt, max_tokens, temperature)
        elif self.model_type == "claude":
            return self._generate_claude(prompt, max_tokens, temperature)
        elif self.model_type in ["biogpt", "llama2", "mistral"]:
            return self._generate_huggingface(prompt, max_tokens, temperature)
        else:
            return "Model not initialized"

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


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("LLM Source - Model Comparison Demo")
    print("=" * 70)

    # Test with different models
    # Note: You'll need appropriate API keys or model access

    # Example 1: Using GPT-4 (requires API key)
    print("\n### Testing GPT-4 ###")
    try:
        llm_gpt4 = LLMSource(model_type="gpt4", api_key=os.getenv("OPENAI_API_KEY"))
        response = llm_gpt4.explain_concept("How does insulin work in the body?")
        print(f"GPT-4 Response:\n{response}\n")
    except Exception as e:
        print(f"GPT-4 not available: {e}\n")

    # Example 2: Using BioGPT (free, runs locally)
    print("\n### Testing BioGPT (Medical-Specific) ###")
    try:
        llm_biogpt = LLMSource(model_type="biogpt")
        response = llm_biogpt.explain_concept("What is diabetes mellitus?")
        print(f"BioGPT Response:\n{response}\n")
    except Exception as e:
        print(f"BioGPT not available: {e}\n")

    # Example 3: Using Llama 2 (requires HuggingFace token)
    print("\n### Testing Llama 2 ###")
    print("Note: Llama 2 is NOT specialized for medical content")
    print("Expected quality: Lower than GPT-4 or BioGPT for medical tasks")
    try:
        llm_llama = LLMSource(model_type="llama2")
        response = llm_llama.answer_question("What drugs interact with aspirin?")
        print(f"Llama 2 Response:\n{response}\n")
    except Exception as e:
        print(f"Llama 2 not available: {e}\n")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    print("1. For BEST quality: Use GPT-4 or Claude (commercial APIs)")
    print("2. For MEDICAL-specific: Use BioGPT (free, medical-trained)")
    print("3. For GENERAL/offline: Use Llama 2 (but expect lower medical accuracy)")
    print("4. For THIS PROJECT: Consider GPT-4 API for LLM source")
    print("=" * 70)
