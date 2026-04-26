"""
LLM Source - AWS Bedrock Integration
Uses AWS Bedrock for medical question answering and conceptual explanations.

Supported Bedrock Models:
- AWS Nova 2 Lite (us.amazon.nova-lite-v1:0) - Default, Fast & Cost-effective
- AWS Nova 2 Pro (us.amazon.nova-pro-v1:0)
- Claude 3.5 Sonnet (anthropic.claude-3-5-sonnet-20240620-v1:0)
- Claude 3 Haiku (anthropic.claude-3-haiku-20240307-v1:0)
- Llama 3 70B (meta.llama3-70b-instruct-v1:0)
- Mistral Large (mistral.mistral-large-2402-v1:0)
"""

import os
import json
from typing import Dict, Any

class LLMSource:
    """
    AWS Bedrock LLM source for medical queries.
    Uses AWS credentials for authentication.
    """

    def __init__(
        self,
        model_id: str = "us.amazon.nova-lite-v1:0",
        region: str = None
    ):
        """
        Initialize AWS Bedrock LLM source

        Args:
            model_id: Bedrock model ID (default: AWS Nova 2 Lite)
            region: AWS region (default: from AWS_REGION env var or us-east-1)

        Required Environment Variables:
            AWS_ACCESS_KEY_ID: Your AWS access key
            AWS_SECRET_ACCESS_KEY: Your AWS secret key
            AWS_REGION: AWS region (optional, default: us-east-1)
        """
        self.model_id = model_id
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.bedrock_client = None

        self._initialize_bedrock()

    def _initialize_bedrock(self):
        """Initialize AWS Bedrock client"""
        try:
            import boto3

            # Create Bedrock Runtime client
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )

            print(f"✅ AWS Bedrock initialized (Region: {self.region}, Model: {self.model_id})")

        except ImportError:
            print("❌ Error: Install boto3 package: pip install boto3")
            self.bedrock_client = None
        except Exception as e:
            print(f"❌ Error initializing Bedrock: {e}")
            print("Make sure AWS credentials are configured in .env or ~/.aws/credentials")
            self.bedrock_client = None

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate text using AWS Bedrock

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)

        Returns:
            Generated text
        """
        if not self.bedrock_client:
            return "Error: Bedrock client not initialized. Check AWS credentials."

        try:
            # Add medical context to prompt
            medical_prompt = (
                "You are a medical expert assistant. "
                "Provide accurate, evidence-based information.\n\n" + prompt
            )

            # Prepare request based on model family
            if "anthropic" in self.model_id:
                # Claude models
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": medical_prompt
                        }
                    ]
                }
            elif "amazon.nova" in self.model_id:
                # AWS Nova models
                request_body = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": medical_prompt}]
                        }
                    ],
                    "inferenceConfig": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature
                    }
                }
            elif "meta.llama" in self.model_id:
                # Llama models
                request_body = {
                    "prompt": medical_prompt,
                    "max_gen_len": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9
                }
            elif "mistral" in self.model_id:
                # Mistral models
                request_body = {
                    "prompt": medical_prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            else:
                return f"Error: Unsupported model family for {self.model_id}"

            # Invoke model
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )

            # Parse response
            response_body = json.loads(response['body'].read())

            # Extract text based on model family
            if "anthropic" in self.model_id:
                return response_body['content'][0]['text']
            elif "amazon.nova" in self.model_id:
                return response_body['output']['message']['content'][0]['text']
            elif "meta.llama" in self.model_id:
                return response_body['generation']
            elif "mistral" in self.model_id:
                return response_body['outputs'][0]['text']
            else:
                return str(response_body)

        except Exception as e:
            return f"Error generating with Bedrock: {e}"

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
        answer using AWS Bedrock, and returns a standardized dict.

        Args:
            text: Free-text medical question

        Returns:
            Dict with 'answer' (str) and 'source' name
        """
        answer = self.answer_question(text)
        return {
            "source": "LLMSource (AWS Bedrock)",
            "answer": answer,
            "model": self.model_id,
            "region": self.region
        }


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    print("=" * 70)
    print("  LLM SOURCE — AWS Bedrock Standalone Test")
    print("=" * 70)

    # Check if AWS credentials are configured
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        print("\n❌ AWS credentials not found!")
        print("\nPlease configure AWS credentials:")
        print("1. Add to .env file:")
        print("   AWS_ACCESS_KEY_ID=your_access_key")
        print("   AWS_SECRET_ACCESS_KEY=your_secret_key")
        print("   AWS_REGION=us-east-1")
        print("\n2. Or configure AWS CLI:")
        print("   aws configure")
        sys.exit(1)

    # Initialize with AWS Nova 2 Lite (fast & cost-effective)
    llm = LLMSource(model_id="us.amazon.nova-lite-v1:0")

    if not llm.bedrock_client:
        print("\n❌ Failed to initialize Bedrock client")
        sys.exit(1)

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
        print(f"Model: {out['model']}")
        print(f"Answer:\n{out['answer'][:400]}...")

    print(f"\n{'=' * 70}")
    print("✅ Done")
    print(f"{'=' * 70}")