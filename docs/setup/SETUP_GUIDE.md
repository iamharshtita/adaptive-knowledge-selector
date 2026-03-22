# Knowledge Source Integration - Setup Guide

This guide will help you set up and integrate all four knowledge sources for the Adaptive Knowledge Source Selection project.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Knowledge Source Setup](#knowledge-source-setup)
4. [API Keys & Credentials](#api-keys--credentials)
5. [Testing Each Source](#testing-each-source)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

---

## Prerequisites

### System Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for local LLMs)
- GPU recommended (but not required) for local LLM models

### Required Accounts & APIs
1. **NCBI Account** (Free) - for PubMed access
   - Register at: https://www.ncbi.nlm.nih.gov/account/
   - Get API key: https://www.ncbi.nlm.nih.gov/account/settings/

2. **OpenAI Account** (Optional, Paid) - for GPT-4
   - Sign up at: https://platform.openai.com/
   - Get API key: https://platform.openai.com/api-keys

3. **Anthropic Account** (Optional, Paid) - for Claude
   - Sign up at: https://www.anthropic.com/
   - Get API key from console

4. **HuggingFace Account** (Free) - for Llama 2 and other models
   - Sign up at: https://huggingface.co/
   - Get token: https://huggingface.co/settings/tokens
   - Request Llama 2 access: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

---

## Installation

### Step 1: Clone/Setup Project

```bash
cd "/Users/harshtita/Desktop/ASU Docs/KRR - Spring26/Course Project"
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Setup Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

Edit `.env` file:
```env
# Required for PubMed
NCBI_API_KEY=your_actual_ncbi_key_here

# Optional - for GPT-4
OPENAI_API_KEY=your_openai_key_here

# Optional - for Claude
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional - for Llama 2 and HuggingFace models
HUGGINGFACE_TOKEN=your_hf_token_here
```

---

## Knowledge Source Setup

### 1️⃣ Text Knowledge Source (PubMed + DailyMed)

**What it does**: Retrieves medical literature and drug labels

**APIs Used**:
- PubMed (via Biopython/Entrez)
- DailyMed (NIH)

**Setup**:
```bash
# Test PubMed access
python knowledge_sources/text_knowledge.py
```

**No additional setup needed** - APIs are public!

**NCBI API Key Benefits**:
- Without key: 3 requests/second
- With key: 10 requests/second

---

### 2️⃣ Knowledge Graph Source (DrugBank + Hetionet)

**What it does**: Stores structured drug-drug interactions and relationships

**Data Sources**:

#### Option A: Sample Data (Quick Start)
The code includes sample data for testing. No additional setup needed!

```bash
# Test with sample data
python knowledge_sources/knowledge_graph.py
```

#### Option B: Real DrugBank Data (Recommended for production)
1. Register at https://go.drugbank.com/
2. Download DrugBank XML file (requires free academic license)
3. Parse XML and load into graph (parser code not included, but structure is ready)

#### Option C: Hetionet Data (Free, Comprehensive)
1. Download Hetionet:
   ```bash
   mkdir -p data/hetionet
   cd data/hetionet

   # Download and extract
   wget https://github.com/hetio/hetionet/raw/master/hetnet/json/hetionet-v1.0.json.bz2
   bunzip2 hetionet-v1.0.json.bz2
   ```

2. Load in code:
   ```python
   kg = KnowledgeGraphSource()
   kg.load_hetionet("data/hetionet/hetionet-v1.0.json")
   ```

---

### 3️⃣ Tool/API Source (OpenFDA + RxNorm)

**What it does**: Provides drug information, interactions, and medical calculators

**APIs Used**:
- OpenFDA (FDA drug data)
- RxNorm (NIH drug names and relationships)

**Setup**:
```bash
# Test API access (no authentication needed!)
python knowledge_sources/tool_api_source.py
```

**No API keys required** - both are free public APIs!

**Features**:
- Drug label lookup
- Drug interaction checking
- Medical calculators (BMI, creatinine clearance, etc.)

---

### 4️⃣ LLM Source (GPT-4 / BioGPT / Llama 2)

**What it does**: Provides conceptual explanations and reasoning

**Model Options**:

#### Option A: GPT-4 (Recommended - Best Quality)
```bash
# Set API key in .env
OPENAI_API_KEY=your_key_here

# Test
python knowledge_sources/llm_source.py
```

**Pros**: Best medical reasoning, easy to use
**Cons**: Costs ~$0.01-0.03 per query

#### Option B: BioGPT (Medical-Specific, Free)
```bash
# No API key needed - downloads model automatically
# Requires ~5GB disk space

# Test (first run downloads model)
python -c "from knowledge_sources.llm_source import LLMSource; llm = LLMSource('biogpt')"
```

**Pros**: Medical domain-specific, runs locally
**Cons**: Slower, requires good GPU for speed

#### Option C: Llama 2 (General Purpose, Free)
```bash
# Requires HuggingFace token and model access approval
# 1. Get token: https://huggingface.co/settings/tokens
# 2. Request access: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

HUGGINGFACE_TOKEN=your_token_here

# Test
python -c "from knowledge_sources.llm_source import LLMSource; llm = LLMSource('llama2')"
```

**Pros**: Free, runs locally, good general knowledge
**Cons**: NOT specialized for medical, may hallucinate, requires GPU

### ⚠️ **Llama 2 for Medical Text Generation - Important Notes**

**Is Llama 2 good for medical text generation?**
- **Short answer**: Not ideal out-of-the-box
- **Better than**: Nothing, but worse than specialized models
- **Use if**: You need free/offline option and can accept lower quality

**Recommendations**:
1. **Best**: Use GPT-4 or Claude via API ($$$)
2. **Good**: Use BioGPT (free, medical-specific)
3. **OK**: Use Llama 2 **after fine-tuning** on medical QA data
4. **Not Recommended**: Use base Llama 2 for critical medical queries

**If you must use Llama 2**:
- Fine-tune on datasets like MedQA, PubMedQA
- Use larger models (13B or 70B, not 7B)
- Always validate outputs with other sources
- Add strong disclaimers about accuracy

---

## Testing Each Source

### Complete Test Script

```bash
# Test all sources
python examples/integrate_all_sources.py
```

### Individual Tests

```bash
# Test Text Knowledge
python knowledge_sources/text_knowledge.py

# Test Knowledge Graph
python knowledge_sources/knowledge_graph.py

# Test Tool/API
python knowledge_sources/tool_api_source.py

# Test LLM
python knowledge_sources/llm_source.py
```

---

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError"
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

#### 2. "NCBI API Error: Too many requests"
- Get NCBI API key and add to .env
- Add delays between requests

#### 3. "CUDA out of memory" (for local LLMs)
- Use smaller model (7B instead of 13B)
- Reduce batch size
- Use CPU instead of GPU (slower but works)

#### 4. "OpenAI API Error: Invalid API key"
- Check API key in .env
- Ensure you have credits in OpenAI account

#### 5. "Llama 2 model not found"
- Request access at HuggingFace
- Approval usually takes 1-2 hours
- Check HuggingFace token is correct

---

## Next Steps

### 1. Build the Complete System

Now that you have all knowledge sources working, build the selector model:

1. **Query Encoder**: Encode queries to vectors (already in text_knowledge.py)
2. **Selector Model**: Neural network to predict best source
3. **Reward Evaluator**: Evaluate answer quality
4. **Training Loop**: Reinforcement learning

### 2. Data Collection

Collect training data:
```python
# Format: (query, best_source, reward)
training_data = [
    ("What drugs interact with aspirin?", "knowledge_graph", 1.0),
    ("Explain how insulin works", "llm", 0.95),
    ("Dosage of ibuprofen", "tool_api", 1.0),
    # ... more examples
]
```

### 3. Evaluation

Create test queries to measure:
- Selection accuracy
- Answer quality
- Improvement over static routing

---

## Resource Summary

### Free Resources
✅ PubMed/DailyMed - Free, no limits with API key
✅ OpenFDA - Free, unlimited
✅ RxNorm - Free, unlimited
✅ Hetionet - Free download
✅ BioGPT - Free, local
✅ Llama 2 - Free, local (with HF account)

### Paid Resources
💰 GPT-4 - ~$0.01-0.03 per query
💰 Claude - ~$0.01-0.02 per query
💰 DrugBank Full - $2,500/year (free academic version available)

### Recommended Minimal Setup (Free)
1. Text: PubMed (with API key)
2. Graph: Sample data or Hetionet
3. Tools: OpenFDA + RxNorm
4. LLM: BioGPT (for medical) or Llama 2 (for general)

### Recommended Production Setup
1. Text: PubMed + DailyMed
2. Graph: DrugBank + Hetionet
3. Tools: OpenFDA + RxNorm + Custom calculators
4. LLM: GPT-4 API (best quality)

---

## Questions?

- Check code comments in each source file
- Review examples/integrate_all_sources.py
- Test individual components before integration

**Ready to build the selector model!** 🚀
