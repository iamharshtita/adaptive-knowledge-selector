# Quick Resource Access Reference

This guide shows you exactly how to access each knowledge resource with code examples.

---

## 1. Text Knowledge Sources

### A. PubMed (via NCBI Entrez)

**Access**: Free API
**Registration**: https://www.ncbi.nlm.nih.gov/account/
**API Key**: https://www.ncbi.nlm.nih.gov/account/settings/
**Rate Limit**: 3 req/sec (10 with API key)

```python
from Bio import Entrez

# Setup
Entrez.email = "your.email@example.com"
Entrez.api_key = "your_api_key"  # Optional but recommended

# Search
handle = Entrez.esearch(db="pubmed", term="aspirin interactions", retmax=10)
record = Entrez.read(handle)
id_list = record["IdList"]

# Fetch details
handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
articles = Entrez.read(handle)
```

**Documentation**: https://www.ncbi.nlm.nih.gov/books/NBK25499/

---

### B. DailyMed

**Access**: Free REST API
**No registration required**
**Documentation**: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm

```python
import requests

# Search for drug
url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
response = requests.get(url, params={'drug_name': 'aspirin'})
data = response.json()

# Get drug label
setid = data['data'][0]['setid']
detail_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.json"
detail = requests.get(detail_url).json()
```

**Endpoints**:
- Search: `https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json`
- Details: `https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.json`

---

## 2. Knowledge Graph Sources

### A. DrugBank

**Access**: Requires registration
**License**: Free for academic, $2,500/year commercial
**Registration**: https://go.drugbank.com/

**Data Format**: XML file download

```python
# After downloading XML file
import xml.etree.ElementTree as ET

tree = ET.parse('drugbank.xml')
root = tree.getroot()

# Parse drug-drug interactions
for drug in root.findall('.//{http://www.drugbank.ca}drug'):
    name = drug.find('.//{http://www.drugbank.ca}name').text
    interactions = drug.findall('.//{http://www.drugbank.ca}drug-interaction')
    # Process interactions...
```

**Download**: https://go.drugbank.com/releases/latest

---

### B. Hetionet

**Access**: Free, open source
**Format**: JSON
**GitHub**: https://github.com/hetio/hetionet

```python
import json
import networkx as nx

# Download
# wget https://github.com/hetio/hetionet/raw/master/hetnet/json/hetionet-v1.0.json.bz2

# Load
with open('hetionet-v1.0.json', 'r') as f:
    data = json.load(f)

# Build graph
G = nx.MultiDiGraph()
for node in data['nodes']:
    G.add_node(node['id'], **node)
for edge in data['edges']:
    G.add_edge(edge['source'], edge['target'], **edge)
```

**Download Command**:
```bash
wget https://github.com/hetio/hetionet/raw/master/hetnet/json/hetionet-v1.0.json.bz2
bunzip2 hetionet-v1.0.json.bz2
```

---

### C. TWOSIDES (Drug-Drug Interactions)

**Access**: Free download
**Website**: http://tatonettilab.org/offsides/
**Format**: CSV files

```python
import pandas as pd

# Download from website, then:
df = pd.read_csv('TWOSIDES.csv')

# Columns: drug1, drug2, side_effect, frequency
interactions = df[df['drug1'] == 'ASPIRIN']
```

---

## 3. Tool/API Sources

### A. OpenFDA

**Access**: Free, no registration
**Rate Limit**: 1000 requests/min
**Documentation**: https://open.fda.gov/apis/

```python
import requests

# Drug labels
url = "https://api.fda.gov/drug/label.json"
params = {'search': 'openfda.brand_name:"aspirin"', 'limit': 1}
response = requests.get(url, params=params)
data = response.json()

# Adverse events
url = "https://api.fda.gov/drug/event.json"
params = {'search': 'patient.drug.medicinalproduct:"aspirin"', 'limit': 10}
response = requests.get(url, params=params)
events = response.json()
```

**Available Endpoints**:
- `/drug/label.json` - Drug labels
- `/drug/event.json` - Adverse events
- `/drug/enforcement.json` - Enforcement reports

---

### B. RxNorm (NIH RxNav)

**Access**: Free, no registration
**Documentation**: https://lhncbc.nlm.nih.gov/RxNav/APIs/

```python
import requests

# Get drug ID
url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
response = requests.get(url, params={'name': 'aspirin'})
rxcui = response.json()['idGroup']['rxnormId'][0]

# Get interactions
url = f"https://rxnav.nlm.nih.gov/REST/interaction/interaction.json"
response = requests.get(url, params={'rxcui': rxcui})
interactions = response.json()
```

**Useful Endpoints**:
- `/rxcui.json` - Get RxCUI for drug name
- `/interaction/interaction.json` - Get drug interactions
- `/rxcui/{rxcui}/properties.json` - Get drug properties

---

### C. PubChem (Chemical Information)

**Access**: Free, no registration
**Documentation**: https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest

```python
import requests

# Search compound
url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/JSON"
response = requests.get(url)
data = response.json()

# Get properties
cid = data['PC_Compounds'][0]['id']['id']['cid']
url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,MolecularWeight/JSON"
properties = requests.get(url).json()
```

---

## 4. LLM Sources

### A. GPT-4 (OpenAI)

**Access**: Paid API ($0.01-0.03/1K tokens)
**Registration**: https://platform.openai.com/
**API Key**: https://platform.openai.com/api-keys

```python
import openai

openai.api_key = "your_api_key"

response = openai.ChatCompletion.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "system", "content": "You are a medical expert."},
        {"role": "user", "content": "How does insulin work?"}
    ],
    max_tokens=500,
    temperature=0.7
)

answer = response.choices[0].message.content
```

**Models Available**:
- `gpt-4-turbo-preview` - Latest, fastest
- `gpt-4` - Original, most capable
- `gpt-3.5-turbo` - Cheaper alternative

---

### B. Claude (Anthropic)

**Access**: Paid API (~$0.01-0.02/1K tokens)
**Registration**: https://www.anthropic.com/
**API Key**: From console

```python
import anthropic

client = anthropic.Anthropic(api_key="your_api_key")

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "How does insulin work?"}
    ]
)

answer = message.content[0].text
```

**Models Available**:
- `claude-3-opus-20240229` - Most capable
- `claude-3-sonnet-20240229` - Balanced
- `claude-3-haiku-20240307` - Fastest, cheapest

---

### C. BioGPT (Microsoft)

**Access**: Free, runs locally
**Source**: HuggingFace
**Model**: https://huggingface.co/microsoft/biogpt

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/biogpt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer("How does insulin work?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Requirements**:
- ~5GB disk space
- GPU recommended (but works on CPU)

---

### D. Llama 2 (Meta)

**Access**: Free, runs locally
**Source**: HuggingFace (requires approval)
**Model**: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, token="your_hf_token")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token="your_hf_token",
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "<s>[INST] How does insulin work? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Requirements**:
- HuggingFace account with model access approval
- ~13GB disk space (7B model)
- GPU strongly recommended (16GB+ VRAM)

**Steps to get access**:
1. Create HuggingFace account
2. Request access at model page
3. Wait for approval (usually 1-2 hours)
4. Generate access token

---

### E. Mistral 7B

**Access**: Free, runs locally
**Source**: HuggingFace
**Model**: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

prompt = "[INST] How does insulin work? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 5. Embeddings for Query Encoding

### Sentence Transformers

**Access**: Free
**Documentation**: https://www.sbert.net/

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Encode queries
query = "What drugs interact with aspirin?"
embedding = model.encode(query)  # Returns 384-dim vector

# Batch encoding
queries = ["query1", "query2", "query3"]
embeddings = model.encode(queries)
```

**Recommended Models**:
- `all-MiniLM-L6-v2` - Fast, 384 dims (used in project spec)
- `all-mpnet-base-v2` - Better quality, 768 dims
- `biobert-nli` - Medical domain-specific

---

## 6. Vector Databases

### FAISS (Facebook AI)

**Access**: Free, local
**GitHub**: https://github.com/facebookresearch/faiss

```python
import faiss
import numpy as np

# Create index
dimension = 384  # For MiniLM
index = faiss.IndexFlatL2(dimension)

# Add vectors
embeddings = np.array([...])  # Your embeddings
index.add(embeddings.astype('float32'))

# Search
query_embedding = np.array([...])
distances, indices = index.search(query_embedding.astype('float32'), k=5)
```

---

## Quick Start Checklist

### Completely Free Setup
- [x] PubMed (with NCBI API key)
- [x] OpenFDA
- [x] RxNorm
- [x] Hetionet (download)
- [x] BioGPT or Mistral (local)

### Production Setup (some costs)
- [x] PubMed (with NCBI API key)
- [x] OpenFDA
- [x] RxNorm
- [x] DrugBank (academic license)
- [x] GPT-4 API (~$50/month)

---

## Rate Limits Summary

| Resource | Limit | Authentication |
|----------|-------|----------------|
| PubMed | 3/sec (10 with key) | API key |
| DailyMed | Unlimited | None |
| OpenFDA | 1000/min | None |
| RxNorm | Unlimited | None |
| GPT-4 | Based on tier | API key |
| Claude | Based on tier | API key |

---

## Cost Comparison (Monthly)

| Setup | Cost | Quality |
|-------|------|---------|
| All Free | $0 | Good |
| Free + GPT-4 | ~$30-100 | Excellent |
| Free + Claude | ~$30-100 | Excellent |
| Production + GPT-4 | ~$2,500+ | Best |

---

This guide provides direct access to all resources. Copy the code snippets and modify for your use case!
