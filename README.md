# Adaptive Knowledge Selector for Medical Queries

An intelligent reinforcement learning system that dynamically selects the best knowledge source (Knowledge Graph, Tools/APIs, LLM, or Documents) for medical and pharmaceutical queries. Built with Deep Q-Networks (DQN) and evaluated with LLM-as-Judge quality metrics.

---

## 🎯 Project Overview

This system intelligently routes medical queries to the most appropriate knowledge source using a trained RL agent. It integrates four distinct biomedical knowledge sources and learns optimal routing decisions through reinforcement learning with automatic reward signals.

### ✨ Key Features

- **🤖 Adaptive RL-based Routing**: DQN agent trained to select optimal knowledge source per query
- **📊 Multi-Source Integration**: 4 specialized knowledge sources
  - 🕸️ **Knowledge Graph** (Hetionet - 47K nodes, 2.25M edges)
  - 🔧 **Tool/API** (OpenFDA, RxNorm, medical calculators)  
  - 🤖 **LLM** (AWS Bedrock Nova 2 Lite)
  - 📚 **PDF Documents** (FAISS semantic search on KRR papers)

- **📈 Confidence Scoring**: All sources return 0-1 confidence with answers
- **🎓 LLM-as-Judge**: Automatic quality evaluation (correctness, relevance, completeness)
- **🔄 Online Learning**: Model continuously improves from real usage with auto-retraining
- **📊 ML Metrics**: Accuracy, precision, recall, F1 scores on 250-query test set
- **🎮 Interactive Dashboard**: Real-time query testing with evaluation feedback

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- AWS Account (for Bedrock access)
- ~2GB disk space (for knowledge graph & vector store)

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd Adaptive-Knowledge-Selector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment variables
cp .env.example .env
# Edit .env with your AWS credentials

# 4. Setup knowledge sources
python scripts/setup_hetionet.py  # Downloads and processes Hetionet (~5 min)
```

### Quick Test

```bash
# Interactive dashboard - query and see routing decisions
python scripts/interactive_dashboard.py

# Example queries:
# "What drugs interact with warfarin?"  → KnowledgeGraph
# "Calculate BMI for 70 kg and 1.75 m"  → ToolAPI
# "How does insulin work?"               → LLM
# "What do KRR papers say about ontologies?" → PDF
```

---

## 📁 Project Structure

```
.
├── models/                          # RL model implementation
│   ├── adaptive_selector.py         # DQN agent (Q-network)
│   ├── reward_evaluator.py          # Reward computation logic
│   └── rl_environment.py            # Experience replay buffer
│
├── knowledge_sources/               # 4 knowledge source implementations
│   ├── knowledge_graph.py           # Hetionet graph queries
│   ├── tool_api_source.py           # OpenFDA, RxNorm, calculators
│   ├── llm_source.py                # AWS Bedrock (Nova 2 Lite)
│   └── pdf_knowledge.py             # FAISS semantic search
│
├── utils/
│   ├── llm_judge.py                 # LLM-based quality evaluator
│   └── s3_sync.py                   # S3 model versioning
│
├── scripts/
│   ├── interactive_dashboard.py     # Main interactive interface ⭐
│   ├── evaluate_model_metrics.py    # ML metrics (accuracy, F1, etc.)
│   ├── train_rl_agent.py            # RL training pipeline
│   └── pretrain_supervised.py       # Phase 1: Supervised pre-training
│
├── data/
│   ├── hetionet/                    # Knowledge graph data
│   ├── pdf_store/                   # FAISS vector store
│   ├── rl_selector/
│   │   ├── adaptive_dqn.pth         # Trained model weights
│   │   ├── training_log.json        # Training history
│   │   └── evaluation_metrics.json  # Test set results
│   └── training_dataset_600.json    # Training queries
│
└── docs/                            # Comprehensive documentation
```

---

## 🧠 System Architecture

### Reinforcement Learning Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  User Query: "What drugs interact with warfarin?"          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Query Encoder (Sentence Transformers)                     │
│  → 384-dimensional embedding                                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  DQN Agent (Neural Network)                                │
│  → Predicts Q-values for each source                       │
│  → Selects source with highest Q-value                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Selected Source: KnowledgeGraphSource                     │
│  → Query execution + confidence score                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Reward Computation (automatic)                            │
│  → Base reward (correct/wrong source)                       │
│  → Confidence bonus                                         │
│  → Misrouting penalties                                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Experience Buffer → Batch Training (every 16 queries)     │
│  → Update Q-network weights                                 │
│  → Save model + upload to S3                                │
└─────────────────────────────────────────────────────────────┘
```

### Two-Phase Training

**Phase 1: Supervised Pre-training**
- 600 labeled queries with ground truth sources
- Cross-entropy loss on source classification
- Achieves 88.9% validation accuracy
- Creates baseline Q-value initialization

**Phase 2: RL Fine-tuning**
- Epsilon-greedy exploration (ε: 1.0 → 0.05)
- Experience replay buffer (2000 capacity)
- Automatic reward signals (no human feedback)
- 50 training episodes
- Average reward: 0.520

---

## 📊 Knowledge Sources

### 1. Knowledge Graph Source (Hetionet)
**Use Cases**: Drug interactions, side effects, treatments, gene relationships

- **Nodes**: 47,031 biomedical entities
  - Compounds (1,552 drugs)
  - Diseases (137)
  - Genes (20,945)
  - Side Effects, Pathways, Anatomy
  
- **Edges**: 2,250,197 relationships
  - treats, causes, binds, upregulates, downregulates, associates, etc.

- **Confidence**: Based on number of results found (0-1)

**Example Query**: "What drugs interact with aspirin?"
```python
# Returns: Drug Interactions for aspirin
# Found 7 interacting drugs: [Meclofenamic acid, ...]
# Confidence: 0.70
```

### 2. Tool/API Source
**Use Cases**: Medical calculations, FDA drug labels, drug information

- **OpenFDA**: Drug labels, adverse events, indications
- **RxNorm**: Drug terminology and interactions
- **Calculators**:
  - BMI (Body Mass Index)
  - CrCl (Creatinine Clearance - Cockcroft-Gault)
  - IBW (Ideal Body Weight)

- **Confidence**: 1.0 (success) or 0.0 (error)

**Example Query**: "Calculate BMI for 70 kg and 1.75 m"
```python
# Returns: BMI = 22.86 (Normal weight)
# Confidence: 1.00
```

### 3. LLM Source (AWS Bedrock Nova 2 Lite)
**Use Cases**: Explanations, mechanisms, conceptual questions

- **Model**: Amazon Nova 2 Lite (cost-effective, fast)
- **Features**: Medical knowledge, reasoning, explanations
- **Cost**: ~$0.0001 per query

- **Confidence**: Heuristic based on answer length and uncertainty detection
  - 0.8: Good answer (>100 chars, no uncertainty)
  - 0.3: Uncertain ("I don't know", "unclear")
  - 0.0: Error

**Example Query**: "How does insulin work?"
```python
# Returns: "Insulin is a hormone produced by beta cells..."
# Confidence: 0.80
```

### 4. PDF Knowledge Source
**Use Cases**: Document-specific queries, research papers, KRR topics

- **Corpus**: 3 KRR medical informatics papers
- **Vector Store**: FAISS with 1,649 chunks
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2, 384-dim)
- **Search**: Semantic similarity

- **Confidence**: FAISS similarity score (0-1)

**Example Query**: "What do KRR papers say about ontologies?"
```python
# Returns: Best match from papers (score: 0.47)
# Confidence: 0.47
```

---

## 🎯 Reward Function Design

The reward function guides the RL agent toward optimal routing:

```python
# Base rewards
if correct_source and has_results:
    reward = +1.0
elif correct_source and no_results:
    reward = -0.2
elif wrong_source and has_results:
    reward = 0.0    # Don't reward wrong choices
elif wrong_source and no_results:
    reward = -0.5

# Confidence bonus (0-0.2)
reward += 0.2 * confidence

# Misrouting penalties (-0.4 to -0.6)
if PDF_for_interaction:
    reward -= 0.5
if PDF_for_calculation:
    reward -= 0.5
if LLM_for_interaction:
    reward -= 0.2

# Perfect routing bonus (+0.15)
if correct_source and confidence > 0.8:
    reward += 0.15

# Final reward range: -1.0 to +1.5
```

---

## 📈 Evaluation & Metrics

### ML-Style Metrics (250-query test set)

```bash
python scripts/evaluate_model_metrics.py
```

**Manual Labeled Test Set (250 queries, balanced)**:
- 63 KnowledgeGraph queries
- 63 ToolAPI queries  
- 63 LLM queries
- 61 PDF queries

**Metrics Computed**:
- **Overall Accuracy**: % queries routed to correct source
- **Per-Source Precision**: Of queries sent to source, % correct
- **Per-Source Recall**: Of queries for source, % caught
- **Per-Source F1**: Harmonic mean of precision & recall
- **Confusion Matrix**: Source confusion patterns

**Example Output**:
```
OVERALL ACCURACY: 0.750 (75.0%)

PER-SOURCE METRICS:
Source                     Precision    Recall       F1        Support
---------------------------------------------------------------------------
KnowledgeGraph                0.900     0.600        0.720       15
ToolAPI                       0.000     0.000        0.000        0
LLM                           0.900     0.600        0.720       15
PDFKnowledge                  0.868     0.943        0.904       35
---------------------------------------------------------------------------
Macro Average                 0.442     0.386        0.406
```

### LLM-as-Judge Evaluation

Automatic quality assessment for each answer:

```python
{
  "correctness": 0.85,    # Is the answer factually accurate?
  "relevance": 0.90,      # Does it answer the question?
  "completeness": 0.75,   # Is sufficient detail provided?
  "quality": 0.83         # Overall score (average)
}
```

**Cost**: ~$0.0001 per evaluation (AWS Nova Lite)

---

## 🎮 Interactive Dashboard

The main interface for testing and continuous learning:

```bash
python scripts/interactive_dashboard.py
```

### Features

**For Each Query Shows**:
1. Query classification (type detection)
2. Router decision (Q-values for all sources)
3. Selected source + answer
4. Confidence score
5. Automatic reward computation
6. LLM judge evaluation (quality scores)
7. Validity check (router vs LLM alignment)

**Auto-Retraining**:
- Stores experiences to `data/rl_selector/online_experiences.jsonl`
- After 16 queries: triggers automatic retraining
- Saves updated model + uploads to S3
- Clears experience log (ready for next batch)

**Commands**:
- `stats` - Show learning statistics
- `examples` - Sample queries for each source
- `quit` - Save and exit

---

## 🔄 Training Pipeline

### Phase 1: Supervised Pre-training

```bash
python scripts/pretrain_supervised.py
```

- Trains on 600 labeled queries
- 80/20 train/validation split
- 50 epochs with early stopping
- Achieves ~89% validation accuracy
- Saves: `data/rl_selector/adaptive_dqn.pth`

### Phase 2: RL Fine-tuning

```bash
python scripts/train_rl_agent.py
```

- Loads Phase 1 weights
- Epsilon-greedy exploration (1.0 → 0.05)
- 50 training episodes
- Experience replay (batch size: 32)
- Average reward: 0.520
- Saves: Updated `adaptive_dqn.pth` + `training_log.json`

### Phase 3: Online Learning

Automatic continuous improvement during real usage via dashboard:
- Collects experiences during queries
- Batch retraining every 16 queries
- Incremental model updates
- S3 versioning for model checkpoints

---

## 📊 Performance Results

### Training Metrics

**Phase 1 (Supervised)**:
- Training Accuracy: 95.2%
- Validation Accuracy: 88.9%
- Loss: 0.285

**Phase 2 (RL Fine-tuning)**:
- Average Reward: 0.520
- Final Epsilon: 0.36
- Episodes: 50
- Time: 23.7 seconds

**Source Distribution (After Training)**:
```
PDFKnowledgeSource:      34.0%
LLMSource:               28.0%
KnowledgeGraphSource:    20.0%
ToolAPISource:           18.0%
```

### Q-Value Analysis

**Drug Interaction Query** → KnowledgeGraph preferred:
```
KnowledgeGraphSource  Q=+0.83 ← SELECTED
ToolAPISource         Q=+0.01
LLMSource             Q=-0.15
PDFKnowledgeSource    Q=-0.22
```

**Calculation Query** → ToolAPI strongly preferred:
```
ToolAPISource         Q=+1.55 ← SELECTED
KnowledgeGraphSource  Q=+0.29
LLMSource             Q=-0.14
PDFKnowledgeSource    Q=-0.59
```

**Concept Query** → LLM preferred:
```
LLMSource             Q=+0.53 ← SELECTED
PDFKnowledgeSource    Q=+0.23
KnowledgeGraphSource  Q=+0.20
ToolAPISource         Q=-0.18
```

### Evaluation Quality

**LLM Judge Results (8 test queries)**:
```
Average Quality:      0.542
Average Correctness:  0.562
Average Relevance:    0.562
Average Completeness: 0.500
```

**Improvements After Retraining**:
- PDF over-selection: 62.5% → 25% (60% reduction)
- Perfect quality answers: 25% → 37.5% (+50%)
- Overall quality: 0.521 → 0.542 (+4%)

---

## 🛠️ Technologies & Dependencies

### Core Libraries
- **PyTorch**: Neural network & RL training
- **Sentence Transformers**: Query embeddings
- **NetworkX**: Graph data structure
- **FAISS**: Vector similarity search
- **boto3**: AWS Bedrock API
- **scikit-learn**: ML utilities

### Key Dependencies
```
torch>=2.0.0
sentence-transformers>=2.2.0
networkx>=3.0
faiss-cpu>=1.7.4
boto3>=1.26.0
python-dotenv>=1.0.0
numpy>=1.24.0
requests>=2.31.0
```

Full list: See `requirements.txt`

---

## 🔧 Configuration

### Environment Variables (.env)

```bash
# AWS Bedrock (required for LLM source)
AWS_REGION=us-west-2
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# S3 Model Versioning (optional)
S3_MODEL_BUCKET=your-bucket-name
AWS_PROFILE=default

# API Keys (optional)
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

### Model Configuration

Edit `models/adaptive_selector.py`:
```python
AdaptiveSelector(
    input_dim=384,      # Embedding dimension
    num_sources=4,      # Number of sources
    lr=0.001            # Learning rate
)
```

---

## 📖 Documentation

### Main Documentation Files

- **CONFIDENCE_AND_EVALUATION.md** - How confidence scoring works
- **REWARD_IMPROVEMENTS.md** - Reward function design details
- **AWS_BEDROCK_SETUP.md** - AWS configuration guide

### Script Documentation

All scripts include detailed docstrings:
```bash
python scripts/interactive_dashboard.py --help
python scripts/evaluate_model_metrics.py --help
python scripts/train_rl_agent.py --help
```

---

## 🚀 Advanced Usage

### Custom Training Dataset

```python
# Add new training examples
training_data = [
    {
        "query": "Your query here",
        "best_source": "KnowledgeGraphSource",
        "query_type": "interaction",
        "reasoning": "Why this source is best"
    }
]

# Save and retrain
with open('data/training_dataset_600.json', 'w') as f:
    json.dump(training_data, f)

# Run training
python scripts/pretrain_supervised.py
python scripts/train_rl_agent.py
```

### Programmatic API

```python
from sentence_transformers import SentenceTransformer
from models.adaptive_selector import AdaptiveSelector
from scripts.train_rl_agent import TrainingSystem

# Initialize
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
agent = AdaptiveSelector(input_dim=384, num_sources=4)
agent.load("data/rl_selector/adaptive_dqn.pth")
system = TrainingSystem()

# Query
query = "What drugs interact with warfarin?"
emb = encoder.encode([query])[0]
action_idx = agent.select_action(emb, epsilon=0.0)
source_name = agent.sources[action_idx]

# Get answer
results = system.query_source(source_name, query)
print(f"Source: {source_name}")
print(f"Answer: {results['answer']}")
print(f"Confidence: {results['confidence']}")
```

### Batch Evaluation

```python
# Evaluate on custom test set
from scripts.evaluate_model_metrics import ModelEvaluator

evaluator = ModelEvaluator()
test_set = [
    {"query": "Query 1", "correct_source": "KnowledgeGraphSource"},
    {"query": "Query 2", "correct_source": "ToolAPISource"}
]

metrics = evaluator.evaluate_test_set(test_set, "Custom Test")
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

---

## 🐛 Troubleshooting

### Common Issues

**1. AWS Bedrock Access Error**
```
Solution: Ensure AWS credentials are set and region supports Bedrock
- Check .env file
- Verify AWS_REGION=us-west-2 (or supported region)
- Enable Bedrock model access in AWS console
```

**2. FAISS Index Not Found**
```
Solution: PDF vector store needs initialization
python scripts/setup_pdf_knowledge.py
```

**3. Hetionet Graph Missing**
```
Solution: Download and setup knowledge graph
python scripts/setup_hetionet.py
```

**4. Model File Not Found**
```
Solution: Train the model first
python scripts/pretrain_supervised.py
python scripts/train_rl_agent.py
```

---

## 🎓 Research & References

### Key Concepts

- **Deep Q-Network (DQN)**: Value-based RL algorithm for discrete action spaces
- **Experience Replay**: Stores transitions for stable off-policy learning
- **Epsilon-Greedy**: Balances exploration vs exploitation
- **LLM-as-Judge**: Using LLMs to evaluate other AI system outputs

### Datasets & Resources

- **Hetionet v1.0**: Open biomedical knowledge graph
  - Paper: https://doi.org/10.7554/eLife.26726
  - Data: https://github.com/hetio/hetionet

- **OpenFDA**: Public FDA data API
  - https://open.fda.gov/

- **RxNorm**: NLM drug terminology
  - https://www.nlm.nih.gov/research/umls/rxnorm/

---

## 🤝 Contributing

This is a course project for **KRR (Knowledge Representation and Reasoning)**, Spring 2026, Arizona State University.

### Team
- **Student**: Harsh Tita
- **Course**: CSE 579 - Knowledge Representation and Reasoning
- **Institution**: Arizona State University

---

## 📝 Project Milestones

### ✅ Completed

- [x] Multi-source knowledge integration (4 sources)
- [x] Hetionet knowledge graph setup
- [x] FAISS semantic search for PDFs
- [x] Query classification system
- [x] Confidence scoring for all sources
- [x] Deep Q-Network (DQN) implementation
- [x] Two-phase training pipeline (supervised + RL)
- [x] Experience replay buffer
- [x] Automatic reward computation
- [x] LLM-as-Judge quality evaluation
- [x] Interactive query dashboard
- [x] Online learning with auto-retraining
- [x] ML metrics evaluation (accuracy, F1, etc.)
- [x] S3 model versioning
- [x] Reward function optimization
- [x] Model improvement validation

### 🎯 Future Enhancements

- [ ] Multi-source ensemble (query multiple, synthesize best answer)
- [ ] Cost-aware routing (consider latency & API costs)
- [ ] User feedback integration (thumbs up/down)
- [ ] A/B testing framework
- [ ] REST API deployment
- [ ] Web UI interface
- [ ] Explainability dashboard (why source was chosen)
- [ ] Additional knowledge sources (UniProt, DrugBank)

---

## 📄 License

Educational project for ASU coursework. All rights reserved.

### Data Sources

- **Hetionet**: CC0 1.0 Universal (Public Domain)
- **OpenFDA**: Public domain (US Government)
- **PubMed**: Public access

---

## 🙏 Acknowledgments

- **Prof. [Name]** - KRR Course Instructor, ASU
- **Hetionet Team** - Open biomedical knowledge graph
- **NCBI/NLM** - PubMed & RxNorm APIs
- **OpenFDA** - Public drug information
- **AWS** - Bedrock LLM infrastructure
- **Anthropic Claude** - Development assistance

---

## 📞 Contact

For questions about this project:
- **GitHub Issues**: [Repository Issues Page]
- **Email**: [Student Email]
- **Course**: CSE 579 - KRR, Spring 2026

---

**Made with ❤️ for KRR Course Project - Spring 2026, Arizona State University**

**⭐ Star this repo if you found it useful!**