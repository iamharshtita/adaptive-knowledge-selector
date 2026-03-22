# Adaptive Knowledge Source Selection System

An intelligent system that dynamically selects the best knowledge source (Text, Knowledge Graph, Tool/API, or LLM) for answering medical and pharmaceutical queries using reinforcement learning.

## 🎯 Project Overview

This system integrates multiple biomedical knowledge sources and uses an adaptive selector model to choose the most appropriate source for each query. Built for the KRR (Knowledge Representation and Reasoning) course project.

### Key Features

- **Multi-Source Integration**: Four different knowledge sources working in harmony
  - 📚 Text Knowledge (PubMed, DailyMed with FAISS semantic search)
  - 🕸️ Knowledge Graph (Hetionet - 47K+ nodes, 2.25M+ edges)
  - 🔧 Tool/API (OpenFDA, RxNorm, medical calculators)
  - 🤖 LLM (GPT-4, Claude, BioGPT support)

- **Interactive Graph Visualizations**: Explore drug relationships visually
  - D3.js force-directed interactive graphs
  - Filter by relationship type (treats, causes, binds, etc.)
  - Drag, click, and explore connections

- **Semantic Search**: FAISS vector database for intelligent text retrieval

- **Adaptive Selection** (In Progress): RL-based model to choose optimal source

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Course Project"

# Install dependencies
pip install -r requirements.txt

# Setup Hetionet knowledge graph
python3 scripts/setup_hetionet.py
```

### Basic Usage

**Visualize a drug's relationships:**
```bash
# Interactive HTML visualization
python3 scripts/create_viz.py metformin
open visualizations/metformin.html

# Show only diseases it treats
python3 scripts/viz_by_relationship.py atorvastatin treats
```

**Search for drugs:**
```bash
# Search by name
python3 scripts/list_drugs.py --search statin

# List all available drugs
python3 scripts/list_drugs.py --all
```

**Explore the knowledge graph:**
```bash
# Interactive command-line explorer
python3 scripts/view_knowledge_graph.py interactive
```

**Query all knowledge sources:**
```bash
# Run integrated demo
python3 examples/integrate_all_sources.py
```

## 📁 Project Structure

```
.
├── knowledge_sources/          # Core knowledge source implementations
│   ├── text_knowledge.py       # PubMed + DailyMed with FAISS
│   ├── knowledge_graph.py      # Hetionet graph integration
│   ├── tool_api_source.py      # OpenFDA, RxNorm, calculators
│   └── llm_source.py           # Multi-LLM support
│
├── scripts/                    # Utility scripts
│   ├── setup_hetionet.py       # Download and setup knowledge graph
│   ├── create_viz.py           # Interactive visualizations
│   ├── viz_by_relationship.py  # Filtered visualizations
│   ├── list_drugs.py           # Drug name finder
│   └── view_knowledge_graph.py # Graph explorer
│
├── examples/                   # Example integrations
│   └── integrate_all_sources.py
│
├── data/                       # Data storage
│   └── hetionet/               # Hetionet graph data
│
├── visualizations/             # Generated HTML visualizations
│
└── docs/                       # Documentation
    ├── setup/                  # Setup guides
    ├── visualization/          # Visualization guides
    └── reference/              # Technical references
```

## 📊 Knowledge Sources

### 1. Text Knowledge Source
- **PubMed**: Medical literature search via Entrez API
- **DailyMed**: Drug label information
- **FAISS**: Vector database for semantic search (384-dim embeddings)
- **Model**: Sentence Transformers (all-MiniLM-L6-v2)

### 2. Knowledge Graph Source
- **Dataset**: Hetionet v1.0
- **Nodes**: 47,031 biomedical entities
  - Compounds (1,552 drugs)
  - Diseases (137)
  - Genes (20,945)
  - Side Effects, Pathways, Anatomy, etc.
- **Edges**: 2,250,197 relationships
  - treats, causes, binds, upregulates, downregulates, etc.
- **Format**: NetworkX MultiDiGraph

### 3. Tool/API Source
- **OpenFDA**: Drug labels, adverse events, recalls
- **RxNorm**: Drug interactions and terminology
- **Calculators**: BMI, Creatinine Clearance

### 4. LLM Source
- **Supported Models**:
  - GPT-4 (recommended)
  - Claude
  - BioGPT (medical-specific)
  - Llama 2
  - Mistral
- **Features**: Medical query answering, explanations

## 🎨 Visualization Features

### Interactive HTML Visualizations

**Full Graph View:**
```bash
python3 scripts/create_viz.py metformin
```
- Shows all relationships (30 nearest neighbors)
- Drag nodes to rearrange
- Click for detailed information
- Hover for highlights

**Filtered by Relationship:**
```bash
# Show only diseases treated
python3 scripts/viz_by_relationship.py atorvastatin treats

# Show side effects
python3 scripts/viz_by_relationship.py metformin causes

# Multiple relationships
python3 scripts/viz_by_relationship.py ibuprofen treats,causes,binds
```

**Available Relationships:**
- `treats` - Diseases the drug treats
- `causes` - Side effects
- `binds` - Gene/protein binding
- `upregulates` - Gene upregulation
- `downregulates` - Gene downregulation

### Color Coding

**Node Types:**
- 🔴 Red = Compounds (Drugs)
- 🟦 Teal = Diseases
- 🟩 Green = Genes
- 🟨 Yellow = Side Effects
- Others = Pathways, Anatomy, etc.

**Relationships (Filtered Views):**
- 🟢 Green = treats
- 🔴 Red = causes
- 🔵 Blue = binds
- 🟠 Orange = downregulates
- 🟣 Purple = upregulates

## 🛠️ Technologies Used

- **Python 3.13**
- **NetworkX**: Graph data structure
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings
- **Biopython**: PubMed access
- **D3.js**: Interactive visualizations
- **OpenAI/Anthropic APIs**: LLM integration

## 📖 Documentation

Comprehensive documentation available in the [docs/](docs/) folder:

- **Setup Guides**
  - [Complete Setup Guide](docs/setup/SETUP_GUIDE.md)
  - [Hetionet Quick Start](docs/setup/HETIONET_QUICK_START.md)

- **Visualization Guides**
  - [Full Visualization Guide](docs/visualization/VISUALIZATION_GUIDE.md)
  - [Quick Viz Reference](docs/visualization/QUICK_VIZ_GUIDE.md)

- **Technical References**
  - [Graph Structure Explanation](docs/reference/GRAPH_EXPLANATION.md)
  - [Resource Access Guide](docs/reference/RESOURCE_ACCESS_GUIDE.md)

## 🔧 Example Usage

### Query All Sources

```python
from knowledge_sources.text_knowledge import TextKnowledgeSource
from knowledge_sources.knowledge_graph import KnowledgeGraphSource
from knowledge_sources.tool_api_source import ToolAPISource
from knowledge_sources.llm_source import LLMSource

# Initialize sources
text_source = TextKnowledgeSource()
kg_source = KnowledgeGraphSource()
api_source = ToolAPISource()
llm_source = LLMSource()

# Query each source
query = "What does metformin treat?"

# Text source - PubMed articles
articles = text_source.search_pubmed("metformin diabetes treatment")

# Knowledge graph - direct relationships
kg_source.load_graph("data/hetionet/hetionet_graph.pkl")
neighbors = kg_source.get_neighbors_with_relations(drug_id)

# API source - drug labels
label = api_source.search_drug_label("metformin")

# LLM source - natural language answer
answer = llm_source.query("What does metformin treat?", model="gpt-4")
```

### Search and Visualize

```python
from knowledge_sources.knowledge_graph import KnowledgeGraphSource

kg = KnowledgeGraphSource()
kg.load_graph("data/hetionet/hetionet_graph.pkl")

# Find drug
matches = kg.search_drug_by_name("aspirin")  # Returns "Acetylsalicylic acid"

# Get statistics
stats = kg.get_statistics()
print(f"Nodes: {stats['nodes']}, Edges: {stats['edges']}")

# Explore relationships
relations = kg.get_relationships_summary(drug_id)
# Returns: {'treats': 3, 'causes': 139, 'binds': 56, ...}
```

## 🚧 Current Status

### ✅ Completed
- [x] All 4 knowledge sources integrated
- [x] Hetionet knowledge graph setup
- [x] FAISS semantic search implementation
- [x] Interactive D3.js visualizations
- [x] Filtered relationship visualizations
- [x] Drug search and exploration tools
- [x] Multi-LLM support
- [x] Complete documentation

### 🔄 In Progress
- [ ] Adaptive selector model (neural network)
- [ ] Reinforcement learning implementation
- [ ] Reward mechanism for source selection
- [ ] Query routing logic
- [ ] Performance benchmarking

### 📋 Planned
- [ ] User feedback collection
- [ ] Model training pipeline
- [ ] Evaluation metrics
- [ ] API endpoint deployment
- [ ] Web interface

## 📝 Available Drugs

The system includes **1,552 compounds** from Hetionet. Common examples:

**Cardiovascular**: atorvastatin, simvastatin, amlodipine, metoprolol, warfarin
**Diabetes**: metformin, glipizide, pioglitazone, insulin
**Pain/Anti-inflammatory**: ibuprofen, acetaminophen, acetylsalicylic acid (aspirin)
**Antibiotics**: amoxicillin, ciprofloxacin, azithromycin

Find more: `python3 scripts/list_drugs.py --search <term>`

## 🤝 Contributing

This is a course project for KRR (Knowledge Representation and Reasoning), Spring 2026, ASU.

## 📄 License

Educational project - ASU Course Work

## 🙏 Acknowledgments

- **Hetionet**: Open-source biomedical knowledge graph
- **PubMed/NCBI**: Medical literature database
- **OpenFDA**: Public drug information API
- **RxNorm**: Drug terminology service

---

**Made with** ❤️ **for KRR Course Project - Spring 2026**
