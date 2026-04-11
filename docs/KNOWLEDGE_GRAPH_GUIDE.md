# 📚 Knowledge Graph - Complete Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setup Flow](#setup-flow)
4. [Knowledge Graph Structure](#knowledge-graph-structure)
5. [Core Concepts](#core-concepts)
6. [Usage Guide](#usage-guide)
7. [Common Tasks](#common-tasks)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

Your Knowledge Graph is a **filtered biomedical knowledge base** focused on drug-disease relationships.

### What You Have
- **Source**: Hetionet v1.0 (biomedical knowledge graph)
- **Size**: 7,339 nodes, 143,056 edges
- **Focus**: 3 medically relevant relationships only
- **Format**: NetworkX MultiDiGraph
- **Storage**: 12 MB (93% smaller than original)

### What It's For
- Drug side effect queries
- Disease treatment queries
- Symptom-disease associations
- Medical Q&A systems
- Adaptive knowledge selector

---

## ✅ Prerequisites

### 1. System Requirements

```bash
# Check Python version (need 3.8+)
python3 --version

# Check available disk space (need ~15 MB)
df -h .
```

### 2. Required Python Packages

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt
```

**Key packages:**
- `networkx>=3.1` - Graph data structure
- `pickle` - Fast graph serialization (built-in)

### 3. Project Structure

```
your-project/
├── data/
│   └── hetionet/
│       └── hetionet_filtered.pkl    ← Your filtered graph (12 MB)
├── knowledge_sources/
│   └── knowledge_graph.py           ← Core KG class
├── scripts/
│   ├── check_kg.py                  ← Inspector tool
│   ├── create_viz.py                ← Visualization
│   └── ...
└── visualizations/                  ← HTML visualizations
```

---

## 🔄 Setup Flow

### Step 1: Initial Setup (One-Time)

```bash
# Navigate to project directory
cd "/Users/harshtita/Desktop/Arizona State University/KRR - Spring26/Adaptive-Knowledge-Selector"

# Verify filtered graph exists
ls -lh data/hetionet/hetionet_filtered.pkl
```

**Expected output:**
```
-rw-r--r--  1 user  staff   12M Apr 10 20:24 hetionet_filtered.pkl
```

✅ **If file exists, you're ready to go!**

❌ **If file missing:**
```bash
# Option A: Re-download and filter (slow, ~2-3 min)
python3 scripts/setup_hetionet.py      # Download original
python3 scripts/filter_hetionet.py     # Create filtered version

# Option B: Contact instructor for direct file
```

### Step 2: Verify Setup

```bash
# Test loading the graph
python3 -c "
from knowledge_sources.knowledge_graph import KnowledgeGraphSource
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')
print(f'✅ Successfully loaded {kg.graph.number_of_nodes()} nodes')
"
```

**Expected output:**
```
Graph loaded from data/hetionet/hetionet_filtered.pkl
✅ Successfully loaded 7339 nodes
```

### Step 3: Quick Test

```bash
# Run inspector to see KG stats
python3 scripts/check_kg.py
```

---

## 🏗️ Knowledge Graph Structure

### Node Types (4 types, 7,339 nodes)

| Type | Count | Description | Example |
|------|-------|-------------|---------|
| **Compound** | 1,089 | Drugs/medications | Metformin, Aspirin |
| **Disease** | 134 | Medical conditions | Type 2 Diabetes, Hypertension |
| **Side Effect** | 5,701 | Drug side effects | Nausea, Headache |
| **Symptom** | 415 | Disease symptoms | Fatigue, Fever |

### Relationship Types (3 types, 143,056 edges)

| Relationship | Direction | Count | Purpose | Example |
|--------------|-----------|-------|---------|---------|
| **treats** | Drug → Disease | 755 | What treats what | Metformin treats Type 2 Diabetes |
| **causes** | Drug → Side Effect | 138,944 | Drug side effects | Metformin causes Nausea |
| **presents** | Disease → Symptom | 3,357 | Disease symptoms | Diabetes presents Thirst |

### Graph Properties

```python
{
    'num_nodes': 7339,
    'num_edges': 143056,
    'density': 0.002656,    # Sparse graph (realistic)
    'is_connected': True,   # All nodes reachable
    'directed': True,       # Edges have direction
    'multigraph': True      # Multiple edges allowed
}
```

---

## 🧠 Core Concepts

### 1. Node Identification

Each node has a unique ID format:
```
<Type>::<Identifier>

Examples:
  Compound::DB00331        → Metformin
  Disease::DOID:9351       → Type 2 Diabetes
  Side Effect::C0027497    → Nausea
```

### 2. Node Attributes

Every node has:
```python
{
    'name': 'Metformin',           # Human-readable name
    'type': 'Compound',            # Node category
    'identifier': 'DB00331'        # Database ID
}
```

### 3. Edge Attributes

Every edge has:
```python
{
    'relation': 'treats',          # Relationship type
    'direction': 'forward'         # Edge direction info
}
```

### 4. MultiDiGraph Explained

**MultiDiGraph** = Multiple + Directed + Graph
- **Multiple**: Can have multiple edges between same nodes
- **Directed**: A→B is different from B→A
- **Graph**: Network of nodes and edges

**Why this matters:**
```python
# Same drug can have multiple relationships with same target
Drug → Gene (binds)
Drug → Gene (upregulates)
Drug → Gene (downregulates)
```

---

## 📖 Usage Guide

### Basic Usage Pattern

```python
from knowledge_sources.knowledge_graph import KnowledgeGraphSource

# 1. Initialize
kg = KnowledgeGraphSource()

# 2. Load graph
kg.load_graph('data/hetionet/hetionet_filtered.pkl')

# 3. Query
matches = kg.search_drug_by_name("metformin")

# 4. Explore
drug_id = matches[0]
for _, target, data in kg.graph.out_edges(drug_id, data=True):
    print(f"{data['relation']}: {kg.graph.nodes[target]['name']}")
```

### Load Time

- **Filtered graph**: ~0.5 seconds ⚡
- **Original graph**: ~3 seconds (if you recreate it)

### Memory Usage

- **Filtered graph**: ~200 MB in RAM
- **Original graph**: ~1.5 GB in RAM

---

## 🎯 Common Tasks

### Task 1: Check KG Statistics

**CLI Method:**
```bash
python3 scripts/check_kg.py
```

**Python Method:**
```python
from knowledge_sources.knowledge_graph import KnowledgeGraphSource

kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')

stats = kg.get_statistics()
print(f"Nodes: {stats['num_nodes']:,}")
print(f"Edges: {stats['num_edges']:,}")
print(f"Density: {stats['density']:.6f}")
```

---

### Task 2: Search for a Drug

**CLI Method:**
```bash
python3 scripts/check_kg.py metformin
python3 scripts/check_kg.py search statin
```

**Python Method:**
```python
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')

# Search by name (case-insensitive, partial match)
matches = kg.search_drug_by_name("metformin")

if matches:
    drug_id = matches[0]
    drug_data = kg.graph.nodes[drug_id]
    print(f"Found: {drug_data['name']}")
    print(f"ID: {drug_id}")
else:
    print("Drug not found")
```

---

### Task 3: Get What a Drug Treats

```python
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')

# Find drug
matches = kg.search_drug_by_name("metformin")
drug_id = matches[0]

# Get diseases it treats
print(f"What {kg.graph.nodes[drug_id]['name']} treats:")
for _, target, data in kg.graph.out_edges(drug_id, data=True):
    if data['relation'] == 'treats':
        disease = kg.graph.nodes[target]['name']
        print(f"  ✓ {disease}")

# Output:
#   ✓ type 2 diabetes mellitus
#   ✓ polycystic ovary syndrome
#   ✓ metabolic syndrome X
```

---

### Task 4: Get Drug Side Effects

```python
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')

# Find drug
matches = kg.search_drug_by_name("metformin")
drug_id = matches[0]

# Get side effects
side_effects = []
for _, target, data in kg.graph.out_edges(drug_id, data=True):
    if data['relation'] == 'causes':
        side_effect = kg.graph.nodes[target]['name']
        side_effects.append(side_effect)

print(f"Side effects: {len(side_effects)}")
print(f"Examples: {', '.join(side_effects[:5])}")

# Output:
#   Side effects: 139
#   Examples: Nausea, Diarrhea, Abdominal pain, Vomiting, Flatulence
```

---

### Task 5: Count Relationships for a Drug

```python
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')

# Find drug
matches = kg.search_drug_by_name("metformin")
drug_id = matches[0]

# Count by relationship type
rel_counts = {}
for _, target, data in kg.graph.out_edges(drug_id, data=True):
    rel = data['relation']
    rel_counts[rel] = rel_counts.get(rel, 0) + 1

print("Relationship summary:")
for rel, count in sorted(rel_counts.items()):
    print(f"  {rel}: {count}")

# Output:
#   causes: 139
#   treats: 3
```

---

### Task 6: Find What Treats a Disease

```python
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')

# Find disease by searching nodes
target_disease = "type 2 diabetes mellitus"
disease_id = None

for node_id, data in kg.graph.nodes(data=True):
    if data.get('type') == 'Disease' and target_disease.lower() in data.get('name', '').lower():
        disease_id = node_id
        break

if disease_id:
    print(f"Treatments for {kg.graph.nodes[disease_id]['name']}:")
    
    # Find incoming 'treats' edges
    for source, target, data in kg.graph.in_edges(disease_id, data=True):
        if data['relation'] == 'treats':
            drug = kg.graph.nodes[source]['name']
            print(f"  ✓ {drug}")
```

---

### Task 7: Get Disease Symptoms

```python
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')

# Find disease
disease_name = "type 2 diabetes mellitus"
disease_id = None

for node_id, data in kg.graph.nodes(data=True):
    if data.get('type') == 'Disease' and disease_name.lower() in data.get('name', '').lower():
        disease_id = node_id
        break

if disease_id:
    print(f"Symptoms of {kg.graph.nodes[disease_id]['name']}:")
    
    # Find outgoing 'presents' edges
    for _, target, data in kg.graph.out_edges(disease_id, data=True):
        if data['relation'] == 'presents':
            symptom = kg.graph.nodes[target]['name']
            print(f"  • {symptom}")
```

---

### Task 8: Visualize a Drug

**Full graph:**
```bash
python3 scripts/create_viz.py metformin
open visualizations/metformin.html
```

**Specific relationships:**
```bash
# Treatments only
python3 scripts/viz_by_relationship.py metformin treats
open visualizations/metformin_treats.html

# Side effects only
python3 scripts/viz_by_relationship.py metformin causes
open visualizations/metformin_causes.html

# Combined view
python3 scripts/viz_by_relationship.py metformin treats,causes
open visualizations/metformin_treats_causes.html
```

---

### Task 9: List All Drugs

```python
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')

# Get all compound nodes
drugs = []
for node_id, data in kg.graph.nodes(data=True):
    if data.get('type') == 'Compound':
        drugs.append(data.get('name', node_id))

drugs.sort()
print(f"Total drugs: {len(drugs)}")
print(f"First 10: {drugs[:10]}")

# Or use the utility script
# python3 scripts/list_drugs.py --all
```

---

### Task 10: Get All Neighbors

```python
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')

# Find drug
matches = kg.search_drug_by_name("metformin")
drug_id = matches[0]

# Get all neighbors (connected nodes)
neighbors = kg.get_neighbors(drug_id)
print(f"Total neighbors: {len(neighbors)}")

# Get neighbors by relationship type
treats_neighbors = kg.get_neighbors(drug_id, relation_type='treats')
causes_neighbors = kg.get_neighbors(drug_id, relation_type='causes')

print(f"Treats: {len(treats_neighbors)}")
print(f"Causes: {len(causes_neighbors)}")
```

---

## 🔧 API Reference

### KnowledgeGraphSource Class

```python
from knowledge_sources.knowledge_graph import KnowledgeGraphSource

kg = KnowledgeGraphSource()
```

### Core Methods

#### `load_graph(path: str)`
Load graph from pickle file.
```python
kg.load_graph('data/hetionet/hetionet_filtered.pkl')
```

#### `save_graph(path: str)`
Save graph to pickle file.
```python
kg.save_graph('data/hetionet/my_graph.pkl')
```

#### `get_statistics() -> Dict`
Get graph statistics.
```python
stats = kg.get_statistics()
# Returns: {'num_nodes': 7339, 'num_edges': 143056, 'density': 0.002656, 'is_connected': True}
```

#### `search_drug_by_name(drug_name: str) -> List[str]`
Search for drugs by name (case-insensitive, partial match).
```python
matches = kg.search_drug_by_name("metformin")
# Returns: ['Compound::DB00331']
```

#### `get_neighbors(node: str, relation_type: str = None) -> List[str]`
Get connected nodes, optionally filtered by relationship type.
```python
# All neighbors
neighbors = kg.get_neighbors('Compound::DB00331')

# Only 'treats' relationships
treats = kg.get_neighbors('Compound::DB00331', relation_type='treats')
```

#### `find_path(source: str, target: str, max_length: int = 3) -> List[List[str]]`
Find paths between two nodes.
```python
paths = kg.find_path('Compound::DB00331', 'Disease::DOID:9351', max_length=3)
# Returns up to 5 paths
```

#### `filter_by_relationships(allowed_relations: List[str], remove_isolated_nodes: bool = True) -> Dict`
Filter graph to keep only specific relationships.
```python
stats = kg.filter_by_relationships(['treats', 'causes', 'presents'])
# Returns filtering statistics
```

### Direct Graph Access

Access NetworkX graph directly for advanced operations:

```python
# Access nodes
kg.graph.nodes[node_id]                    # Get node data
kg.graph.nodes(data=True)                  # Iterate all nodes

# Access edges
kg.graph.out_edges(node_id, data=True)     # Outgoing edges
kg.graph.in_edges(node_id, data=True)      # Incoming edges
kg.graph.edges(data=True)                  # All edges

# Graph properties
kg.graph.number_of_nodes()                 # Node count
kg.graph.number_of_edges()                 # Edge count
kg.graph.has_node(node_id)                 # Check node exists
kg.graph.has_edge(source, target)          # Check edge exists
```

---

## 🛠️ Troubleshooting

### Problem 1: "Graph file not found"

**Error:**
```
FileNotFoundError: data/hetionet/hetionet_filtered.pkl not found
```

**Solution:**
```bash
# Check if file exists
ls data/hetionet/hetionet_filtered.pkl

# If missing, recreate:
python3 scripts/setup_hetionet.py      # Download original
python3 scripts/filter_hetionet.py     # Create filtered
```

---

### Problem 2: "Drug not found"

**Error:**
```
❌ Drug 'aspirin' not found
```

**Solution:**
```bash
# Drugs use chemical names, search first:
python3 scripts/check_kg.py search aspirin
# Shows: Acetylsalicylic acid

# Then use correct name:
python3 scripts/check_kg.py "Acetylsalicylic acid"
```

---

### Problem 3: "Module not found"

**Error:**
```
ModuleNotFoundError: No module named 'networkx'
```

**Solution:**
```bash
pip install -r requirements.txt
```

---

### Problem 4: "Slow loading"

**Issue:** Graph takes long to load

**Solution:**
```python
# You're using filtered graph, right?
kg.load_graph('data/hetionet/hetionet_filtered.pkl')  # ✅ Fast (0.5s)

# Not this:
kg.load_graph('data/hetionet/hetionet_graph.pkl')     # ❌ Slow (3s) + file deleted
```

---

### Problem 5: "Empty results"

**Issue:** Query returns no results

**Solution:**
```python
# Check if drug exists
matches = kg.search_drug_by_name("metformin")
if not matches:
    print("Drug not found - try searching")

# Check if relationship exists
rel_counts = {}
for _, _, data in kg.graph.out_edges(drug_id, data=True):
    rel = data['relation']
    rel_counts[rel] = rel_counts.get(rel, 0) + 1
print(f"Available relationships: {rel_counts}")
```

---

## 📚 Quick Reference

### Essential Commands

```bash
# Check KG
python3 scripts/check_kg.py

# Search drug
python3 scripts/check_kg.py metformin
python3 scripts/check_kg.py search statin

# Visualize
python3 scripts/create_viz.py metformin
python3 scripts/viz_by_relationship.py metformin treats

# Interactive explorer
python3 scripts/view_knowledge_graph.py interactive
```

### Essential Code Snippets

```python
from knowledge_sources.knowledge_graph import KnowledgeGraphSource

# Load
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')

# Search
matches = kg.search_drug_by_name("metformin")
drug_id = matches[0]

# Query relationships
for _, target, data in kg.graph.out_edges(drug_id, data=True):
    if data['relation'] == 'treats':
        print(kg.graph.nodes[target]['name'])
```

---

## 🎓 Best Practices

### 1. Always Use Filtered Graph
```python
# ✅ Good - Fast, focused
kg.load_graph('data/hetionet/hetionet_filtered.pkl')

# ❌ Bad - Slow, too much data, file deleted anyway
kg.load_graph('data/hetionet/hetionet_graph.pkl')
```

### 2. Search Before Querying
```python
# ✅ Good - Verify drug exists first
matches = kg.search_drug_by_name("metformin")
if matches:
    drug_id = matches[0]
    # ... query

# ❌ Bad - Assumes drug ID
drug_id = "Compound::DB00331"  # What if wrong?
```

### 3. Check Relationship Type
```python
# ✅ Good - Filter by relationship
for _, target, data in kg.graph.out_edges(drug_id, data=True):
    if data['relation'] == 'treats':
        # ... process

# ❌ Bad - Process all relationships
for _, target, data in kg.graph.out_edges(drug_id, data=True):
    # ... assumes all are 'treats'
```

### 4. Use CLI Tools for Quick Checks
```bash
# ✅ Good - Quick check
python3 scripts/check_kg.py metformin

# ❌ Bad - Write Python script for one-time check
```

### 5. Visualize Complex Queries
```bash
# ✅ Good - Visual understanding
python3 scripts/viz_by_relationship.py metformin treats,causes

# ❌ Bad - Try to understand 139 side effects from text
```

---

## 📖 Related Documentation

- **[FILTERED_KG_GUIDE.md](reference/FILTERED_KG_GUIDE.md)** - Detailed filtered KG usage
- **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)** - Complete visualization guide
- **[README.md](../README.md)** - Project overview

---

## 🚀 Next Steps

1. **Integrate with Adaptive Selector**
   - Use KG as one of 4 knowledge sources
   - Query for drug-disease relationships
   - Fast, structured answers

2. **Extend Functionality**
   - Add custom relationships
   - Create subgraphs
   - Advanced path finding

3. **Optimize Performance**
   - Cache common queries
   - Pre-compute frequently accessed data
   - Batch processing

---

**Your Knowledge Graph is ready to power your adaptive knowledge selector!** 🎉

Need help? Check the troubleshooting section or contact your instructor.
