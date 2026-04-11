# Filtered Knowledge Graph Guide

## 📋 Overview

Your Knowledge Graph has been filtered to focus on **medically relevant relationships** for drug and disease queries. This makes it faster, smaller, and more focused.

---

## 🎯 What Relationships Are Kept?

### 1. **treats** (755 relationships)
- **Direction:** Compound → Disease
- **Meaning:** Which drugs treat which diseases
- **Example:** Metformin treats Type 2 Diabetes

### 2. **causes** (138,944 relationships)
- **Direction:** Compound → Side Effect
- **Meaning:** What side effects drugs can cause
- **Example:** Metformin causes Nausea, Diarrhea, etc.

### 3. **presents** (3,357 relationships)
- **Direction:** Disease → Symptom
- **Meaning:** What symptoms a disease presents with
- **Example:** Type 2 Diabetes presents with Increased Thirst, Fatigue

---

## 📊 Before vs After Filtering

| Metric | Original (Full) | Filtered | Reduction |
|--------|-----------------|----------|-----------|
| **Nodes** | 47,031 | 7,339 | -84.4% |
| **Edges** | 2,250,197 | 143,056 | -93.6% |
| **Storage** | 177.76 MB | 12.18 MB | -93.1% |
| **Relationships** | 16 types | 3 types | -81.3% |

### Node Types Remaining

| Node Type | Count | Purpose |
|-----------|-------|---------|
| **Side Effect** | 5,701 | Known drug side effects |
| **Compound** | 1,089 | Drugs/medications |
| **Symptom** | 415 | Disease symptoms |
| **Disease** | 134 | Medical conditions |
| **TOTAL** | **7,339** | |

---

## 🚀 How to Use

### Option 1: Load Filtered Graph (Recommended)

```python
from knowledge_sources.knowledge_graph import KnowledgeGraphSource

kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')  # ← Filtered version

# Query as usual
matches = kg.search_drug_by_name("metformin")
```

### Option 2: Load Original Full Graph

```python
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_graph.pkl')  # ← Full version
```

### Option 3: Use in Unified System

```python
from examples.integrate_all_sources import UnifiedKnowledgeSystem

# Default: Uses filtered graph
system = UnifiedKnowledgeSystem(email="your@email.com")

# To use full graph:
system = UnifiedKnowledgeSystem(email="your@email.com", use_filtered_kg=False)
```

---

## 🔧 Creating the Filtered Graph

If you don't have the filtered graph yet or want to recreate it:

```bash
# Run the filter script
python scripts/filter_hetionet.py
```

This will:
1. Load the original Hetionet graph
2. Remove all edges except 'treats', 'causes', 'presents'
3. Remove isolated nodes (nodes with no edges)
4. Save to `data/hetionet/hetionet_filtered.pkl`

---

## 📝 Example Queries

### Query 1: What does Metformin treat?

```python
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')

# Find metformin
matches = kg.search_drug_by_name("metformin")
drug_id = matches[0]  # 'Compound::DB00331'

# Get diseases it treats
for _, target, data in kg.graph.out_edges(drug_id, data=True):
    if data['relation'] == 'treats':
        disease_name = kg.graph.nodes[target]['name']
        print(f"Treats: {disease_name}")

# Output:
# Treats: type 2 diabetes mellitus
# Treats: polycystic ovary syndrome
# Treats: metabolic syndrome X
```

### Query 2: What side effects does Metformin cause?

```python
# Get side effects
side_effects = []
for _, target, data in kg.graph.out_edges(drug_id, data=True):
    if data['relation'] == 'causes':
        side_effect = kg.graph.nodes[target]['name']
        side_effects.append(side_effect)

print(f"Metformin has {len(side_effects)} known side effects")
print(f"Examples: {', '.join(side_effects[:5])}")

# Output:
# Metformin has 139 known side effects
# Examples: Nausea, Diarrhea, Abdominal pain, Vomiting, Flatulence
```

### Query 3: Get relationship summary

```python
# Count relationships by type
rel_counts = {}
for _, _, data in kg.graph.out_edges(drug_id, data=True):
    rel = data['relation']
    rel_counts[rel] = rel_counts.get(rel, 0) + 1

print(f"Metformin relationships:")
for rel, count in rel_counts.items():
    print(f"  {rel}: {count}")

# Output:
# Metformin relationships:
#   causes: 139
#   treats: 3
```

### Query 4: What symptoms does Type 2 Diabetes present?

```python
# Find type 2 diabetes
disease_id = "Disease::DOID:9351"

symptoms = []
for _, target, data in kg.graph.out_edges(disease_id, data=True):
    if data['relation'] == 'presents':
        symptom = kg.graph.nodes[target]['name']
        symptoms.append(symptom)

print(f"Type 2 Diabetes presents with: {', '.join(symptoms)}")
```

---

## 🎨 Visualizations

The filtered graph works with all visualization scripts:

```bash
# Create visualization with filtered graph
python scripts/create_viz.py metformin

# Filtered relationship visualization
python scripts/viz_by_relationship.py metformin treats
python scripts/viz_by_relationship.py metformin causes
```

**Note:** Visualizations will be cleaner and faster with the filtered graph!

---

## ⚡ Performance Benefits

### Loading Speed
- **Original:** ~3 seconds
- **Filtered:** ~0.5 seconds (6x faster!)

### Query Speed
- Fewer edges to traverse = faster queries
- More focused results = less noise

### Memory Usage
- **Original:** ~1.5 GB in RAM
- **Filtered:** ~200 MB in RAM (7.5x less!)

---

## 🔄 Switching Between Graphs

### Method 1: Change file path

```python
# In your code, change:
kg.load_graph('data/hetionet/hetionet_filtered.pkl')  # Filtered
# or
kg.load_graph('data/hetionet/hetionet_graph.pkl')     # Original
```

### Method 2: Environment variable (Advanced)

Create a config file:

```python
# config.py
USE_FILTERED_KG = True  # Toggle here

KG_PATH = (
    'data/hetionet/hetionet_filtered.pkl' if USE_FILTERED_KG
    else 'data/hetionet/hetionet_graph.pkl'
)
```

Then in your code:

```python
from config import KG_PATH

kg = KnowledgeGraphSource()
kg.load_graph(KG_PATH)
```

---

## 🧪 Verification

To verify your filtered graph is working correctly:

```bash
# Compare original vs filtered
python scripts/compare_graphs.py

# Test filtered graph
python -c "
from knowledge_sources.knowledge_graph import KnowledgeGraphSource
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')
print(f'Nodes: {kg.graph.number_of_nodes()}')
print(f'Edges: {kg.graph.number_of_edges()}')
"
```

Expected output:
```
Nodes: 7339
Edges: 143056
```

---

## 📚 Use Cases

### Perfect for:
✅ Drug side effect queries
✅ Disease treatment queries  
✅ Symptom-disease associations
✅ Medical Q&A systems
✅ Drug recommendation systems

### Not suitable for:
❌ Gene-protein interactions
❌ Molecular pathways
❌ Drug-drug interactions (use RxNorm API instead)
❌ Biochemical mechanisms

---

## 🛠️ Customization

Want different relationships? Edit the filter:

```python
# In scripts/filter_hetionet.py, change:
ALLOWED_RELATIONSHIPS = ['treats', 'causes', 'presents']

# To include more relationships:
ALLOWED_RELATIONSHIPS = ['treats', 'causes', 'presents', 'binds', 'palliates']
```

Then run:
```bash
python scripts/filter_hetionet.py
```

---

## 🎯 Summary

**Filtered KG Benefits:**
- 🚀 **6x faster** loading
- 💾 **93% smaller** storage
- 🎯 **100% focused** on medical relationships
- ⚡ **Faster queries** with less noise
- 🧹 **Cleaner visualizations**

**When to use:**
- Use **filtered** for most medical Q&A tasks (default)
- Use **original** only if you need gene/pathway data

**Files:**
- Original: `data/hetionet/hetionet_graph.pkl` (178 MB)
- Filtered: `data/hetionet/hetionet_filtered.pkl` (12 MB) ⭐

---

## 📞 Quick Reference Commands

```bash
# Create filtered graph
python scripts/filter_hetionet.py

# Compare graphs
python scripts/compare_graphs.py

# Test filtered graph
python -c "from knowledge_sources.knowledge_graph import KnowledgeGraphSource; kg = KnowledgeGraphSource(); kg.load_graph('data/hetionet/hetionet_filtered.pkl'); print(kg.get_statistics())"

# Visualize with filtered graph
python scripts/create_viz.py metformin
```

---

**Made with ❤️ for KRR Course Project - Spring 2026**
