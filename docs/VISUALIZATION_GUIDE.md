# 🎨 Knowledge Graph Visualization Guide

## 📊 Overview

You have **4 types of visualizations** available:

1. **Full Graph View** - Show all relationships for a drug
2. **Filtered View** - Show only specific relationships (treats, causes, presents)
3. **Multiple Relationships** - Combine multiple relationship types
4. **Interactive Explorer** - Browse the entire graph

All visualizations are **interactive D3.js graphs** with:
- ✨ Drag nodes to rearrange
- 🖱️ Click nodes for details
- 🎨 Color-coded by node type
- 🔍 Hover to highlight connections

---

## 🚀 Quick Start

### 1. **Full Graph View** (All Relationships)

Show all connections for a drug (up to 30 neighbors):

```bash
python3 scripts/create_viz.py metformin
python3 scripts/create_viz.py ibuprofen
python3 scripts/create_viz.py atorvastatin
```

**Output:** `visualizations/metformin.html`

**When to use:**
- Get complete overview of a drug
- See all connections at once
- Explore drug's full profile

---

### 2. **Filtered View** (Single Relationship)

Show only ONE type of relationship:

```bash
# What does it treat?
python3 scripts/viz_by_relationship.py metformin treats

# What side effects does it cause?
python3 scripts/viz_by_relationship.py metformin causes

# What symptoms does a disease present?
python3 scripts/viz_by_relationship.py "type 2 diabetes" presents
```

**Output:** `visualizations/metformin_treats.html`

**When to use:**
- Focus on specific aspect (treatments OR side effects)
- Cleaner, less cluttered view
- Answer specific questions

---

### 3. **Multiple Relationships** (Combination)

Combine multiple relationship types:

```bash
# Show both treats and causes
python3 scripts/viz_by_relationship.py metformin treats,causes

# All three relationships
python3 scripts/viz_by_relationship.py metformin treats,causes,presents
```

**Output:** `visualizations/metformin_treats_causes.html`

**When to use:**
- Compare different aspects
- Risk-benefit analysis (treats vs causes)
- Multi-dimensional view

---

### 4. **Interactive Explorer**

Browse the entire graph interactively:

```bash
python3 scripts/view_knowledge_graph.py interactive
```

**Commands:**
- `search metformin` - Find a drug
- `explore Compound::DB00331` - Explore a node
- `stats` - Show statistics
- `quit` - Exit

**When to use:**
- Explore multiple drugs
- Navigate relationships
- Research and discovery

---

## 🎨 Visualization Features

### Color Coding

**Node Types:**
- 🔴 **Red** = Compound (Drugs)
- 🟦 **Teal** = Disease
- 🟩 **Green** = Symptom
- 🟡 **Yellow** = Side Effect

**Relationship Colors (Filtered Views):**
- 🟢 **Green** = treats
- 🔴 **Red** = causes
- 🔵 **Blue** = presents

### Interactive Features

1. **Drag Nodes** - Click and drag to rearrange
2. **Click Info** - Click any node to see details
3. **Hover Highlight** - Hover to highlight connections
4. **Zoom** - Scroll to zoom in/out
5. **Pan** - Click and drag background to pan

---

## 📝 Common Use Cases

### Use Case 1: What does a drug treat?

```bash
# Create visualization
python3 scripts/viz_by_relationship.py metformin treats

# Open in browser
open visualizations/metformin_treats.html
```

**You'll see:**
- Drug in center (red)
- Diseases around it (teal)
- Green arrows showing "treats" relationship

---

### Use Case 2: What side effects does a drug have?

```bash
# Create visualization
python3 scripts/viz_by_relationship.py metformin causes

# Open in browser
open visualizations/metformin_causes.html
```

**You'll see:**
- Drug in center (red)
- Side effects around it (yellow)
- Red arrows showing "causes" relationship

---

### Use Case 3: Compare treatments vs side effects

```bash
# Create combined visualization
python3 scripts/viz_by_relationship.py metformin treats,causes

# Open in browser
open visualizations/metformin_treats_causes.html
```

**You'll see:**
- Drug in center (red)
- Diseases it treats (teal, green arrows)
- Side effects it causes (yellow, red arrows)

---

### Use Case 4: Explore a disease

```bash
# What symptoms does diabetes present?
python3 scripts/viz_by_relationship.py "type 2 diabetes mellitus" presents

# What treats diabetes?
python3 scripts/check_kg.py "type 2 diabetes mellitus"
```

---

## 🎯 Advanced Options

### Customize Number of Neighbors

Edit the script to show more/fewer nodes:

```python
# In scripts/create_viz.py, line ~14:
def create_working_html(kg, drug_name, max_neighbors=30):
                                            # ^^^ Change this number

# Show 50 neighbors instead of 30
max_neighbors=50
```

### Control Nodes Per Relationship

For filtered views:

```python
# In scripts/viz_by_relationship.py, line ~17:
def create_filtered_viz(kg, drug_name, relationship_types=None, max_per_type=10):
                                                                # ^^^ Change this

# Show 20 nodes per relationship type
max_per_type=20
```

---

## 📂 Output Files

All visualizations are saved to `visualizations/` folder:

```
visualizations/
├── metformin.html                    # Full graph
├── metformin_treats.html             # Treats only
├── metformin_causes.html             # Side effects only
├── metformin_treats_causes.html      # Combined
├── atorvastatin.html                 # Another drug
└── ibuprofen.html                    # Another drug
```

Each file is **self-contained** - no internet connection needed to view!

---

## 🌐 How to Open Visualizations

### Method 1: Command Line
```bash
open visualizations/metformin.html
```

### Method 2: Browser
1. Open your browser (Chrome, Firefox, Safari)
2. Press **Cmd+O** (Mac) or **Ctrl+O** (Windows)
3. Navigate to `visualizations/` folder
4. Select the HTML file

### Method 3: Finder/Explorer
1. Navigate to `visualizations/` folder
2. Double-click the HTML file

---

## 💡 Pro Tips

### 1. Create Multiple Views

```bash
# Create different perspectives of the same drug
python3 scripts/create_viz.py metformin              # Full view
python3 scripts/viz_by_relationship.py metformin treats    # Treatments
python3 scripts/viz_by_relationship.py metformin causes    # Side effects
```

### 2. Compare Drugs

```bash
# Visualize different drugs
python3 scripts/create_viz.py metformin
python3 scripts/create_viz.py insulin
python3 scripts/create_viz.py glipizide

# Open all in different browser tabs to compare
```

### 3. Search First

If you don't know exact drug name:

```bash
# Search first
python3 scripts/check_kg.py search statin

# Then visualize
python3 scripts/create_viz.py atorvastatin
```

### 4. Start Simple

For complex drugs with many connections:
```bash
# Start with filtered view (cleaner)
python3 scripts/viz_by_relationship.py atorvastatin treats

# Then explore full graph if needed
python3 scripts/create_viz.py atorvastatin
```

---

## 🛠️ Available Scripts

| Script | Purpose | Example |
|--------|---------|---------|
| `create_viz.py` | Full graph view | `python3 scripts/create_viz.py metformin` |
| `viz_by_relationship.py` | Filtered view | `python3 scripts/viz_by_relationship.py metformin treats` |
| `view_knowledge_graph.py` | Interactive CLI | `python3 scripts/view_knowledge_graph.py interactive` |
| `visualize_graph.py` | Advanced viz | `python3 scripts/visualize_graph.py metformin` |
| `simple_html_viz.py` | Simple viz | `python3 scripts/simple_html_viz.py metformin` |

---

## 📊 Example Workflows

### Workflow 1: Research a New Drug

```bash
# 1. Search for the drug
python3 scripts/check_kg.py search metformin

# 2. Get basic info
python3 scripts/check_kg.py metformin

# 3. Visualize treatments
python3 scripts/viz_by_relationship.py metformin treats

# 4. Visualize side effects
python3 scripts/viz_by_relationship.py metformin causes

# 5. Full overview
python3 scripts/create_viz.py metformin

# 6. Open all visualizations
open visualizations/metformin*.html
```

### Workflow 2: Compare Treatment Options

```bash
# For diabetes treatment comparison
python3 scripts/viz_by_relationship.py metformin treats,causes
python3 scripts/viz_by_relationship.py glipizide treats,causes
python3 scripts/viz_by_relationship.py insulin treats,causes

# Open all to compare side-by-side
```

### Workflow 3: Investigate Disease

```bash
# 1. Check info
python3 scripts/check_kg.py "type 2 diabetes mellitus"

# 2. See symptoms
python3 scripts/viz_by_relationship.py "type 2 diabetes mellitus" presents

# 3. Find treatments (use code)
python3 -c "
from knowledge_sources.knowledge_graph import KnowledgeGraphSource
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_filtered.pkl')
# Find drugs that treat this disease
for u, v, data in kg.graph.edges(data=True):
    if data['relation'] == 'treats' and 'type 2 diabetes' in kg.graph.nodes[v]['name'].lower():
        print(kg.graph.nodes[u]['name'])
"
```

---

## 🎯 Quick Reference

```bash
# FULL GRAPH
python3 scripts/create_viz.py <drug_name>

# FILTERED (ONE RELATIONSHIP)
python3 scripts/viz_by_relationship.py <drug_name> treats
python3 scripts/viz_by_relationship.py <drug_name> causes
python3 scripts/viz_by_relationship.py <drug_name> presents

# COMBINED (MULTIPLE RELATIONSHIPS)
python3 scripts/viz_by_relationship.py <drug_name> treats,causes
python3 scripts/viz_by_relationship.py <drug_name> treats,causes,presents

# OPEN VISUALIZATION
open visualizations/<drug_name>.html

# INTERACTIVE EXPLORER
python3 scripts/view_knowledge_graph.py interactive
```

---

## 🚨 Troubleshooting

### "Drug not found"
```bash
# Search first to find exact name
python3 scripts/check_kg.py search <partial_name>
```

### "No visualization created"
```bash
# Check if drug has any relationships
python3 scripts/check_kg.py <drug_name>
```

### "File not opening"
```bash
# Use full path
open "/Users/harshtita/Desktop/Arizona State University/KRR - Spring26/Adaptive-Knowledge-Selector/visualizations/metformin.html"
```

### "Graph looks cluttered"
```bash
# Use filtered view instead
python3 scripts/viz_by_relationship.py <drug_name> treats
```

---

## 📚 Related Tools

- **Check KG**: `python3 scripts/check_kg.py`
- **List Drugs**: `python3 scripts/list_drugs.py --search <term>`
- **Compare Graphs**: `python3 scripts/compare_graphs.py`

---

**Happy Visualizing! 🎨**

All your visualizations are interactive, beautiful, and saved as HTML files you can share or present!
