# Knowledge Graph Visualization Guide

Complete guide to visualizing and exploring your knowledge graph.

---

## 🎨 Available Visualization Methods

### 1. **Interactive HTML (Best for Exploration!)**
```bash
# Create interactive web-based visualization
python3 scripts/simple_html_viz.py metformin

# Opens in browser - you can:
# - Drag nodes around
# - Click for details
# - Zoom and pan
# - Hover for info
```

**Output**: HTML file you can open in any web browser
**Location**: `visualizations/metformin_interactive.html`

---

### 2. **Static Image Visualization**
```bash
# Create PNG image of drug neighborhood
python3 scripts/visualize_graph.py drug metformin
```

**Output**: High-resolution PNG image
**Location**: `visualizations/metformin_graph.png`

---

### 3. **Text-Based Exploration**
```bash
# Interactive command-line exploration
python3 scripts/view_knowledge_graph.py interactive

# Then type commands like:
> search metformin
> explore Compound::DB00331
> stats
```

---

## 📂 Generated Files

After running visualizations, check the `visualizations/` folder:

```
visualizations/
├── metformin_graph.png                  # Static image
├── metformin_interactive.html           # Interactive web page
├── atorvastatin_graph.png              # Other drugs...
└── treats_network.png                   # Relationship networks
```

---

## 🌟 **Recommended Workflow**

### Step 1: Find Your Drug
```bash
python3 scripts/list_drugs.py --search <drug_name>
```

### Step 2: Create Interactive Visualization
```bash
python3 scripts/simple_html_viz.py <drug_name>
```

### Step 3: Open in Browser
```bash
# macOS
open visualizations/<drug_name>_interactive.html

# Or just double-click the HTML file
```

---

## 🔍 What You Can See in Visualizations

### Node Types (Color Coded)
- 🔴 **Red** - Compounds (Drugs)
- 🟦 **Teal** - Diseases
- 🟩 **Green** - Genes
- 🟨 **Yellow** - Side Effects
- 🟦 **Blue** - Pathways
- 🟪 **Purple** - Other types

### Relationships (Arrows)
- → Directed relationships
- Arrow shows direction of relationship
- Hover to see relationship type

---

## 💡 Example Visualizations

### Example 1: Metformin Network
```bash
python3 scripts/simple_html_viz.py metformin
```

**What you'll see:**
- Metformin (center, large red circle)
- Connected to 20-30 surrounding entities
- Side effects it causes (yellow)
- Genes it binds to (green)
- Diseases it treats (teal)

**Interactive features:**
- Drag Metformin around
- Click any node to see:
  - Name and type
  - All its connections
  - Relationship details
- Nodes bounce and connect dynamically

---

### Example 2: Compare Multiple Drugs
```bash
# Visualize different drugs
python3 scripts/simple_html_viz.py metformin
python3 scripts/simple_html_viz.py atorvastatin
python3 scripts/simple_html_viz.py ibuprofen

# Open all three in browser tabs to compare
```

---

### Example 3: Disease-Drug Network
```bash
# Find drugs that treat a disease
python3 scripts/view_knowledge_graph.py interactive

# Then:
> search diabetes
> explore Disease::DOID:9351
```

---

## 🎯 Advanced: Custom Visualizations

### Create Your Own Subgraph
```python
from knowledge_sources.knowledge_graph import KnowledgeGraphSource
import matplotlib.pyplot as plt
import networkx as nx

# Load graph
kg = KnowledgeGraphSource()
kg.load_graph('data/hetionet/hetionet_graph.pkl')

# Find specific drugs
metformin = kg.search_drug_by_name("metformin")[0]
insulin = kg.search_drug_by_name("insulin")  # If available

# Get their neighborhoods
neighbors1 = kg.get_neighbors(metformin)[:10]

# Create subgraph
nodes = [metformin] + neighbors1
subgraph = kg.graph.subgraph(nodes)

# Visualize
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(subgraph)
nx.draw(subgraph, pos, with_labels=True, node_size=500)
plt.savefig('my_custom_viz.png', dpi=300)
```

---

## 🖼️ Understanding the Visualizations

### Interactive HTML Features

**1. Node Information Panel (Right Side)**
- Click any node
- Shows: Name, Type, ID, Connections
- Lists relationships

**2. Legend (Bottom Right)**
- Color coding for node types
- Quick reference

**3. Physics Simulation**
- Nodes push/pull each other
- Connected nodes stay together
- Drag nodes to rearrange
- Auto-adjusts to your view

**4. Hover Effects**
- Nodes grow on hover
- Easy to see what you're selecting

---

### Static PNG Features

**1. Color Coding**
- Clear visual separation by type

**2. Arrows**
- Show relationship direction
- From source → to target

**3. Labels**
- Node names shown
- Truncated if too long

**4. Layout**
- Spring layout (force-directed)
- Most connected in center
- Clear visual clustering

---

## 📊 Graph Statistics in Visualizations

When you create visualizations, you'll see:

```
Metformin Network:
├── Center node: Metformin (Compound)
├── 139 "causes" → Side effects
├── 56 "binds" → Genes
├── 22 "downregulates" → Genes
├── 8 "upregulates" → Genes
└── 3 "treats" → Diseases
```

---

## 🚀 Quick Reference

| Want to... | Command |
|-----------|---------|
| Interactive web view | `python3 scripts/simple_html_viz.py <drug>` |
| Static image | `python3 scripts/visualize_graph.py drug <drug>` |
| Text exploration | `python3 scripts/view_knowledge_graph.py interactive` |
| List all drugs | `python3 scripts/list_drugs.py` |
| Search drugs | `python3 scripts/list_drugs.py --search <term>` |

---

## 🎓 Understanding Graph Structure

### How It's Built

```
JSON File (Hetionet)
       ↓
NetworkX MultiDiGraph
       ↓
Nodes: 47K entities
Edges: 2.25M relationships
       ↓
Visualization Tools
```

### Internal Structure

**Nodes**:
```python
'Compound::DB00331': {
    'name': 'Metformin',
    'type': 'Compound',
    'identifier': 'DB00331'
}
```

**Edges**:
```python
'Compound::DB00331' → 'Disease::DOID:9351'
    relation: 'treats'
```

See [GRAPH_EXPLANATION.md](GRAPH_EXPLANATION.md) for detailed technical explanation.

---

## 💻 System Requirements

**For Interactive HTML**:
- Any modern web browser
- No special requirements
- Works offline

**For Static Images**:
- matplotlib installed ✓
- ~8GB RAM
- Graphical display (or save to file)

---

## 🎨 Customization Tips

### Change Colors
Edit `scripts/simple_html_viz.py`:
```javascript
const colorMap = {
    'Compound': '#YOUR_COLOR',  // Change these
    'Disease': '#YOUR_COLOR',
    // ...
};
```

### Change Number of Neighbors
```bash
# Edit max_neighbors in the script
# or modify function call:
create_simple_html(kg, "metformin", max_neighbors=50)
```

### Change Layout
For static images, try different layouts:
- `nx.spring_layout()`  ← Current
- `nx.circular_layout()`
- `nx.kamada_kawai_layout()`
- `nx.random_layout()`

---

## 🐛 Troubleshooting

**"Can't open HTML file"**
- Copy the full path shown in output
- Paste in browser address bar
- Or right-click HTML → Open With → Browser

**"Image looks messy"**
- Reduce `max_neighbors` parameter
- Try different drugs (some have fewer connections)
- Increase figure size in code

**"Visualization takes forever"**
- Reduce number of nodes
- Use pickle file (faster loading)
- Close other applications

---

## 📖 Next Steps

1. ✅ Visualize several drugs
2. ✅ Compare their networks
3. ✅ Explore relationships
4. Build selector model to choose best source
5. Use visualizations to understand why certain sources are better

---

**Your graph is ready to explore!** 🎉

Start with:
```bash
python3 scripts/simple_html_viz.py metformin
```

Then open the HTML file and start clicking around!
