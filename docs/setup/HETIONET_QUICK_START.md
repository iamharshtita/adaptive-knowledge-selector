# Hetionet Quick Start Guide

**Hetionet** is now the primary knowledge graph source for this project. No account or registration needed!

---

## Why Hetionet?

✅ **100% FREE** - No account, no license, no restrictions
✅ **Comprehensive** - 47,031 nodes, 2.25M relationships
✅ **Research-grade** - Published in eLife journal
✅ **Fast setup** - Single command to download
✅ **Rich data** - Drugs, diseases, genes, proteins, side effects, and more

---

## Quick Setup (5 minutes)

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
python scripts/setup_hetionet.py
```

This will:
1. Download Hetionet (~260MB)
2. Decompress the file
3. Load into knowledge graph
4. Save as pickle for fast loading
5. Run tests to verify

### Option 2: Manual Download

```bash
# Download and extract
wget https://github.com/hetio/hetionet/raw/master/hetnet/json/hetionet-v1.0.json.bz2
bunzip2 hetionet-v1.0.json.bz2

# Move to data directory
mkdir -p data/hetionet
mv hetionet-v1.0.json data/hetionet/
```

---

## Usage in Your Code

### Loading Hetionet

```python
from knowledge_sources.knowledge_graph import KnowledgeGraphSource

# Initialize
kg = KnowledgeGraphSource()

# Option 1: Load from pickle (FAST - ~10 seconds)
kg.load_graph('data/hetionet/hetionet_graph.pkl')

# Option 2: Load from JSON (SLOWER - ~2 minutes)
kg.load_hetionet('data/hetionet/hetionet-v1.0.json')
```

### Basic Queries

```python
# Get statistics
stats = kg.get_statistics()
print(f"Nodes: {stats['num_nodes']:,}")
print(f"Edges: {stats['num_edges']:,}")

# Search for drugs
matches = kg.search_drug_by_name("Metformin")
print(f"Found: {matches}")

# Get neighbors
if matches:
    drug_id = matches[0]
    neighbors = kg.get_neighbors(drug_id)
    print(f"Connected to {len(neighbors)} entities")

# Find paths between entities
paths = kg.find_path(source_node, target_node, max_length=3)
```

---

## What's in Hetionet?

### Node Types (11 total)
- **Compound** - Drugs and chemicals
- **Disease** - Diseases and conditions
- **Gene** - Human genes
- **Anatomy** - Anatomical structures
- **Biological Process** - GO biological processes
- **Cellular Component** - GO cellular components
- **Molecular Function** - GO molecular functions
- **Pathway** - Biological pathways
- **Pharmacologic Class** - Drug classes
- **Side Effect** - Adverse drug reactions
- **Symptom** - Disease symptoms

### Relationship Types (24 total)
Examples:
- `Compound--treats-->Disease`
- `Compound--binds-->Gene`
- `Compound--causes-->Side Effect`
- `Disease--associates-->Gene`
- `Gene--participates-->Pathway`
- And 19 more...

---

## Hetionet vs DrugBank Comparison

| Feature | Hetionet | DrugBank |
|---------|----------|----------|
| **Cost** | FREE | Free academic, $2,500/yr commercial |
| **Setup** | 1 command | Account + download + parse XML |
| **Nodes** | 47,031 | ~14,000 drugs only |
| **Relationships** | 2.25M across 11 types | Primarily drug-focused |
| **Data Types** | Drugs, diseases, genes, proteins, etc. | Primarily drugs |
| **Integration** | Already coded | Requires XML parsing |
| **Time to Setup** | 5 minutes | 30+ minutes |

---

## Testing Your Setup

```bash
# Test the knowledge graph
python knowledge_sources/knowledge_graph.py

# Test integrated system
python examples/integrate_all_sources.py
```

---

## Example Queries for Medical Domain

### 1. Drug Interactions
```python
# Note: Hetionet doesn't directly encode drug-drug interactions
# But you can find indirect relationships through shared targets/pathways

drug_id = kg.search_drug_by_name("Metformin")[0]
neighbors = kg.get_neighbors(drug_id)
# Analyze connected genes, proteins, diseases
```

### 2. Disease Treatments
```python
# Find compounds that treat a disease
# Look for edges: Compound--treats-->Disease

for source, target, data in kg.graph.edges(data=True):
    if data.get('relation') == 'CtD' and target == 'Disease::DOID:9351':  # Diabetes
        print(f"Treatment: {source}")
```

### 3. Drug Targets
```python
# Find genes/proteins that a drug binds to
drug_id = "Compound::DB00331"  # Metformin's ID in Hetionet

for source, target, data in kg.graph.edges(drug_id, data=True):
    if data.get('relation') in ['CbG', 'CdG', 'CuG']:  # Binds, downregulates, upregulates
        print(f"Target: {target}")
```

---

## Hetionet Identifier Format

Hetionet uses specific ID formats:
- **Compounds**: `Compound::DB00331` (DrugBank IDs)
- **Diseases**: `Disease::DOID:9351` (Disease Ontology IDs)
- **Genes**: `Gene::1234` (Entrez Gene IDs)

Use `search_drug_by_name()` to convert common names to Hetionet IDs.

---

## Relationship Abbreviations

Common relationship codes in Hetionet:
- `CbG` - Compound binds Gene
- `CtD` - Compound treats Disease
- `CcSE` - Compound causes Side Effect
- `DaG` - Disease associates Gene
- `GiG` - Gene interacts Gene
- `GpPW` - Gene participates Pathway

Full list: https://het.io/about/

---

## Performance Tips

1. **Use pickle for repeated loading**
   ```python
   # First time: load JSON and save pickle
   kg.load_hetionet('data/hetionet/hetionet-v1.0.json')
   kg.save_graph('data/hetionet/hetionet_graph.pkl')

   # Future times: load pickle (10x faster)
   kg.load_graph('data/hetionet/hetionet_graph.pkl')
   ```

2. **Index frequently queried nodes**
   - Build a drug name → ID mapping once
   - Cache search results

3. **Limit path search depth**
   ```python
   paths = kg.find_path(source, target, max_length=3)  # Not 5+
   ```

---

## Troubleshooting

### "Hetionet not found"
```bash
# Run setup script
python scripts/setup_hetionet.py

# Or check file location
ls -la data/hetionet/
```

### "Memory error loading JSON"
- You need ~8GB RAM to load Hetionet
- Use pickle file instead (much smaller)
- Close other applications

### "Drug name not found"
- Hetionet uses DrugBank IDs internally
- Try searching partial names: `search_drug_by_name("met")`
- Check drug synonyms

---

## Additional Resources

- **Hetionet Website**: https://het.io/
- **GitHub Repository**: https://github.com/hetio/hetionet
- **Research Paper**: https://elifesciences.org/articles/26726
- **Node/Edge Browser**: https://het.io/browser/

---

## Next Steps

1. ✅ Download Hetionet: `python scripts/setup_hetionet.py`
2. ✅ Test loading: `python knowledge_sources/knowledge_graph.py`
3. ✅ Integrate: `python examples/integrate_all_sources.py`
4. Build selector model for adaptive routing
5. Train with reinforcement learning

---

**You're all set!** Hetionet provides everything you need for the knowledge graph source without any account setup.
