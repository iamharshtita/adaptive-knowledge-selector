# Documentation

Complete documentation for the Adaptive Knowledge Source Selection System.

## Main Guides

Comprehensive guides for getting started:

- [Knowledge Graph Guide](KNOWLEDGE_GRAPH_GUIDE.md) - Complete guide with prerequisites, setup flow, and API reference
- [Visualization Guide](VISUALIZATION_GUIDE.md) - Complete visualization guide with all methods and examples

## Setup Guides

Get started quickly with these setup guides:

- [Setup Guide](setup/SETUP_GUIDE.md) - Complete installation and setup instructions
- [Hetionet Quick Start](setup/HETIONET_QUICK_START.md) - Quick start guide for Hetionet knowledge graph

## Reference Documentation

Technical references and resources:

- [Filtered KG Guide](reference/FILTERED_KG_GUIDE.md) - Details about the filtered knowledge graph
- [Graph Explanation](reference/GRAPH_EXPLANATION.md) - Technical details on graph structure and implementation
- [Resource Access Guide](reference/RESOURCE_ACCESS_GUIDE.md) - How to access all knowledge sources

---

## Quick Links

**Setup & Installation:**
```bash
# Install dependencies
pip install -r requirements.txt

# Setup Hetionet
python3 scripts/setup_hetionet.py
```

**Create Visualizations:**
```bash
# Interactive visualization
python3 scripts/create_viz.py <drug_name>

# Filtered by relationship type
python3 scripts/viz_by_relationship.py <drug_name> treats
```

**Explore the Graph:**
```bash
# Search for drugs
python3 scripts/list_drugs.py --search <term>

# Interactive exploration
python3 scripts/view_knowledge_graph.py interactive
```
