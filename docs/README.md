# Documentation

Complete documentation for the Adaptive Knowledge Source Selection System.

## Setup Guides

Get started quickly with these setup guides:

- [Setup Guide](setup/SETUP_GUIDE.md) - Complete installation and setup instructions
- [Hetionet Quick Start](setup/HETIONET_QUICK_START.md) - Quick start guide for Hetionet knowledge graph

## Visualization Guides

Learn how to visualize and explore the knowledge graph:

- [Visualization Guide](visualization/VISUALIZATION_GUIDE.md) - Complete visualization guide with all methods
- [Quick Viz Guide](visualization/QUICK_VIZ_GUIDE.md) - Quick reference for creating interactive visualizations

## Reference Documentation

Technical references and resources:

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
