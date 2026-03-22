# How the Knowledge Graph is Built

## Graph Structure

### What is a Graph?
A **graph** is a data structure with:
- **Nodes** (vertices): Entities like drugs, diseases, genes
- **Edges** (relationships): Connections between entities

```
     Drug A ----treats----> Disease X
       |
       |--causes--> Side Effect Y
       |
       |--binds--> Gene Z
```

### Our Implementation

We use **NetworkX** (Python library) to build a **MultiDiGraph**:

```python
import networkx as nx

# MultiDiGraph = Multiple Directed Graph
# - Multiple: Can have multiple edges between same nodes
# - Directed: Edges have direction (A→B is different from B→A)

graph = nx.MultiDiGraph()
```

**Why MultiDiGraph?**
- Same drug can have multiple relationships with same entity
- Example: Drug A → Gene B (binds, downregulates, upregulates)
- Direction matters: "Drug treats Disease" ≠ "Disease treats Drug"

---

## How We Load Hetionet

### Step 1: Download JSON
```python
# Hetionet comes as a JSON file
{
  "nodes": [
    {
      "kind": "Compound",
      "identifier": "DB00331",
      "name": "Metformin"
    },
    {
      "kind": "Disease",
      "identifier": "DOID:9351",
      "name": "type 2 diabetes mellitus"
    }
  ],
  "edges": [
    {
      "source_id": ["Compound", "DB00331"],
      "target_id": ["Disease", "DOID:9351"],
      "kind": "treats"
    }
  ]
}
```

### Step 2: Parse and Create Graph
```python
# In knowledge_graph.py:

for node in data['nodes']:
    # Create unique ID: "Type::Identifier"
    node_id = f"{node['kind']}::{node['identifier']}"

    # Add to graph with attributes
    graph.add_node(
        node_id,
        name=node['name'],
        type=node['kind']
    )

for edge in data['edges']:
    source_kind, source_id = edge['source_id']
    target_kind, target_id = edge['target_id']

    # Create full IDs
    source = f"{source_kind}::{source_id}"
    target = f"{target_kind}::{target_id}"

    # Add edge with relationship type
    graph.add_edge(
        source,
        target,
        relation=edge['kind']
    )
```

### Step 3: Result
```
Graph Structure:
- Nodes: 47,031 entities
- Edges: 2,250,197 relationships
- Stored in memory as NetworkX graph
- Can query, traverse, visualize
```

---

## Internal Data Structure

### Nodes Storage
```python
# NetworkX stores nodes as:
{
    'Compound::DB00331': {
        'name': 'Metformin',
        'type': 'Compound',
        'identifier': 'DB00331'
    },
    'Disease::DOID:9351': {
        'name': 'type 2 diabetes mellitus',
        'type': 'Disease',
        'identifier': 'DOID:9351'
    }
}
```

### Edges Storage
```python
# NetworkX stores edges as adjacency list:
{
    'Compound::DB00331': {
        'Disease::DOID:9351': {
            0: {'relation': 'treats'}
        },
        'Side Effect::C0014335': {
            0: {'relation': 'causes'}
        },
        'Gene::5563': {
            0: {'relation': 'binds'},
            1: {'relation': 'downregulates'}  # Multiple edges!
        }
    }
}
```

---

## Query Operations

### 1. Get Neighbors
```python
# What is Metformin connected to?
neighbors = graph.neighbors('Compound::DB00331')
# Returns: iterator of all connected nodes
```

### 2. Get Edges with Relationship
```python
# What does Metformin treat?
for source, target, data in graph.out_edges('Compound::DB00331', data=True):
    if data['relation'] == 'treats':
        print(f"Treats: {target}")
```

### 3. Find Paths
```python
# How are Drug A and Disease X connected?
paths = nx.all_simple_paths(graph, source, target, cutoff=3)
# Returns: list of paths (lists of nodes)
```

---

## Why This Structure is Powerful

### 1. Fast Queries
- O(1) node lookup: `graph.nodes['Compound::DB00331']`
- O(1) edge lookup: `graph['source']['target']`
- Efficient traversal algorithms built-in

### 2. Flexible Relationships
- Store any metadata on nodes/edges
- Multiple relationship types
- Bidirectional queries (in/out edges)

### 3. Rich Algorithms
NetworkX provides:
- Shortest paths
- Centrality measures
- Community detection
- Graph traversal
- Subgraph extraction

---

## Memory Usage

```
Hetionet in memory:
- JSON file: 712 MB on disk
- NetworkX graph: ~1.5 GB in RAM
- Pickle file: ~400 MB on disk (compressed)

Why pickle is faster:
- JSON: Parse 712 MB, build graph (2 min)
- Pickle: Load directly into memory (3 sec)
```

---

## Summary

**What we built:**
```
Hetionet Knowledge Graph
    ↓
NetworkX MultiDiGraph
    ↓
47K nodes + 2.25M edges
    ↓
In-memory structure for fast queries
```

**Key features:**
- ✅ Directed relationships (A→B ≠ B→A)
- ✅ Multiple edges (A→B can have multiple types)
- ✅ Rich metadata (names, types, properties)
- ✅ Fast queries (O(1) lookups)
- ✅ Built-in graph algorithms
