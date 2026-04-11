"""
Create visualizations filtered by relationship type
Shows specific relationships like 'treats', 'causes', etc.
"""

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_sources.knowledge_graph import KnowledgeGraphSource


def create_filtered_viz(kg: KnowledgeGraphSource, drug_name: str,
                       relationship_types: list = None,
                       max_per_type: int = 10):
    """
    Create visualization showing only specific relationship types

    Args:
        kg: Knowledge graph
        drug_name: Drug to visualize
        relationship_types: List of relationship types to show (e.g., ['treats', 'causes'])
        max_per_type: Maximum nodes to show per relationship type
    """

    if relationship_types is None:
        relationship_types = ['treats', 'causes', 'binds', 'downregulates', 'upregulates']

    # Find drug
    matches = kg.search_drug_by_name(drug_name)
    if not matches:
        print(f"❌ Drug '{drug_name}' not found")
        return

    drug_id = matches[0]
    drug_data = kg.graph.nodes[drug_id]

    print(f"\n📊 Creating filtered visualization for: {drug_data['name']}")
    print(f"   Showing relationships: {', '.join(relationship_types)}")

    # Collect nodes by relationship type
    nodes_by_type = {rel: [] for rel in relationship_types}
    edges_list = []

    for _, target, data in kg.graph.out_edges(drug_id, data=True):
        rel = data.get('relation', 'unknown')
        if rel in relationship_types:
            if len(nodes_by_type[rel]) < max_per_type:
                nodes_by_type[rel].append(target)
                edges_list.append((drug_id, target, rel))

    # Count what we found
    print("\n   Found:")
    total_nodes = 0
    for rel, nodes in nodes_by_type.items():
        print(f"      {rel}: {len(nodes)} nodes")
        total_nodes += len(nodes)

    if total_nodes == 0:
        print(f"\n❌ No relationships of types {relationship_types} found for {drug_name}")
        return

    # Build node list (drug + all connected nodes)
    all_nodes = [drug_id] + [n for nodes in nodes_by_type.values() for n in nodes]

    # Create node data
    nodes = []
    node_id_to_index = {}

    for i, node_id in enumerate(all_nodes):
        node_dict = kg.graph.nodes[node_id]
        nodes.append({
            'id': node_id,
            'name': node_dict.get('name', node_id),
            'type': node_dict.get('type', 'unknown'),
            'is_drug': node_id == drug_id
        })
        node_id_to_index[node_id] = i

    # Create edge data
    links = []
    for source, target, relation in edges_list:
        links.append({
            'source': node_id_to_index[source],
            'target': node_id_to_index[target],
            'relation': relation
        })

    # Generate HTML
    graph_data = json.dumps({'nodes': nodes, 'links': links})

    rel_types_str = '_'.join(relationship_types)

    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>''' + drug_data['name'] + ''' - ''' + ', '.join(relationship_types) + '''</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #1a1a1a; color: white; }
        #graph { border: 1px solid #444; background: #2a2a2a; }
        .node { cursor: pointer; stroke: #fff; stroke-width: 2px; }
        .link { stroke-opacity: 0.6; stroke-width: 2px; }
        .node-label { font-size: 10px; pointer-events: none; fill: white;
                     text-shadow: 0 1px 0 #000, 1px 0 0 #000, 0 -1px 0 #000, -1px 0 0 #000; }
        #info { position: fixed; top: 20px; right: 20px; background: rgba(0,0,0,0.8);
               padding: 15px; border-radius: 5px; max-width: 300px; }
        h2 { margin-top: 0; color: #4ECDC4; }
        .legend { position: fixed; bottom: 20px; right: 20px; background: rgba(0,0,0,0.8);
                 padding: 15px; border-radius: 5px; }
        .legend-item { display: flex; align-items: center; margin: 5px 0; }
        .legend-color { width: 15px; height: 15px; margin-right: 8px; border-radius: 50%; }
        .rel-legend { position: fixed; top: 20px; left: 20px; background: rgba(0,0,0,0.8);
                     padding: 15px; border-radius: 5px; max-width: 250px; }
    </style>
</head>
<body>
    <h1>''' + drug_data['name'] + ''' - Filtered View</h1>
    <p>Showing: ''' + ', '.join(relationship_types) + ''' (''' + str(total_nodes) + ''' connections) | Drag nodes | Click for details</p>

    <svg id="graph" width="1400" height="800"></svg>

    <div class="rel-legend">
        <h3 style="margin-top:0;margin-bottom:10px;">Relationship Colors</h3>
        <div class="legend-item"><div class="legend-color" style="background: #2E8B57"></div><span>treats</span></div>
        <div class="legend-item"><div class="legend-color" style="background: #DC143C"></div><span>causes</span></div>
        <div class="legend-item"><div class="legend-color" style="background: #4169E1"></div><span>binds</span></div>
        <div class="legend-item"><div class="legend-color" style="background: #FF8C00"></div><span>downregulates</span></div>
        <div class="legend-item"><div class="legend-color" style="background: #9370DB"></div><span>upregulates</span></div>
    </div>

    <div id="info">
        <h2>Information</h2>
        <div id="info-content"><p>Click on a node to see details</p></div>
    </div>

    <div class="legend">
        <h3 style="margin-top:0;margin-bottom:10px;">Node Types</h3>
        <div class="legend-item"><div class="legend-color" style="background: #FF6B6B"></div><span>Compound (Drug)</span></div>
        <div class="legend-item"><div class="legend-color" style="background: #4ECDC4"></div><span>Disease</span></div>
        <div class="legend-item"><div class="legend-color" style="background: #95E1D3"></div><span>Gene</span></div>
        <div class="legend-item"><div class="legend-color" style="background: #FFD93D"></div><span>Side Effect</span></div>
        <div class="legend-item"><div class="legend-color" style="background: #A8E6CF"></div><span>Other</span></div>
    </div>

    <script>
        const data = ''' + graph_data + ''';
        const width = 1400, height = 800;
        const svg = d3.select("#graph");

        const colorMap = {
            'Compound': '#FF6B6B', 'Disease': '#4ECDC4', 'Gene': '#95E1D3',
            'Side Effect': '#FFD93D', 'Pathway': '#A8E6CF', 'Symptom': '#FFB6B9',
            'Anatomy': '#C7CEEA', 'Biological Process': '#B4E7CE',
            'Molecular Function': '#FFE5B4', 'Cellular Component': '#E0BBE4',
            'Pharmacologic Class': '#FEC8D8'
        };

        const relColorMap = {
            'treats': '#2E8B57', 'causes': '#DC143C', 'binds': '#4169E1',
            'downregulates': '#FF8C00', 'upregulates': '#9370DB'
        };

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).distance(150))
            .force("charge", d3.forceManyBody().strength(-400))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(40));

        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 20).attr("refY", 0)
            .attr("markerWidth", 6).attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#999");

        const link = svg.append("g").selectAll("line").data(data.links).enter().append("line")
            .attr("class", "link")
            .attr("stroke", d => relColorMap[d.relation] || '#999')
            .attr("marker-end", "url(#arrowhead)");

        const node = svg.append("g").selectAll("circle").data(data.nodes).enter().append("circle")
            .attr("class", "node")
            .attr("r", d => d.is_drug ? 25 : 12)
            .attr("fill", d => colorMap[d.type] || '#CCCCCC')
            .call(d3.drag()
                .on("start", (e,d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
                .on("drag", (e,d) => { d.fx=e.x; d.fy=e.y; })
                .on("end", (e,d) => { if (!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }))
            .on("click", (e, d) => {
                const connections = data.links.filter(l => l.source.id === d.id || l.target.id === d.id);
                let html = '<h3>' + d.name + '</h3>';
                html += '<p><strong>Type:</strong> ' + d.type + '</p>';
                html += '<p><strong>Connections:</strong> ' + connections.length + '</p>';
                html += '<h4>Relationships:</h4><ul>';
                connections.slice(0, 10).forEach(c => {
                    const other = c.source.id === d.id ? c.target : c.source;
                    html += '<li>' + c.relation + ': ' + other.name + '</li>';
                });
                if (connections.length > 10) html += '<li>... and ' + (connections.length - 10) + ' more</li>';
                html += '</ul>';
                document.getElementById('info-content').innerHTML = html;
            })
            .on("mouseover", function(e, d) {
                d3.select(this).transition().duration(200).attr("r", d => d.is_drug ? 30 : 17);
            })
            .on("mouseout", function(e, d) {
                d3.select(this).transition().duration(200).attr("r", d => d.is_drug ? 25 : 12);
            });

        const labels = svg.append("g").selectAll("text").data(data.nodes).enter().append("text")
            .attr("class", "node-label").attr("text-anchor", "middle").attr("dy", ".35em")
            .text(d => d.name.length > 20 ? d.name.substring(0, 17) + '...' : d.name);

        simulation.on("tick", () => {
            link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
            node.attr("cx", d => d.x).attr("cy", d => d.y);
            labels.attr("x", d => d.x).attr("y", d => d.y + 30);
        });
    </script>
</body>
</html>'''

    # Save file
    filename = f"visualizations/{drug_name.replace(' ', '_')}_{rel_types_str}.html"
    os.makedirs('visualizations', exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)

    abs_path = os.path.abspath(filename)
    print(f"\n✓ Filtered visualization created!")
    print(f"✓ File: {filename}")
    print(f"\n🌐 Open: open \"{filename}\"")

    return filename


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python viz_by_relationship.py <drug_name> [relationship_types]")
        print("\nExamples:")
        print("  python viz_by_relationship.py atorvastatin treats")
        print("  python viz_by_relationship.py metformin treats,causes")
        print("  python viz_by_relationship.py ibuprofen treats,causes,binds")
        print("\nAvailable relationship types:")
        print("  treats, causes, binds, downregulates, upregulates")
        sys.exit(1)

    drug_name = sys.argv[1]

    # Parse relationship types
    if len(sys.argv) >= 3:
        rel_types = [r.strip() for r in sys.argv[2].split(',')]
    else:
        rel_types = ['treats', 'causes']  # Default to diseases and side effects

    # Load graph
    pickle_path = "data/hetionet/hetionet_filtered.pkl"
    if not os.path.exists(pickle_path):
        print("❌ Knowledge graph not found!")
        print("Run: python scripts/setup_hetionet.py")
        sys.exit(1)

    print("Loading knowledge graph...")
    kg = KnowledgeGraphSource()
    kg.load_graph(pickle_path)

    filename = create_filtered_viz(kg, drug_name, rel_types)

    if filename:
        print(f"\n✨ Now you can clearly see the {', '.join(rel_types)} relationships!")
