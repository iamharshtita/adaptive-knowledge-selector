"""
Simple HTML Interactive Visualization
Creates a basic interactive graph viewer without pyvis dependency issues
"""

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_sources.knowledge_graph import KnowledgeGraphSource


def create_simple_html(kg: KnowledgeGraphSource, drug_name: str, max_neighbors: int = 30):
    """Create a simple D3.js interactive visualization"""

    # Find drug
    matches = kg.search_drug_by_name(drug_name)
    if not matches:
        print(f"❌ Drug '{drug_name}' not found")
        return

    drug_id = matches[0]
    drug_data = kg.graph.nodes[drug_id]

    # Get neighborhood
    neighbors = list(kg.get_neighbors(drug_id))[:max_neighbors]
    nodes_to_include = [drug_id] + neighbors
    subgraph = kg.graph.subgraph(nodes_to_include)

    # Prepare data
    nodes = []
    links = []

    node_id_to_index = {}
    for i, node in enumerate(subgraph.nodes()):
        node_data_dict = kg.graph.nodes[node]
        nodes.append({
            'id': node,
            'name': node_data_dict.get('name', node),
            'type': node_data_dict.get('type', 'unknown'),
            'is_drug': node == drug_id
        })
        node_id_to_index[node] = i

    for u, v, data in subgraph.edges(data=True):
        links.append({
            'source': node_id_to_index[u],
            'target': node_id_to_index[v],
            'relation': data.get('relation', 'unknown')
        })

    # Create HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Knowledge Graph: {drug_data['name']}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: white;
        }}
        #graph {{
            border: 1px solid #444;
            background: #2a2a2a;
        }}
        .node {{
            cursor: pointer;
            stroke: #fff;
            stroke-width: 2px;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
            stroke-width: 2px;
        }}
        .node-label {{
            font-size: 10px;
            pointer-events: none;
            text-shadow: 0 1px 0 #000, 1px 0 0 #000, 0 -1px 0 #000, -1px 0 0 #000;
        }}
        #info {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            max-width: 300px;
        }}
        h2 {{
            margin-top: 0;
            color: #4ECDC4;
        }}
        .legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 15px;
            height: 15px;
            margin-right: 8px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <h1>Knowledge Graph: {drug_data['name']}</h1>
    <p>Showing {len(neighbors)} connections | Drag nodes to explore | Hover for details</p>

    <svg id="graph" width="1400" height="800"></svg>

    <div id="info">
        <h2>Information</h2>
        <div id="info-content">
            <p>Click on a node to see details</p>
        </div>
    </div>

    <div class="legend">
        <h3 style="margin-top:0;margin-bottom:10px;">Node Types</h3>
        <div class="legend-item">
            <div class="legend-color" style="background: #FF6B6B"></div>
            <span>Compound (Drug)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #4ECDC4"></div>
            <span>Disease</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #95E1D3"></div>
            <span>Gene</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #FFD93D"></div>
            <span>Side Effect</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #A8E6CF"></div>
            <span>Other</span>
        </div>
    </div>

    <script>
        const data = {json.dumps({'nodes': nodes, 'links': links})};

        const width = 1400;
        const height = 800;

        const svg = d3.select("#graph");

        const colorMap = {{
            'Compound': '#FF6B6B',
            'Disease': '#4ECDC4',
            'Gene': '#95E1D3',
            'Side Effect': '#FFD93D',
            'Pathway': '#A8E6CF',
            'Symptom': '#FFB6B9',
            'Anatomy': '#C7CEEA',
            'Biological Process': '#B4E7CE',
            'Molecular Function': '#FFE5B4',
            'Cellular Component': '#E0BBE4',
            'Pharmacologic Class': '#FEC8D8'
        }};

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));

        const link = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .enter().append("line")
            .attr("class", "link")
            .attr("marker-end", "url(#arrowhead)");

        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#999");

        const node = svg.append("g")
            .selectAll("circle")
            .data(data.nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => d.is_drug ? 20 : 10)
            .attr("fill", d => colorMap[d.type] || '#CCCCCC')
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("click", clicked)
            .on("mouseover", mouseover)
            .on("mouseout", mouseout);

        const labels = svg.append("g")
            .selectAll("text")
            .data(data.nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .attr("text-anchor", "middle")
            .attr("dy", ".35em")
            .text(d => d.name.length > 20 ? d.name.substring(0, 17) + '...' : d.name);

        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            labels
                .attr("x", d => d.x)
                .attr("y", d => d.y + 25);
        }});

        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}

        function clicked(event, d) {{
            const connections = data.links.filter(l =>
                l.source.id === d.id || l.target.id === d.id
            );

            const info = `
                <h3>${{d.name}}</h3>
                <p><strong>Type:</strong> ${{d.type}}</p>
                <p><strong>ID:</strong> ${{d.id}}</p>
                <p><strong>Connections:</strong> ${{connections.length}}</p>
                <h4>Relationships:</h4>
                <ul>
                    ${{connections.slice(0, 5).map(c => {{
                        const other = c.source.id === d.id ? c.target : c.source;
                        return \`<li>${{c.relation}}: ${{other.name}}</li>\`;
                    }}).join('')}}
                    ${{connections.length > 5 ? \`<li>... and ${{connections.length - 5}} more</li>\` : ''}}
                </ul>
            `;

            document.getElementById('info-content').innerHTML = info;
        }}

        function mouseover(event, d) {{
            d3.select(this)
                .transition()
                .duration(200)
                .attr("r", d => d.is_drug ? 25 : 15);
        }}

        function mouseout(event, d) {{
            d3.select(this)
                .transition()
                .duration(200)
                .attr("r", d => d.is_drug ? 20 : 10);
        }}
    </script>
</body>
</html>"""

    # Save file
    filename = f"visualizations/{drug_name.replace(' ', '_')}_interactive.html"
    os.makedirs('visualizations', exist_ok=True)

    with open(filename, 'w') as f:
        f.write(html)

    abs_path = os.path.abspath(filename)
    print(f"\n✓ Interactive visualization created!")
    print(f"✓ File: {filename}")
    print(f"\n🌐 Open in browser:")
    print(f"   file://{abs_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_html_viz.py <drug_name>")
        print("Example: python simple_html_viz.py metformin")
        sys.exit(1)

    drug_name = ' '.join(sys.argv[1:])

    # Load graph
    pickle_path = "data/hetionet/hetionet_graph.pkl"
    if not os.path.exists(pickle_path):
        print("❌ Knowledge graph not found!")
        print("Run: python scripts/setup_hetionet.py")
        sys.exit(1)

    print("Loading knowledge graph...")
    kg = KnowledgeGraphSource()
    kg.load_graph(pickle_path)

    create_simple_html(kg, drug_name)
