import matplotlib.pyplot as plt
import networkx as nx

# Example 1: Batch-level Graph
G_batch = nx.DiGraph()
G_batch.add_edges_from([("Event1", "Event2"), ("Event2", "Event3"), ("Event1", "Event4")])

plt.figure(figsize=(5, 4))
pos = nx.spring_layout(G_batch, seed=42)
nx.draw(G_batch, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=800)
plt.title("Batch-level Example Graph\n(one log file → one graph)")
plt.show()

# Example 2: Entity-level Graph
G_entity = nx.DiGraph()
G_entity.add_edges_from([("ProcessA", "FileX"), ("ProcessA", "ProcessB"), ("FileX", "ProcessB"), ("ProcessB", "Socket1")])
node_colors = ["red" if node == "ProcessB" else "lightgreen" for node in G_entity.nodes()]

plt.figure(figsize=(5, 4))
pos = nx.spring_layout(G_entity, seed=24)
nx.draw(G_entity, pos, with_labels=True, node_color=node_colors, edge_color="gray", node_size=800)
plt.title("Entity-level Example Graph\n(highlight malicious node)")
plt.show()
