import networkx as nx
import spacy

def create_graph_from_text(text):
    # Load SpaCy model for NER and dependency parsing
    nlp = spacy.load("en_core_web_sm")
    
    # Process the text
    doc = nlp(text)
    
    # Create a graph
    G = nx.Graph()
    
    # Add entities as nodes
    for ent in doc.ents:
        G.add_node(ent.text, type=ent.label_)
    
    # Add dependencies as edges
    for token in doc:
        if token.dep_ != 'punct':
            G.add_edge(token.head.text, token.text, type=token.dep_)
    
    return G

# Example usage
text = "SpaCy is an open-source library for advanced Natural Language Processing in Python."
graph = create_graph_from_text(text)

print("Nodes:", graph.nodes())
print("Edges:", graph.edges())

# You can now use this graph with your NLP model
# For example, you might generate node embeddings:
# node_embeddings = node2vec.Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200).fit()
