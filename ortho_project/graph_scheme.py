import json
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def get_graph_schema():
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                # Get schema visualization
                result = session.run("CALL db.schema.visualization()")
                graph_data = result.single()

                nodes = {}
                relationships = []

                # Process nodes
                for node in graph_data['nodes']:
                    label = list(node.labels)[0]  # Assuming one label per node
                    nodes[label] = {
                        "properties": [],
                        "relationships": []
                    }

                # Process relationships
                for rel in graph_data['relationships']:
                    start_node = list(rel.start_node.labels)[0]
                    end_node = list(rel.end_node.labels)[0]
                    rel_type = rel.type

                    relationships.append({
                        "start": start_node,
                        "type": rel_type,
                        "end": end_node
                    })

                    nodes[start_node]["relationships"].append({
                        "type": rel_type,
                        "direction": "outgoing",
                        "target": end_node
                    })
                    nodes[end_node]["relationships"].append({
                        "type": rel_type,
                        "direction": "incoming",
                        "source": start_node
                    })

                # Get properties for all node types
                for label in nodes.keys():
                    query = f"""
                    MATCH (n:{label})
                    WITH n LIMIT 1
                    RETURN keys(n) AS properties
                    """
                    result = session.run(query)
                    properties = result.single()['properties']
                    nodes[label]["properties"] = properties

                schema = {
                    "nodes": nodes,
                    "relationships": relationships
                }

                # Convert the schema to a JSON string
                return json.dumps(schema, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})