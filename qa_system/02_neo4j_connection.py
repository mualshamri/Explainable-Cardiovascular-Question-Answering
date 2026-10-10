import getpass

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    NEO4J_PASSWORD = getpass.getpass("Enter Neo4j password: ")

neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
)
neo4j_driver.verify_connectivity()


try:
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
    )
    LLAMAINDEX_AVAILABLE = True
except Exception:
    graph_store = None
    LLAMAINDEX_AVAILABLE = False

print("Connected to Neo4j.")
print("LlamaIndex supporting layer available:", LLAMAINDEX_AVAILABLE)
