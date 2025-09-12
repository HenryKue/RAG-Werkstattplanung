from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.exceptions import ResponseHandlingException

client = QdrantClient(host="localhost", port=6333, timeout=20.0)

name = "werkstattplaner_senTran"

if not client.collection_exists(name):
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        )
        print("Collection erfolgreich angelegt.")
    except ResponseHandlingException as e:
        print("Warnung: Timeout - vermutlich wurde die Collection trotzdem angelegt.")
else:
    print(f"Collection: {name} already exists.")