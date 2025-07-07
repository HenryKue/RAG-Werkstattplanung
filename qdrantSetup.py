from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(host="localhost", port=6333)

name = "werkstattplaner_senTran"

client.recreate_collection(
    collection_name=name,
    vectors_config=VectorParams(
        size=1024,                 # ← OpenAI text-embedding-3-small 1536, 384 multi-qa-MiniLM-L6-cos-v1, 768 multi-qa-mpnet-base-dot-v1
        distance=Distance.COSINE
    )
)