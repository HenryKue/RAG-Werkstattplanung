import pinecone
from pinecone import Pinecone

import openai

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import fitz  # PyMuPDF
from markdownify import markdownify as md

pinecone_client = Pinecone(api_key="pcsk_nZmD6_QScSNNTFfbzFQdvQ2Eq9DKPgLb6eadHtQ2WfGa5AfJ6dq3JxcEUbMx84ckWgFda")
index_name="audi-test"
index = pinecone_client.Index(index_name)

client = openai.OpenAI(api_key="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")

DATA_PATH = "data/"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=1250,
    length_function=len,
    add_start_index=True
)

documents = load_documents()

chunks = text_splitter.split_documents(documents)

def embedding_model(query, openai_client, model="text-embedding-3-small"):
  # Getting the embedding
  response = openai_client.embeddings.create(
      model=model,
      input=query
  )

  embedding = response.data[0].embedding
  return embedding

data_to_upsert = []

for i, chunk in enumerate(chunks):

    #if (i >= 50 & i <= 55):
    #   print(f"\n--- Chunk {i + 1} ---\n")
    #    print(chunk.page_content)

    data_to_upsert.append(
        {
            "id": str(i),
            "values": embedding_model(chunk.page_content, client),
            "metadata":{"content": chunk.page_content}
        }
    )
    
index.upsert(data_to_upsert, namespace="ns1")
print(f"Uploaded {len(chunks)} Embeddings to Pinecone!")

    