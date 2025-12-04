import ollama
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

TF_ENABLE_ONEDNN_OPTS = 0


qClient = QdrantClient(host="localhost", port=6333)
transModel = SentenceTransformer("intfloat/multilingual-e5-large-instruct")


def embeddingModel(text):
  passage = "query: " + text
  return transModel.encode(passage, normalize_embeddings=True).tolist()

def retrieveDatabase(query_embedding,k):
    results = qClient.search(
        collection_name="werkstattplaner_senTran",
        query_vector=query_embedding,
        limit=k
    )

    resultString =""
    debugString = ""
    counter = 1
    for i in results:
       resultString += str(counter) + " Kontextauszug: " + i.payload.get("volltext", "") + "\n\n"
       debugString += str(counter) + " Kontextauszug: " + i.payload.get("volltext", "") + "\n\n" + i.payload.get("datei", "") + "\n\n" + str(i.payload.get("seiten", "")) +"\n\n"
       counter += 1

    print("**************************************************************************************")
    print(debugString)
    print("**************************************************************************************")
    return resultString

def retrieveImages(query_embedding,k):
    result = qClient.search(
        collection_name="werkstattplaner_senTran",
        query_vector=query_embedding,
        limit=k
    )

    return result

def prompt_builder(system_message, context):
  return system_message['content'].format(context)

system_prompt = {
  "role": "system",
  "content": """
Du bist ein technischer Assistent für Werkstattpersonal in einer Autowerkstatt. Deine Aufgabe ist es, auf Basis des bereitgestellten Kontexts präzise Schritt-für-Schritt-Anleitungen für Reparaturen, Bauteilaustausch oder sonstige Arbeitsvorgänge zu erstellen.

**Verbindliche Regeln:**
1. Verwende **ausschließlich Informationen aus dem bereitgestellten Kontext**. Ziehe **kein externes Wissen** oder vortrainiertes Wissen hinzu.
2. **Ignoriere irrelevante Passagen**. Nutze nur Inhalte, die **klar und direkt zur gestellten Aufgabe passen**.

4. Gib die Anleitung als **nummerierte Liste** mit **präzisen, handlungsorientierten Arbeitsschritten** aus - ohne Einleitungen, Erklärungen oder Wiederholungen.
5. Verwende **technisch korrekte Fachsprache** und **grammatikalisch saubere Sätze**. Deine Zielgruppe ist **geschultes Werkstattpersonal**, keine Laien.

Dein Ziel ist eine **direkt umsetzbare, werkstatttaugliche Anleitung** - kurz, klar und eindeutig.

Kontext:
{}
"""
}

def ragChatbot(query):
    query_embedding = embeddingModel(query)
    best_match = retrieveDatabase(query_embedding,3)
    augmented_prompt = prompt_builder(system_prompt,best_match)

    messages = [{"role": "system","content": augmented_prompt},
                {"role": "user","content": query}]
    
    response = ollama.chat(model="openhermes",messages=messages,stream=False)
    images = getImages(response["message"]["content"])
    answerImg = {
       "answer": response["message"]["content"],
       "images": images
    }
    return answerImg


def getImages(answer):
   embedding = embeddingModel(answer)
   entry = retrieveImages(embedding,1)
   return entry[0].payload.get("bilder", "")

def test():
   return "Hello"

app = FastAPI()

#CORS
origins = [
    "http://localhost:5173",        # Vite dev server
    "http://127.0.0.1:5173",
    "http://192.168.178.33:5000"    #Production Server    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            
    allow_credentials=True,
    allow_methods=["*"],              
    allow_headers=["*"],              
)

class Query(BaseModel):
   query: str = None
   

@app.get("/")
def root():
   return {"Hello": "World"}

@app.post("/query")
def queryPosted(query: Query):
   answer =  ragChatbot(query.query)
   return {
            "Answer": answer["answer"],
            "Images": answer["images"]
           }
