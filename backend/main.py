import ollama
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
import openai

TF_ENABLE_ONEDNN_OPTS = 0


qClient = QdrantClient(host="localhost", port=6333)
client = openai.OpenAI(api_key="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")
transModel = SentenceTransformer("intfloat/multilingual-e5-large-instruct")


def embedding_model(text):
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
3. Wenn die gestellte Aufgabe mit dem Kontext **nicht beantwortet werden kann**, gib exakt folgenden Satz aus:  
   _"Das weiß ich leider noch nicht."_
4. Gib die Anleitung als **nummerierte Liste** mit **präzisen, handlungsorientierten Arbeitsschritten** aus - ohne Einleitungen, Erklärungen oder Wiederholungen.
5. Verwende **technisch korrekte Fachsprache** und **grammatikalisch saubere Sätze**. Deine Zielgruppe ist **geschultes Werkstattpersonal**, keine Laien.

Dein Ziel ist eine **direkt umsetzbare, werkstatttaugliche Anleitung** - kurz, klar und eindeutig.

Kontext:
{}
"""
}





def ragChatbot(query):
    query_embedding = embedding_model(query)
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
    #return (response["message"]["content"])

#    response = client.chat.completions.create(
#      model="gpt-4.1",
#      messages=messages,
#      max_tokens=1500
#    )
#    images = getImages(response.choices[0].message.content)
#    answerImg = {
#       "answer": responseresponse.choices[0].message.content,
#       "images": images
#    }
#    return answerImg


def getImages(answer):
   embedding = embedding_model(answer)
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
    allow_origins=origins,            # oder ["*"] für Entwicklung (unsicher in Prod)
    allow_credentials=True,
    allow_methods=["*"],              # GET, POST, etc.
    allow_headers=["*"],              # Content-Type, Authorization, etc.
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
