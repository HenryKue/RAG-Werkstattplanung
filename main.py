from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import ollama

from sentence_transformers import SentenceTransformer

import gradio as gr

import pinecone
from pinecone import Pinecone

from qdrant_client import QdrantClient

import openai


qClient = QdrantClient(host="localhost", port=6333)

transModel = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

pinecone_client = Pinecone(api_key="pcsk_nZmD6_QScSNNTFfbzFQdvQ2Eq9DKPgLb6eadHtQ2WfGa5AfJ6dq3JxcEUbMx84ckWgFda")
index_name="audi-test"
index = pinecone_client.Index(index_name)

client = openai.OpenAI(api_key="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")

def embedding_model(text):
  passage = "query: " + text
  return transModel.encode(passage, normalize_embeddings=True).tolist()

def retrieve_faq(query_embedding, index, top_k=1):
    results = qClient.search(
        collection_name="werkstattplaner_senTran",
        query_vector=query_embedding,
        limit=2  # Anzahl der ähnlichen Treffer
    )

    resultString =""
    debugString = ""
    counter = 1
    for i in results:
       resultString += str(counter) + ". Auszug aus dem Dokument: " + i.payload.get("volltext", "") + "\n\n"
       debugString += str(counter) + ". Auszug aus dem Dokument: " + i.payload.get("volltext", "") + "\n\n" + i.payload.get("datei", "") + "\n\n" + str(i.payload.get("seiten", "")) +"\n\n"
       counter += 1

    print("**************************************************************************************")
    print(debugString)
    print("**************************************************************************************")
    return resultString

def prompt_builder(system_message, context):
  return system_message['content'].format(context)

system_prompt = {
                    "role": "system",
                    "content": """
                    Du bist ein technischer Assistent für Werkstattplanung in einer Autowerkstatt.
                    Erstelle eine Schritt-für-Schritt-Anleitung für den folgenden Prozess.
                    Verwende ausschließlich die Daten aus dem bereitgestellten Kontext. Gib keine Informationen aus, die nicht im Kontext stehen.
                    Wiederhole nicht den Prompt.
                    Achte darauf grammatikalisch korrekte Sätze zu schreiben.
                    Wenn der gegebene Kontext keine Informationen zu der Frage liefert Antworte mit:
                    "Das weiß ich leider noch nicht." 

                    Kontext: 
                    {}
                    """,
                }


def ragChatbot(query):
    query_embedding = embedding_model(query)
    best_match = retrieve_faq(query_embedding,index)
    augmented_prompt = prompt_builder(system_prompt,best_match)

    messages = [{"role": "system","content": augmented_prompt},
                {"role": "user","content": query}]
    
    response = ollama.chat(model="openhermes",messages=messages)


    return (response["message"]["content"])
    
  
demo = gr.Interface(fn=ragChatbot,
                    inputs=[gr.Text(label="Query")],
                    outputs=[gr.Text(label="Answer"),],)

demo.launch()
