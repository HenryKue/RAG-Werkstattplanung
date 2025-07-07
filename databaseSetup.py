from pinecone import Pinecone

from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient

from sentence_transformers import SentenceTransformer

import openai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


import pymupdf as pymu
import os
import ollama
import json
import hashlib
import re


transModel = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

pinecone_client = Pinecone(api_key="pcsk_nZmD6_QScSNNTFfbzFQdvQ2Eq9DKPgLb6eadHtQ2WfGa5AfJ6dq3JxcEUbMx84ckWgFda")
index_name="audi-test"
index = pinecone_client.Index(index_name)

client = openai.OpenAI(api_key="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")

qClient = QdrantClient(host="localhost", port=6333)

DATA_PATH = "dataTest/"

imageDir = "images/"

path = "pdfTest"
outputFile = "modelResults.json"

globalIndex = 0

ignoreImages = {
    "4659504500d4ee1c7d950526c6edf620",
    "5dcada4ca7e6ddc93c59a2ce6cd168e6"
}


jsonTemplate =  """{
    "modelle": [],
    "spezifikationen": []
  } """


specPrompt = {
    "role": "system",
    "content": """
    Du bist ein technisches System zur Erkennung Spezifikationen. Durchsuche das gegebene Dokument um die Fragen zu beantworten.
    Antworte mit den Motortypen, Antriebsarten, oder Spezifikationen die das Dokument behandelt, wenn du welche findest. 
    Denk dir keine Daten aus. Verwende nur was in dem Dokument steht.

    Der Dateiname gibt Hinweise worum es in dem Dokument geht. Beziehe diesen in die Analyse mit ein:

    Dokumentenname: {filename}

    **Ignoriere vollständig alles, was wie ein Fahrzeugmodell aussieht.**
    
    Deine Antwort darf keine Sätze enthalten.
    Alles was mit "Audi" beginnt ist keine Spezifikation!

    Antworte nicht mit Dingen die nicht im Dokument vorkommen.
    "Benzin" oder "Diesel" allein ist keine Spezifikation!

    Wenn du eine Antwort findest gib die Daten in einem JSON Objekt zurück.
    Verwende in dem Objekt nur Strings.

    {{
        "spezifikationen": [**Antwort hier einfügen**]
    }}

    Dokument:
    {document}
"""
}

specRecPrompt = {
    "role": "system",
    "content": """
    Du bist ein technisches System zur Erkennung von Spezifikationen. 
    Das hier ist bereits der zweite Aufruf an dich. Du hast beim ersten Mal in den Spezifikationen den Fahrzeugtyp genannt.
    Führe folgenden Aufgabentypen erneut aus. Achte diesmal explizit darauf keine Fahrzeugmodelle zu nennen. Das Wort "Audi" ist in deiner Antwort verboten!
    
    Durchsuche das gegebene Dokument um die Fragen zu beantworten.
    Antworte mit den Motortypen, Antriebsarten, oder Spezifikationen die das Dokument behandelt, wenn du welche findest. 

    Der Dateiname gibt Hinweise darauf worum es in dem Dokument geht. Beziehe diesen in die Analyse mit ein:

    Dokumentenname: {filename}

    **Ignoriere vollständig alles, was wie ein Fahrzeugmodell aussieht.**
    
    Deine Antwort darf keine Sätze enthalten.
    Alles was mit "Audi" beginnt ist keine Spezifikation!

    Antworte nicht mit Dingen die nicht im Dokument vorkommen.
    "Benzin" oder "Diesel" allein ist keine Spezifikation!

    Wenn du eine Antwort findest gib die Daten in einem JSON Objekt zurück.
    Verwende in dem Objekt nur Strings.

    {{
        "spezifikationen": [**Antwort hier einfügen**]
    }}

    Dokument:
    {document}
"""
}

autoPrompt = {
    "role": "system",
    "content": """
    Du bist ein technisches System zur Erkennung von Fahrzeugtypen. Durchsuche das gegebene Dokument um die Fragen zu beantworten.
    Antworte mit den Fahrzeugtypen wenn du welche findest. 

    Der Dateiname KANN Hinweise darauf liefern worum es in dem Dokument geht und wonach du suchen KÖNNTEST. Beziehe dies in die Analyse mit ein:

    Dokumentenname: {filename}
    
    Nenne nicht den Motortyp oder die Antriebsart.
    Deine Antwort darf keine Sätze enthalten.

    Wenn du eine Antwort findest gib die Daten in einem JSON Objekt zurück.
    Verwende in dem Objekt nur Strings.

    {{
        "modelle": [**Antwort hier einfügen**]
    }}

    Dokument:
    {document}
"""
}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=1250,
    length_function=len,
    add_start_index=True
)

results = {}

def load_documents(file):
    loader = TextLoader(os.path.join(DATA_PATH, file), encoding="utf-8")
    documents = loader.load()
    return documents

def embedding_model(text):
  passage = "passage: " + text
  return transModel.encode(passage, normalize_embeddings=True).tolist()



def prompt_builder(system_message, context, filename):
  return system_message['content'].format(
      filename=filename,
      document=context
  )
def runModel(modelMessages):
    modelResponse = ollama.chat(model="openhermes", messages=modelMessages)
    responseContent = modelResponse["message"]["content"]

    try:
        responseParsed = json.loads(responseContent)

        # Validierung und Normalisierung
        def flatten_list(lst):
            result = []
            for item in lst:
                if isinstance(item, dict):
                    # Nimm den ersten String-Wert im dict
                    for v in item.values():
                        if isinstance(v, str):
                            result.append(v)
                            break
                elif isinstance(item, (str, int, float)):
                    result.append(str(item))
            return result

        # Initialisiere struktur, falls Schlüssel fehlen
        if not isinstance(responseParsed, dict):
            raise ValueError("Antwort ist kein JSON-Objekt")

        modelle = flatten_list(responseParsed.get("modelle", []))
        spezifikationen = flatten_list(responseParsed.get("spezifikationen", []))
        return {
            "message": {
                "content": json.dumps({
                    "modelle": modelle,
                    "spezifikationen": spezifikationen
                }, ensure_ascii=False)
            }
        }

    except Exception as parse_err:
        print(f"⚠️ JSON-Fehler bei Datei {file}: {parse_err}")
        return runModel(modelMessages)

def audiCheck(specs,i,text,filename):
    global spec_messages

    try:
        specs = json.loads(specs)
    except json.JSONDecodeError as e:
        print(f"Fehler beim Parsen von specs: {e}")
        return None

    newSpecs = {
    "modelle": [],
    "spezifikationen": []
                } 
    if "modelle" in specs:
        newSpecs["modelle"].append(specs["modelle"])

    for j in specs.get("spezifikationen", []):
        if not j.strip().startswith("Audi"):
            newSpecs["spezifikationen"].append(j)

    if len(newSpecs["spezifikationen"]) == 0:
        spec_prompt = prompt_builder(specRecPrompt,text,filename) 
        spec_messages = [{"role": "system","content": spec_prompt},
        {"role": "user","content": "Welche Motortypen oder Spezifikationen findest du?"}]
        newResponse = runModel(spec_messages)
        newContent = newResponse["message"]["content"]
        print("Rekursionstiefe: " + str(i))
        i += 1
        print(newContent)
        return audiCheck(newContent,i,text,filename)  # Rekursiv weitergeben
    else:
        return json.dumps(newSpecs)
    
def extractText(path, chunkSize, overlap):
    doc = pymu.open(path)
    fullText = ""
    pageNo = 1

    for page in doc:
        text = page.get_text("text")
        fullText += text + "\n" + "Seite: " + str(pageNo) + "\n"
        pageNo += 1
        

    chunks = []
    startChar = 0
    while startChar < len(fullText):
        end = startChar + chunkSize
        chunk = fullText[startChar:end].strip()
        if chunk:
            chunks.append(chunk)
        startChar = startChar + chunkSize - overlap

    return chunks

def extractDocumentData(pdfPath, imageDir, chunkLength,file):
    doc = pymu.open(pdfPath)
    docChunks = []
    file = file[:-4]


    for pageNumber, page in enumerate(doc):
        dict = page.get_text("dict")["blocks"]
        currentChunk=""
        currentImages=[]

        images = page.get_images(full=True)

        for block in dict:
            blockText = ""
            if block.get("type") == 1:
                if images:
                    img = images.pop(0)
                    xref = img[0]
                    imageFilename = f"{file}_page_{pageNumber+1}_{xref}.png"
                    imagePath = os.path.join(imageDir, imageFilename)
                    pixmap = pymu.Pixmap(doc, xref)
                    
                    imgBytes = pixmap.tobytes("png")
                    imgHash = hashlib.md5(imgBytes).hexdigest()

                    if imgHash not in ignoreImages:
                        if pixmap.n < 5:
                            pixmap.save(imagePath)
                        else:
                            pixmap = pymu.Pixmap(pymu.csRGB, pixmap)
                            pixmap.save(imagePath)
                        currentImages.append(imagePath)
            
            elif "lines" in block:
                for line in block["lines"]:
                    blockText = ""
                    for span in line["spans"]:
                        blockText += span["text"] + " "
                    currentChunk += blockText + "\n"
                    if len(currentChunk) >= (chunkLength/2):
                        docChunks.append({
                            "seiten": pageNumber + 1,
                            "text": currentChunk,
                            "bilder": currentImages
                        })      
                        currentImages = []
                        currentChunk = ""

        if currentChunk.strip() and len(currentChunk) < (chunkLength / 2):
            docChunks.append({
                "seiten": pageNumber + 1,
                "text": currentChunk.strip(),
                "bilder": currentImages
            })

    docChunks = createRealChunks(docChunks)

    return docChunks

def createRealChunks(halfedChunks):
    fullChunks = []
    for i, chunk in enumerate(halfedChunks):
        if i  < len(halfedChunks) -1:
            pages = [chunk["seiten"]] if isinstance(chunk["seiten"], int) else chunk["seiten"]
            nextPages = [halfedChunks[i+1]["seiten"]] if isinstance(halfedChunks[i+1]["seiten"], int) else halfedChunks[i+1]["seiten"]
            fullChunks.append({
                "seiten": pages + nextPages,
                "text": chunk["text"] + halfedChunks[i+1]["text"],
                "bilder": chunk["bilder"] + halfedChunks[i+1]["bilder"],
                "volltext": chunk["text"] + halfedChunks[i+1]["text"]
            })
        else:
            fullChunks.append({
                "seiten": chunk["seiten"],
                "text": chunk["text"],
                "bilder": chunk["bilder"],
                "volltext": chunk["text"]
            })

    return fullChunks

def startExtraction(pdfPath, imageDir, chunkLength,file):
    if validateFirstChapter(pdfPath):
        return extractDocumentChapter(pdfPath, imageDir,file)
    else:
        return extractDocumentData(pdfPath, imageDir, chunkLength,file)

#def extractDocumentChapter(pdfPath, imageDir, file):
#    doc = pymu.open(pdfPath)
#    docChunks = []
#    file = file[:-4]
#
#    toc = doc.get_toc()
#    
#    # Nutze alle TOC-Einträge, sortiert nach Seite
#    toc_sorted = sorted(toc, key=lambda x: x[2])
#
#    for i, (level, title, start_page) in enumerate(toc_sorted):
#        # Finde Ende des Kapitels
#        if i + 1 < len(toc_sorted):
#            end_page = toc_sorted[i + 1][2] - 1
#        else:
#            end_page = len(doc)  # Letztes Kapitel bis Ende des Dokuments
#
#        startId = start_page - 1
#        endId = end_page - 1
#
#        chapterText = ""
#        chapterImagePaths = []
#        chapterPages = []
#
#        for page_num in range(startId, endId + 1):
#            page = doc[page_num]
#            chapterText += page.get_text()
#            images = page.get_images(full=True)
#
#            for img in images:
#                xref = img[0]
#                imageFilename = f"{file}_page_{page_num+1}_{xref}.png"
#                imagePath = os.path.join(imageDir, imageFilename)
#
#                try:
#                    pixmap = pymu.Pixmap(doc, xref)
#                except Exception as e:
#                    print(f"Fehler bei Bild xref {xref} auf Seite {page_num+1}: {e}")
#                    continue
                #
#                imgBytes = pixmap.tobytes("png")
#                imgHash = hashlib.md5(imgBytes).hexdigest()
#
#                if imgHash not in ignoreImages:
#                    ignoreImages.add(imgHash)
#                    if pixmap.n < 5:
#                        pixmap.save(imagePath)
#                    else:
#                        pixmap = pymu.Pixmap(pymu.csRGB, pixmap)
#                        pixmap.save(imagePath)
#
#                    chapterImagePaths.append(imagePath)
#
#            chapterPages.append(page_num + 1)
#
#        docChunks.append({
#            "kapitel": title.strip(),
#            "seiten": chapterPages,
#            "text": chapterText.strip(),
#            "bilder": chapterImagePaths
#        })
#
#    return docChunks

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def extractDocumentChapter(pdfPath, imageDir, file, chunk_size=1000, overlap=200):
    doc = pymu.open(pdfPath)
    docChunks = []
    file = file[:-4]

    toc = doc.get_toc()
    toc_sorted = sorted(toc, key=lambda x: x[2])

    for i, (level, title, start_page) in enumerate(toc_sorted):
        if i + 1 < len(toc_sorted):
            end_page = toc_sorted[i + 1][2] - 1
        else:
            end_page = len(doc)

        startId = start_page - 1
        endId = end_page - 1

        chapterText = ""
        chapterImagePaths = []
        chapterPages = []

        for page_num in range(startId, endId + 1):
            page = doc[page_num]
            chapterText += page.get_text()
            images = page.get_images(full=True)

            for img in images:
                xref = img[0]
                imageFilename = f"{file}_page_{page_num+1}_{xref}.png"
                imagePath = os.path.join(imageDir, imageFilename)

                try:
                    pixmap = pymu.Pixmap(doc, xref)
                except Exception as e:
                    print(f"Fehler bei Bild xref {xref} auf Seite {page_num+1}: {e}")
                    continue

                imgBytes = pixmap.tobytes("png")
                imgHash = hashlib.md5(imgBytes).hexdigest()

                if imgHash not in ignoreImages:
                    ignoreImages.add(imgHash)
                    if pixmap.n < 5:
                        pixmap.save(imagePath)
                    else:
                        pixmap = pymu.Pixmap(pymu.csRGB, pixmap)
                        pixmap.save(imagePath)

                    chapterImagePaths.append(imagePath)

            chapterPages.append(page_num + 1)

        chapterText = chapterText.strip()
        textChunks = chunk_text(chapterText, chunk_size=chunk_size, overlap=overlap)

        for chunk in textChunks:
            docChunks.append({
                "kapitel": title.strip(),
                "seiten": chapterPages,
                "text": chunk.strip(),
                "volltext": chapterText,
                "bilder": chapterImagePaths,
            })

    return docChunks



def validateFirstChapter(pdf_path):
    print(f"\n📄 Datei: {os.path.basename(pdf_path)}")
    try:
        doc = pymu.open(pdf_path)
    except Exception as e:
        print(f"❌ Fehler beim Öffnen: {e}")
        return

    toc = doc.get_toc()
    if not toc:
        print("⚠️ Kein Inhaltsverzeichnis gefunden.")
        return

    # Nur erstes Kapitel auf erster Ebene
    for level, title, toc_page in toc:
        if level == 1:
            actual_page = findActualChapterPage(doc, title, toc_page)
            if actual_page is None:
                print(f"   ❓ Kapitel '{title}' nicht im Text gefunden (TOC: {toc_page})")
                return False
            elif actual_page != toc_page:
                print(f"   ⚠️ Kapitel '{title}' - TOC: {toc_page}, Tatsächlich: {actual_page}")
                return False
            else:
                print(f"   ✅ Kapitel '{title}' beginnt korrekt auf Seite {toc_page}")
                return True
            
def normalize(text):
    """Bereinigt Text für robusteren Vergleich."""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9]', ' ', text)).strip().lower()

def findActualChapterPage(doc, title, toc_page_hint):
    """Sucht nach dem tatsächlichen Vorkommen des Kapiteltitels."""
    norm_title = normalize(title)
    for page_num in range(max(0, toc_page_hint - 1), min(len(doc), toc_page_hint + 5)):
        text = doc[page_num].get_text()
        if norm_title in normalize(text):
            return page_num + 1  # Seitenzahlen beginnen bei 1
    return None

for file in os.listdir(path):
    if file.endswith(".pdf"):
        pdfPath = os.path.join(path, file)
        print("******************************************************************************")
        print(pdfPath)
        doc = pymu.open(pdfPath)
        text = ""
        for page in doc[:2]:  # Nur die ersten zwei Seiten als Kontext
                text += page.get_text()
        
        #Automodell
        auto_prompt = prompt_builder(autoPrompt,text[:3000],file)
        autoMessages = [{"role": "system","content": auto_prompt},
                {"role": "user","content": "Welche Autmodelle findest du?"}]
        autoResponse = runModel(autoMessages)
        autoContent = autoResponse["message"]["content"]

        #Spezifikationen
        spec_prompt = prompt_builder(specPrompt,text[:3000],file)
        spec_messages = [{"role": "system","content": spec_prompt},
        {"role": "user","content": "Welche Motortypen oder Spezifikationen findest du?"}]
        spec_response = runModel(spec_messages)
        spec_content = audiCheck(spec_response["message"]["content"],0,text[:3000],file)

        parsed2 = json.loads(jsonTemplate)

        data2 = json.loads(autoContent)
        parsed2["modelle"] = data2.get("modelle", [])

        data2 = json.loads(spec_content)
        parsed2["spezifikationen"] = data2.get("spezifikationen", [])

        print(parsed2)

        fileName = os.path.splitext(file)[0] + ".md"
        documents = load_documents(fileName)

        imageFileDir = os.path.join(imageDir, file[:-4])
        os.makedirs(imageFileDir, exist_ok=True)

        #chunks = text_splitter.split_documents(documents)
        #chunks = extractText(pdfPath, 2500, 1250)
        chunks = startExtraction(pdfPath,imageFileDir,5000,file)

        points = []

        for i, chunk in enumerate(chunks):
            if chunk["text"].strip() != "":
                vector = embedding_model(chunk["text"])
                payload = {
                    "content": chunk["text"] + "\nDieser Text gilt für die folgenden Modelle: " + ", ".join(parsed2["modelle"]) +
                               "\nWeitere Spezifikationen: " + ", ".join(parsed2["spezifikationen"]),
                    "modelle": parsed2["modelle"],
                    "spezifikationen": parsed2["spezifikationen"],
                    "datei": file,
                    "seiten": chunk["seiten"],
                    "bilder": chunk["bilder"],
                    "volltext": chunk["volltext"]
                }
                points.append(
                    PointStruct(id=globalIndex, vector=vector, payload=payload)
                )

                globalIndex += 1

        # Hilfsfunktion für Batches (bleibt gleich)
        def chunk_list(lst, chunk_size):
            for j in range(0, len(lst), chunk_size):
                yield lst[j:j + chunk_size]

        # Upload in Batches zu Qdrant
        for batch in chunk_list(points, 100):
            qClient.upsert(collection_name="werkstattplaner_senTran", points=batch)

        print(f"Uploaded {len(chunks)} Embeddings to Qdrant!")