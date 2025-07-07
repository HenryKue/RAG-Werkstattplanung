import pymupdf as pymu
import os
import ollama
import json
import pandas as pd

path = "pdfTest"
outputFile = "modelResults.json"

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


results = {}

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

        results[file] = parsed2

        print(parsed2)


with open(outputFile, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


excelData = []

for filename, data in results.items():
    excelData.append({
        "Dateiname": filename,
        "Modelle": ", ".join(data.get("modelle", [])),
        "Spezifikationen": ", ".join(data.get("spezifikationen", []))
    })

# DataFrame erstellen
df = pd.DataFrame(excelData)

# In Excel schreiben
df.to_excel("ModellSpezifikationen.xlsx", index=False, engine="openpyxl")
print("✅ Daten erfolgreich in 'ModellSpezifikationen.xlsx' gespeichert.")