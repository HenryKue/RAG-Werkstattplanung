import pymupdf as pymu
import os
import ollama
import json
import pandas as pd

path = "pdfMotor"
outputFile = "modelResults.json"

jsonTemplate =  """{
    "modelle": [],
    "spezifikationen": []
  } """

system_prompt = {
    "role": "system",
    "content": """
Du bist ein Assistent zur Analyse von technischen Reparaturdokumenten aus der Automobilbranche.

In diesen Dokumenten sind häufig zu Beginn Fahrzeugmodelle und technische Spezifikationen aufgeführt - oft als Liste und nicht in vollständigen Sätzen.

Deine Aufgabe ist:
1. **Alle genannten Fahrzeugmodelle zu extrahieren**, inklusive Marke, Modellbezeichnung und (falls vorhanden) Modelljahr.
2. **Alle technischen Spezifikationen** zu extrahieren, z.B. Getriebearten, Antriebsarten, Baujahre, Motorvarianten usw.

Wenn du im Text die Zeichenfolge "EA" gefolgt von einer Zahl findest, gehe davon aus das es sich um Motorspezifikationen handelt.
Übernimm den Motor für die Spezifikationen.

Wenn du Wörter wie "Zyl", "Motor", "antrieb", oder "getriebe" findest gehe davon aus das es sich um Motorspezifikationen handelt.
Übernimm den Motor für die Spezifikationen.

Gib deine Antwort **ausschließlich im folgenden JSON-Format** zurück:

{{
  "modelle": [],
  "spezifikationen": []
}}

Wenn keine spezifischen Modelle genannt sind, gib:
{{ "modelle": ["alle"], "spezifikationen": [] }}

In "modelle" und "spezifikationen" sind nur Strings erlaubt.
Keine Kommentare, keine Fließtexte, keine Wiederholung des Inputs. Nur gültiges JSON.

Dokument (Auszug):
{}
"""
}

system_prompt = {
    "role": "system",
    "content": """
Du bist ein Assistent zur Analyse von technischen Reparaturdokumenten aus der Automobilbranche.

In diesen Dokumenten sind häufig zu Beginn Fahrzeugmodelle und technische Spezifikationen aufgeführt - oft als Liste und nicht in vollständigen Sätzen.

Deine Aufgabe ist:
1. **Alle genannten Fahrzeugmodelle zu extrahieren**, inklusive Marke, Modellbezeichnung und (falls vorhanden) Modelljahr.
2. **Alle technischen Spezifikationen** zu extrahieren, z.B. Getriebearten, Antriebsarten, Baujahre, Motorvarianten usw.

Wenn du im Text die Zeichenfolge "EA" gefolgt von einer Zahl findest, gehe davon aus das es sich um Motorspezifikationen handelt.
Übernimm den Motor für die Spezifikationen.

Wenn du Wörter wie "Zyl", "Motor", "antrieb", oder "getriebe" findest gehe davon aus das es sich um Motorspezifikationen handelt.
Übernimm den Motor für die Spezifikationen.

Gib deine Antwort **ausschließlich im folgenden JSON-Format** zurück:

{{
  "modelle": [**Modelle hier einfügen**],
  "spezifikationen": [**Spezifikationen hier einfügen**]
}}

Wenn du eine Antwort findest gib die Daten in einem JSON Objekt zurück.
Verwende in dem Objekt nur Strings für die Werte.
Keine Kommentare, keine Fließtexte, keine Wiederholung des Inputs. Nur gültiges JSON.

Dokument (Auszug):
{}
"""
}

fallback_prompt = {
    "role": "system",
    "content": """
Du bist ein technischer Assistent für die Analyse von Reparaturdokumenten aus der Automobilbranche.

Ignoriere alle Modellnamen. Konzentriere dich **nur auf technische Spezifikationen**, die im Dokument erwähnt werden. Dazu gehören z. B.:

- Getriebearten (z.B. „6 Gang-Schaltgetriebe 0B1“, „Automatikgetriebe“)
- Antriebsarten („Frontantrieb“, „Quattro“, „Allrad“)
- Motorisierungen („2.0 TDI“, „1.8 TFSI“, „Diesel“, „Benzin“)
- Baujahre und Modellzeiträume („ab 2008“, „bis 2014“, „Modelljahr 2009“)
- sonstige technische Eigenschaften, wenn eindeutig klassifizierend

Wenn du im Text die Zeichenfolge "EA" gefolgt von einer Zahl findest, gehe davon aus das es sich um Motorspezifikationen handelt.
Übernimm den Motor für die Spezifikationen.

Wenn du Wörter wie "Zyl.", "Motor", "antrieb", oder "getriebe" findest gehe davon aus das es sich um Motorspezifikationen handelt.
Übernimm den Motor für die Spezifikationen.

Antwortformat ausschließlich im JSON-Format:

{{
  "spezifikationen": []
}}

In "spezifikationen" ist nur Strings erlaubt.
Benutze für die Spezifikationen keine Sätze, nur technische Spezifikationen die relevant sind.
Keine Kommentare, keine Wiederholung, kein Fließtext.

Dokument (Auszug):
{}
"""
}

testPrompt = {
    "role": "system",
    "content": """
    Du bist ein technisches System zur Erkennung von Motortypen. Durchsuche das gegebene Dokument um die Fragen zu beantworten.
    Antworte mit den Motortypen oder Antriebsarten wenn du welche findest. 
    
    Nenne nicht das Fahrzeugmodell.
    Deine Antwort darf keine Sätze enthalten.

    Wenn du eine Antwort findest gib die Daten in einem JSON Objekt zurück.
    Verwende in dem Objekt nur Strings.

    {{
        "spezifikationen": [**Antwort hier einfügen**]
    }}

    Dokument:
    {}
"""
}

autoPrompt = {
    "role": "system",
    "content": """
    Du bist ein technisches System zur Erkennung von Fahrzeugtypen. Durchsuche das gegebene Dokument um die Fragen zu beantworten.
    Antworte mit den Fahrzeugtypen wenn du welche findest. 
    
    Nenne nicht dden Motortyp oder die Antriebsart.
    Deine Antwort darf keine Sätze enthalten.

    Wenn du eine Antwort findest gib die Daten in einem JSON Objekt zurück.
    Verwende in dem Objekt nur Strings.

    {{
        "modelle": [**Antwort hier einfügen**]
    }}

    Dokument:
    {}
"""
}


results = {}

def prompt_builder(system_message, context):
  return system_message['content'].format(context)

def runModel2(modelMessages):
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
                    print(f"☝☝☝☝☝☝")
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

def runModel(modelMessages):
    modelResponse = ollama.chat(model="openhermes",messages=modelMessages)
    responseContent = modelResponse["message"]["content"]
    try:
        responseParsed = json.loads(responseContent)
    except Exception as parse_err:
            print(f"⚠️ JSON-Fehler bei Datei {file}: {parse_err}")
            return runModel(modelMessages)
    

    return modelResponse


for file in os.listdir(path):
    if file.endswith(".pdf"):
        pdfPath = os.path.join(path, file)
        print("******************************************************************************")
        print(pdfPath)

        doc = pymu.open(pdfPath)
        text = ""
        for page in doc[:2]:  # Nur die ersten zwei Seiten als Kontext
                text += page.get_text()

        augmented_prompt = prompt_builder(system_prompt,text[:3000])

        messages = [{"role": "system","content": augmented_prompt},
                {"role": "user","content": "Für welche Modelle gilt dieses Dokument?"}]
        
        #Automodell
        auto_prompt = prompt_builder(autoPrompt,text[:3000])
        autoMessages = [{"role": "system","content": auto_prompt},
                {"role": "user","content": "Welche Autmodelle findest du?"}]
        autoResponse = ollama.chat(model="openhermes",messages=autoMessages)
        autoContent = autoResponse["message"]["content"]

        #Spezifikationen
        testAug = prompt_builder(testPrompt,text[:3000])
        fallback_messages = [{"role": "system","content": testAug},
        {"role": "user","content": "Welche Motortypen findest du?"}]
        fallback_response = runModel(fallback_messages)
        fallback_content = fallback_response["message"]["content"]

        parsed2 = json.loads(jsonTemplate)

        data2 = json.loads(autoContent)
        parsed2["modelle"] = data2.get("modelle", [])

        data2 = json.loads(fallback_content)
        parsed2["spezifikationen"] = data2.get("spezifikationen", [])

        #print(autoContent)

        
        response = runModel2(messages)
        content = response["message"]["content"]

        try:
            parsed = json.loads(content)

            if not parsed.get("spezifikationen"):
                #fallback_augmented = fallback_prompt["content"].format(text[:4000])
                #fallback_messages = [
                #    {"role": "system", "content": fallback_augmented},
                #    {"role": "user", "content": "Welche technischen Spezifikationen werden genannt?"}
                #]
                testAug = prompt_builder(testPrompt,text[:3000])
                fallback_messages = [{"role": "system","content": testAug},
                {"role": "user","content": "Welche Motortypen findest du?"}]
                fallback_response = runModel(fallback_messages)
                fallback_content = fallback_response["message"]["content"]
                #print(f"⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪⏪")
                
                fallback_data = json.loads(fallback_content)
                parsed["spezifikationen"] = fallback_data.get("spezifikationen", [])
            

            results[file] = parsed2

            print(parsed2)
        except Exception as parse_err:
            print(f"⛔ JSON-Fehler bei Datei {file}: {parse_err}")
            results[file] = {"error": str(parse_err), "raw": content}


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