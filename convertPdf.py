import fitz  # PyMuPDF
from markdownify import markdownify as md
import os

dataPath = "pdf/"
outputPath = "data/"

for file in os.listdir(dataPath):
    if file.endswith(".pdf"):
        # PDF laden
        filePath = os.path.join(dataPath, file)
        doc = fitz.open(filePath)
        all_text = ""

        # Seitenweise extrahieren
        for page in doc:
            text = page.get_text()
            all_text += text + "\n\n"

        # In Markdown umwandeln
        md_text = md(all_text)

        # Ausgabe-Dateiname vorbereiten
        fileName = os.path.splitext(file)[0] + ".md"
        outputFile = os.path.join(outputPath, fileName)

        # In Datei speichern
        with open(outputFile, "w", encoding="utf-8") as f:
            f.write(md_text)
        print("File created:\t" + outputFile)