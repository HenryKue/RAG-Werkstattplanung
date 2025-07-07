import fitz  # PyMuPDF
import pandas as pd

def extract_toc_to_excel(pdf_path, output_excel_path):
    # PDF öffnen
    doc = fitz.open(pdf_path)
    
    # Inhaltsverzeichnis extrahieren
    toc = doc.get_toc()

    if not toc:
        print("Das PDF enthält kein Inhaltsverzeichnis.")
        return

    # Liste zur Speicherung der Einträge
    toc_entries = []

    for entry in toc:
        level, title, page = entry
        toc_entries.append({
            "Kapitel": title,
            "Seite": page,
            "Level": level
        })

    # In DataFrame umwandeln
    df = pd.DataFrame(toc_entries)

    # In Excel-Datei speichern
    df.to_excel(output_excel_path, index=False)
    print(f"Inhaltsverzeichnis erfolgreich gespeichert in: {output_excel_path}")

# Beispielaufruf
pdf_datei = "pdfTest\A005A403100-Systembeschreibung_-_Licht_innen.pdf"              # <-- Hier deinen PDF-Pfad eintragen
excel_datei = "inhaltsverzeichnis.xlsx" # <-- Ziel-Dateiname

extract_toc_to_excel(pdf_datei, excel_datei)
