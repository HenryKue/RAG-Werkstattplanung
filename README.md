Ein lokal ausführbares Retrieval-Augmented-Generation-System zur automatisierten Generierung fahrzeugspezifischer Arbeitsanweisungen. Entwickelt im Rahmen meiner Bachelorarbeit an der FH Dortmund in Kooperation mit der Reinhardt Automobile GmbH.

🚀 Überblick

Dieses Projekt demonstriert eine komplette lokale RAG-Architektur, die aus OEM-Dokumenten technische Arbeitsanweisungen generiert.
Alle Komponenten laufen lokal und erfüllen damit Datenschutzanforderungen der Dokumente.



🧱 Architektur
User → Frontend → FastAPI → Embedding → Qdrant → LLM → Antwort + Bildreferenzen

📦 Komponenten

FastAPI – Middleware zur Steuerung des RAG-Prozesses

Qdrant – Vektordatenbank für semantische Suche

Ollama – lokales LLM (OpenHermes/Mistral 7B)

Docker – Container für Qdrant

Python – Datenpipeline, Embeddings, RAG-Logik

🔍 Features

Vollständig lokal ausführbar

Dokumentvorverarbeitung & Chunking

Embeddings & semantische Suche

Kontextbasierte Generierung technischer Arbeitsanweisungen

Matching von Text und Bildern

Evaluationspipeline (Inhalt, Sprache, Bildkonsistenz)

📄 Beispiel (Dummy-Daten)

Eingabe:
„Wie tausche ich die Batterie bei Modell A?“

Ausgabe:
(… Beispiel generierter Text auf Basis von Dummy-Chunks …)

🧪 Evaluation

Das System wurde in zwei Stufen getestet:

mit lokalem LLM (OpenHermes/Mistral 7B)

als Vergleich mit ChatGPT-4 (cloudbasiertes LLM)

Ergebnis:
Die Architektur funktioniert vollständig; die Antwortqualität wird primär vom LLM limitiert.

⚠️ Hinweis

Aus Datenschutzgründen enthält dieses Repository keine Dokumente für den Betrieb.
Der produktive Datensatz (OEM-Dokumente) wurde nicht hochgeladen.

📬 Kontakt

Henry Küfner
LinkedIn: [https://www.linkedin.com/in/henry-kuefner/](https://www.linkedin.com/in/henry-kuefner/)
