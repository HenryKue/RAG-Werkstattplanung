import pymupdf as pymu
import os
import re
import hashlib

ignoreImages = {
    "4659504500d4ee1c7d950526c6edf620",
    "5dcada4ca7e6ddc93c59a2ce6cd168e6"
}

def extractDocumentData(pdfPath, imageDir, chunkLength,file):
    doc = pymu.open(pdfPath)
    docChunks = []
    file = file[:-4]

    #test


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
                            "page": pageNumber + 1,
                            "text": currentChunk,
                            "images": currentImages
                        })      
                        currentImages = []
                        currentChunk = ""

        if currentChunk.strip() and len(currentChunk) < (chunkLength / 2):
            docChunks.append({
                "page": pageNumber + 1,
                "text": currentChunk.strip(),
                "images": currentImages
            })

    docChunks = createRealChunks(docChunks)

    return docChunks

def createRealChunks(halfedChunks):
    fullChunks = []
    for i, chunk in enumerate(halfedChunks):
        if i  < len(halfedChunks) -1:
            pages = [chunk["page"]] if isinstance(chunk["page"], int) else chunk["page"]
            nextPages = [halfedChunks[i+1]["page"]] if isinstance(halfedChunks[i+1]["page"], int) else halfedChunks[i+1]["page"]
            fullChunks.append({
                "page": pages + nextPages,
                "text": chunk["text"] + halfedChunks[i+1]["text"],
                "images": chunk["images"] + halfedChunks[i+1]["images"]
            })
        else:
            fullChunks.append(chunk)

    return fullChunks



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

def extractDocumentChapter(pdfPath, imageDir, file):
    doc = pymu.open(pdfPath)
    docChunks = []
    file = file[:-4]

    toc = doc.get_toc()

    # Kapitel auf Ebene 2
    chapters = [entry for entry in toc if entry[0] == 2]
    
    for i, (level, title, start_page) in enumerate(chapters):
        end_page = chapters[i + 1][2] - 1 if i + 1 < len(chapters) else len(doc)

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

        docChunks.append({
            "page": chapterPages,
            "text": chapterText.strip(),
            "images": chapterImagePaths
        })

    return docChunks


def startExtraction(pdfPath, imageDir, chunkLength,file):
    if validateFirstChapter(pdfPath):
        extractDocumentChapter(pdfPath, imageDir,file)
    else:
        extractDocumentData(pdfPath, imageDir, chunkLength,file)

returnedChunks = startExtraction("pdf/001KAH00000-Karosserie-Instandsetzung__Karosserie-Montagearbeiten.pdf", "imagesTest",5000,"001KAH00000-Karosserie-Instandsetzung__Karosserie-Montagearbeiten.pdf")
#returnedChunks = extractDocumentData("pdf/001KAH00000-Karosserie-Instandsetzung__Karosserie-Montagearbeiten.pdf", "imagesTest", 5000,"001KAH00000-Karosserie-Instandsetzung__Karosserie-Montagearbeiten.pdf")
