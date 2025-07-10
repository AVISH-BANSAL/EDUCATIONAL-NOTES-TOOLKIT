import os
import math
import shutil
import fitz  # PyMuPDF
import cv2
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, request, redirect, url_for
from PIL import Image
from datetime import datetime
from transformers import pipeline, TrOCRProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from spellchecker import SpellChecker

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ------------- Summarizer -------------
def summarize_pdf(pdf_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summary = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summary += result[0]['summary_text'] + "\n\n"
    return summary.strip()

# ------------- Handwritten OCR -------------
def handwritten_ocr_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=300,poppler_path=r"C:\Users\mukes\Downloads\Summer training folder\poppler-24.08.0\Library\bin")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    spell = SpellChecker()
    all_text = ""
    for i, img in enumerate(images):
        img_path = f"{STATIC_FOLDER}/page_{i+1}.jpg"
        img.save(img_path, 'JPEG')
        gray = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(img_path, binary)
        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        words = text.split()
        corrected = " ".join([spell.correction(w) or w for w in words])
        all_text += f"--- Page {i+1} ---\n{corrected}\n\n"
    return all_text.strip()

# ------------- Concept Extractor by Font -------------
def concept_font_extractor(pdf_path):
    doc = fitz.open(pdf_path)
    extracted = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            extracted.append({"text": text, "font_size": span["size"]})
    font_sizes = sorted(set(item['font_size'] for item in extracted), reverse=True)
    if len(font_sizes) < 2:
        return "Not enough font variation to extract topics.", [], []
    main_font, sub_font = font_sizes[:2]
    topics = []
    current_topic = None
    for item in extracted:
        if item['font_size'] == main_font:
            current_topic = item['text']
            topics.append((current_topic, []))
        elif item['font_size'] == sub_font and current_topic:
            topics[-1][1].append(item['text'])
    if not topics:
        return "No topics found.", [], []

    qualities = []
    model = DecisionTreeClassifier()
    train = pd.DataFrame([
        {'length': 5, 'has_example': 0, 'quality': 'Poor'},
        {'length': 12, 'has_example': 0, 'quality': 'Moderate'},
        {'length': 18, 'has_example': 1, 'quality': 'Good'},
        {'length': 30, 'has_example': 1, 'quality': 'Excellent'},
        {'length': 45, 'has_example': 0, 'quality': 'Good'},
        {'length': 50, 'has_example': 1, 'quality': 'Excellent'}
    ])
    X = train[['length', 'has_example']]
    y = LabelEncoder().fit_transform(train['quality'])
    model.fit(X, y)
    le = LabelEncoder().fit(train['quality'])
    for topic, subs in topics:
        wc = sum(len(s.split()) for s in subs)
        has_example = int(any('e.g.' in s or 'example' in s for s in subs))
        pred = model.predict([[wc, has_example]])[0]
        qualities.append(le.inverse_transform([pred])[0])
    images = []
    chunk_size = 5
    for i in range(0, len(topics), chunk_size):
        G = nx.DiGraph()
        chunk = topics[i:i+chunk_size]
        for topic, subs in chunk:
            G.add_node(topic, color='skyblue')
            for sub in subs:
                G.add_node(sub, color='lightgreen')
                G.add_edge(topic, sub)
        pos = nx.spring_layout(G, k=0.6)
        plt.figure(figsize=(10, 7))
        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        nx.draw(G, pos, node_color=node_colors, with_labels=True, arrows=True, node_size=1500, font_size=8)
        filename = f"concept_map_font_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i//chunk_size + 1}.png"
        filepath = os.path.join(STATIC_FOLDER, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        images.append(filename)
    csv_file = f"study_guide_font_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame([{'Topic': t, 'Subtopics': ', '.join(s), 'Note Quality': q}
                  for (t, s), q in zip(topics, qualities)]).to_csv(os.path.join(STATIC_FOLDER, csv_file), index=False)
    return csv_file, images, None

# ------------- Concept Extractor by Symbol -------------
def concept_symbol_extractor(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    lines = full_text.split('\n')
    topics = []
    current_topic = None
    for line in lines:
        clean = line.strip()
        if clean.endswith(':') or clean.endswith(' -'):
            current_topic = clean.rstrip(':').rstrip('-').strip()
            topics.append((current_topic, []))
        elif clean and current_topic:
            topics[-1][1].append(clean)
    if not topics:
        return None, [], "No topics found using symbols."
    qualities = []
    model = DecisionTreeClassifier()
    train = pd.DataFrame([
        {'length': 5, 'has_example': 0, 'quality': 'Poor'},
        {'length': 12, 'has_example': 0, 'quality': 'Moderate'},
        {'length': 18, 'has_example': 1, 'quality': 'Good'},
        {'length': 30, 'has_example': 1, 'quality': 'Excellent'},
        {'length': 45, 'has_example': 0, 'quality': 'Good'},
        {'length': 50, 'has_example': 1, 'quality': 'Excellent'}
    ])
    X = train[['length', 'has_example']]
    y = LabelEncoder().fit_transform(train['quality'])
    model.fit(X, y)
    le = LabelEncoder().fit(train['quality'])
    for topic, subs in topics:
        wc = sum(len(s.split()) for s in subs)
        has_example = int(any('e.g.' in s or 'example' in s for s in subs))
        pred = model.predict([[wc, has_example]])[0]
        qualities.append(le.inverse_transform([pred])[0])
    images = []
    chunk_size = 5
    for i in range(0, len(topics), chunk_size):
        G = nx.DiGraph()
        chunk = topics[i:i+chunk_size]
        for topic, subs in chunk:
            G.add_node(topic, color='skyblue')
            for sub in subs:
                G.add_node(sub, color='lightgreen')
                G.add_edge(topic, sub)
        pos = nx.spring_layout(G, k=0.6)
        plt.figure(figsize=(10, 7))
        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        nx.draw(G, pos, node_color=node_colors, with_labels=True, arrows=True, node_size=1500, font_size=8)
        filename = f"concept_map_symbol_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i//chunk_size + 1}.png"
        filepath = os.path.join(STATIC_FOLDER, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        images.append(filename)
    csv_file = f"study_guide_symbol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame([{'Topic': t, 'Subtopics': ', '.join(s), 'Note Quality': q}
                  for (t, s), q in zip(topics, qualities)]).to_csv(os.path.join(STATIC_FOLDER, csv_file), index=False)
    return csv_file, images, None

# ------------- Flask Routes -------------

# ... same imports and functions ...

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Educational Notes Toolkit</title>
    <style>
        body {
            background-image: url('https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?auto=format&fit=crop&w=1500&q=80');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            margin: 0;
            padding: 0;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        form {
            background: white;
            padding: 50px 60px;
            border-radius: 60px;
            width: 60vw;  /* Reduced Width */
            height:30vw;
            box-shadow: 0 0 30px rgba(0,0,0,0.4);
            text-align: center;
            font-size: 20rem;
        }
        h2 {
            font-size: 12rem;
            margin-bottom: 30px;
            color: #2e8b57;
            text-shadow: 0 0 6px #a1d99b;
        }
        input[type="file"] {
            margin-bottom: 30px;
            font-size: 5rem;
            cursor: pointer;
        }
        label {
            display: block;
            margin-bottom: 15px;
            cursor: pointer;
            font-size: 5rem;
        }
        input[type="radio"] {
            margin-right: 15px;
            transform: scale(1.5);
            vertical-align: middle;
            cursor: pointer;
        }
        button {
            background-color: #2e8b57;
            border: none;
            color: white;
            padding: 18px 50px;
            font-size: 5rem;
            font-weight: bold;
            border-radius: 12px;
            cursor: pointer;
            transition: background 0.3s ease;
            margin-top: 35px;
        }
        button:hover {
            background-color: #59b45f;
        }
    </style>
    </head>
    <body>
    <form method="POST" enctype="multipart/form-data" action="/process">
        <h2>
        
         Educational Notes Toolkit</h2>
        <input type="file" name="pdf" required><br>
        <label><input type="radio" name="option" value="summarizer" required> Summarizer</label>
        <label><input type="radio" name="option" value="ocr"> Handwritten OCR</label>
        <label><input type="radio" name="option" value="concept_font"> Concept Extractor by Font Size</label>
        <label><input type="radio" name="option" value="concept_symbol"> Concept Extractor by Symbols ('-' or ':')</label>
        <button type="submit">Process</button>
    </form>
    </body>
    </html>
    ''')

@app.route('/process', methods=['POST'])
def process():
    file = request.files.get('pdf')
    option = request.form.get('option')
    if not file or not option:
        return redirect('/')

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    if option == 'summarizer':
        summary = summarize_pdf(filepath)
        return render_template_string('''
            <body style="background:black; color:white; font-family:sans-serif; padding:50px; text-align:center;">
                <h2 style="font-size:3.5rem;">Summarized Notes:</h2>
                <pre style="font-size:2rem;">{{ summary }}</pre>
                <a href="/" style="font-size:1.8rem;">← Back</a>
            </body>
        ''', summary=summary)

    elif option == 'ocr':
        ocr_text = handwritten_ocr_pdf(filepath)

        # Save OCR output to a text file
        ocr_filename = f"ocr_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        ocr_filepath = os.path.join(STATIC_FOLDER, ocr_filename)
        with open(ocr_filepath, "w", encoding="utf-8") as f:
            f.write(ocr_text)

        return render_template_string('''
            <body style="background:black; color:white; font-family:sans-serif; padding:50px; text-align:center;">
                <h2 style="font-size:3.5rem;">Extracted Handwritten Text:</h2>
                <pre style="font-size:2rem;">{{ ocr_text }}</pre>
                <a href="/static/images/{{ ocr_file }}" download style="font-size:2rem; color:#2e8b57;">📄 Download OCR Text (.txt)</a><br><br>
                <a href="/" style="font-size:1.8rem;">← Back</a>
            </body>
        ''', ocr_text=ocr_text, ocr_file=ocr_filename)


    elif option == 'concept_font':
        csv_file, images, error = concept_font_extractor(filepath)
        if error:
            return f"<h3 style='font-size:2rem; color:white;'>{error}</h3><a href='/' style='font-size:1.8rem; color:#2e8b57;'>← Back</a>"
        img_tags = ''.join(f'<img src="/static/images/{img}" width="600" style="margin:20px auto; display:block;">' for img in images)
        return render_template_string('''
            <body style="background:black; color:white; font-family:sans-serif; padding:50px; text-align:center;">
                <h2 style="font-size:3.5rem;">Concept Maps & CSV (Font-based):</h2>
                {{ imgs|safe }}
                <a href="/static/images/{{ csv }}" download style="font-size:2rem; color:#2e8b57;">Download Study Guide CSV</a><br><br>
                <a href="/" style="font-size:1.8rem;">← Back</a>
            </body>
        ''', imgs=img_tags, csv=csv_file)

    elif option == 'concept_symbol':
        csv_file, images, error = concept_symbol_extractor(filepath)
        if error:
            return f"<h3 style='font-size:2rem; color:white;'>{error}</h3><a href='/' style='font-size:1.8rem; color:#2e8b57;'>← Back</a>"
        img_tags = ''.join(f'<img src="/static/images/{img}" width="600" style="margin:20px auto; display:block;">' for img in images)
        return render_template_string('''
            <body style="background:black; color:white; font-family:sans-serif; padding:50px; text-align:center;">
                <h2 style="font-size:3.5rem;">Concept Maps & CSV (Symbol-based):</h2>
                {{ imgs|safe }}
                <a href="/static/images/{{ csv }}" download style="font-size:2rem; color:#2e8b57;">Download Study Guide CSV</a><br><br>
                <a href="/" style="font-size:1.8rem;">← Back</a>
            </body>
        ''', imgs=img_tags, csv=csv_file)

    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)