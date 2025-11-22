# 🚀 **AI Knowledge Base Builder + SBERT Fine-Tuning Pipeline**

A complete end-to-end system for building a high-quality AI knowledge base, extracting data from multiple sources, cleaning and deduplicating text, generating contradiction datasets, computing semantic similarity with SBERT, and fine-tuning a Sentence-BERT model for relevance classification.

This project supports downstream applications such as:
🔹 **RAG-based Chatbots**
🔹 **AI Tutors**
🔹 **Semantic Search Engines**
🔹 **ML Interview Assistants**
🔹 **Knowledge Extractors**

---

# 📘 **Table of Contents**

* [Overview](#overview)
* [Features](#features)
* [Pipeline Architecture](#pipeline-architecture)
* [Data Sources](#data-sources)
* [1. PDF Extraction Pipeline](#1-pdf-extraction-pipeline)
* [2. Wikipedia Extraction Pipeline](#2-wikipedia-extraction)
* [3. Web Scraping (Javatpoint)](#3-web-scraping-javatpoint)
* [4. SBERT Similarity Scoring](#4-sbert-similarity-scoring)
* [5. Contradictory Dataset Generation](#5-contradictory-dataset)
* [6. Fine-Tuning SBERT](#6-fine-tuning-sbert)
* [Model Evaluation](#model-evaluation)
* [Folder Structure](#folder-structure)
* [Installation](#installation)
* [Run the Pipeline](#run-the-pipeline)
* [Future Work](#future-work)

---

# 📌 **Overview**

This repository contains a modular pipeline for building a **clean, augmented, deduplicated machine-learning knowledge dataset**, followed by an **SBERT fine-tuning step** for classifying relevance/accuracy of responses.

The final output is a trained model capable of distinguishing:

✔ **Relevant / correct** Question–Answer pairs
✘ **Contradictory / incorrect** answers

This model is then usable inside **RAG-like pipelines** to validate generative outputs.

---

# 🌟 **Features**

### ✅ **Automated Data Collection**

* Extracts paragraphs from PDFs
* Scrapes Javatpoint tutorials
* Fetches Wikipedia summaries
* Removes headers/footers, noise, duplicate content

### ✅ **Data Cleaning & Preprocessing**

* Page-aware paragraph stitching
* Removes numbering, sections, and irregular formatting
* Restores missing word spacing
* Normalizes punctuation & whitespace

### ✅ **SBERT Similarity Scoring**

* Computes semantic similarity between question & answer pairs
* Filters out weak or irrelevant Q/A pairs

### ✅ **Contradictory Dataset Generation**

* Automatically generates **incorrect answers**
* Uses SBERT to measure semantic dissimilarity
* Produces labeled negative training examples

### ✅ **Fine-Tuning Sentence-BERT**

* Binary relevance classifier
* Uses CosineSimilarityLoss
* Trains on combined positive + negative dataset

### ⭐ **Final Output**

A fine-tuned SBERT model for **semantic relevance classification**, usable in:

✔ LLM Guardrails
✔ RAG Response Validation
✔ Automated Q/A Checking
✔ Semantic Similarity Ranking

---

# 🏗 **Pipeline Architecture**

```
                 ┌──────────────────────────┐
                 │      PDF Extractor       │
                 └─────────────┬────────────┘
                               │
                 ┌──────────────────────────┐
                 │  Wikipedia Summaries     │
                 └─────────────┬────────────┘
                               │
                 ┌──────────────────────────┐
                 │   Web Scraper (JTP)      │
                 └─────────────┬────────────┘
                               │
                     Clean / Deduplicate
                               │
                     Build Master Dataset
                               │
            ┌────────────────────────────────────┐
            │   SBERT Similarity Computation     │
            └────────────────────────────────────┘
                               │
                   Filter Strong/Weak Pairs
                               │
             ┌─────────────────────────────┐
             │ Contradictory Q/A Generator │
             └─────────────────────────────┘
                               │
                      Labeled Dataset
                               │
               ┌────────────────────────────┐
               │   SBERT Fine-Tuning Model  │
               └────────────────────────────┘
                               │
                       Deployment / RAG
```

---

# 📚 **Data Sources**

### ✔ PDF textbooks

Text extracted with:

* `pdfplumber`
* custom header/footer detection
* paragraph continuation logic

### ✔ Wikipedia

Extracted via the `wikipedia` Python library.

### ✔ Javatpoint Tutorials

Full-page scraping of:

* AI
* Machine Learning
* Deep Learning
* Data Science
* NLP
* Statistics
* Big Data
  (over **300+ URLs**)

---

# 📘 **1. PDF Extraction Pipeline**

You implemented:

* Header/footer removal
* Page number cleanup
* Section numbering cleanup
* Normalized paragraphs
* Handling multi-line continuation

Key functions:

### **`remove_header_footer()`**

Removes headers, footers, page numbers, chapter titles.

### **`clean_paragraph()`**

Removes unwanted characters & fixes spacing.

### **`extract_paragraphs_with_continuation()`**

Stitches paragraphs across pages.

This produces:
📄 `extracted_paragraphs.csv`

---

# 🌐 **2. Wikipedia Extraction**

You generated summaries for **~300 ML topics**, filtering out:

* Disambiguation pages
* Missing content
* Errors

Results saved to:
📄 `wikipedia.csv`

---

# 🌍 **3. Web Scraping (Javatpoint)**

You built a robust scraper with:

✔ HTML cleaning
✔ Removing navigation elements
✔ Chunking tutorials by `<h2>` / `<h3>`
✔ Processing lists, code blocks
✔ Combining sections into clean chunks

Saved to:
📄 `javaTpoint.csv`

---

# 🤖 **4. SBERT Similarity Scoring**

You computed semantic similarity for:

* Your main dataset (GPT-generated answers)
* The contradictory dataset

Using:

```python
model = SentenceTransformer('all-MiniLM-L6-v2')
similarity_scores = util.cos_sim(question_embeddings, answer_embeddings).diagonal()
```

Filtered high-quality pairs:

✔ `Similarity > 0.7 → Label = 1` (related)
✔ `Similarity < 0.7 → discarded`

Outputs:

📄 `Updated_File_With_Similarity_Scores_SBERT.csv`
📄 `Filtered_Data.csv`
📄 `Labeled_Data.csv`

---

# ⚠ **5. Contradictory Dataset**

You generated **100 synthetic wrong answers** that deliberately contradict the questions.

SBERT scored them → kept **less similar** ones.

Label assigned:

✔ `Label = 0` for contradictory answers

Outputs:

📄 `Contradictory.csv`
📄 `Contradictory_Similarity_Scores_SBERT.csv`
📄 `Contradictory_Labeled_Data.csv`

---

# 🧠 **6. Fine-Tuning SBERT**

Final dataset:

* Positive examples → Label = 1
* Negative examples → Label = 0

You concatenated:

```
Positive: Labeled_Data.csv  
Negative: Contradictory_Labeled_Data.csv
```

Used **CosineSimilarityLoss**, 3 epochs:

```python
train_loss = losses.CosineSimilarityLoss(model)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=500,
    output_path='./sbert_finetuned_model'
)
```

---

# 📊 **Model Evaluation**

You calculated self-similarity classification:

→ **Final accuracy: 0.67**

This will improve with:

* Larger dataset
* Better negative sampling
* Hard negative mining
* Multi-sentence contrastive training

---

# 📁 **Folder Structure**

```
project/
│
├── data/
│   ├── extracted_paragraphs.csv
│   ├── wikipedia.csv
│   ├── javaTpoint.csv
│   ├── Labeled_Data.csv
│   ├── Contradictory_Labeled_Data.csv
│   ├── master_dataset.csv
│
├── models/
│   └── sbert_finetuned_model/
│
├── pdf_extractor.py
├── wikipedia_extractor.py
├── javatpoint_scraper.py
├── similarity_scoring.py
├── contradictory_generator.py
├── sbert_finetune.py
└── README.md
```

---

# ⚙ **Installation**

```bash
pip install -r requirements.txt
```

Key dependencies:

```
pdfplumber
sentence-transformers
beautifulsoup4
pandas
torch
sklearn
requests
lxml
wikipedia
```

---

# ▶ **Run the Pipeline**

### 1️⃣ Extract PDF

```bash
python pdf_extractor.py
```

### 2️⃣ Scrape Javatpoint

```bash
python javatpoint_scraper.py
```

### 3️⃣ Get Wikipedia Data

```bash
python wikipedia_extractor.py
```

### 4️⃣ Compute Similarity

```bash
python similarity_scoring.py
```

### 5️⃣ Generate Contradictory Examples

```bash
python contradictory_generator.py
```

### 6️⃣ Fine-Tune SBERT

```bash
python sbert_finetune.py
```
