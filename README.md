# PDF-Analyzer
# PDF Question Answering using k-NN and TF-IDF

## Overview

This project demonstrates a Python script for extracting text from a PDF document and using **TF-IDF** and **k-Nearest Neighbors (k-NN)** to answer user questions by retrieving the most relevant section from the document. It is designed to efficiently handle textual information from PDFs and provide concise responses based on user queries.

---

## Features

1. **PDF Text Extraction**: Extracts all text content from a PDF using `PyMuPDF` (`fitz` library).
2. **Text Segmentation**: Splits the extracted text into sections based on paragraph breaks.
3. **TF-IDF Vectorization**: Converts text into numerical vectors for comparison.
4. **k-NN Search**: Finds the most relevant section for a user-provided query using cosine similarity.

---

## Dependencies

Ensure you have the following Python libraries installed:
- `PyMuPDF` (`fitz`) for PDF text extraction.
- `scikit-learn` for TF-IDF vectorization and k-NN.
- `re` for regular expression-based text segmentation.

Install the required libraries using:
```bash
pip install pymupdf scikit-learn
```

---

## Usage

### 1. Prepare Your PDF
Place your PDF document in the same directory as the script. Update the `pdf_path` variable with the name of your PDF file:
```python
pdf_path = "ASE.pdf"
```

### 2. Run the Script
Run the Python script:
```bash
python pdf_qa.py
```

### 3. Enter Your Question
After the script processes the PDF, you will be prompted to enter a question. The script will retrieve the most relevant section of the text based on your query:
```bash
Please enter your question: What is software engineering?
```

The output will display:
- The relevant section index.
- The content of the most relevant section.

---

## Example Output

```plaintext
Number of sections: 25

Section 0 preview:
Software Engineering is a systematic approach to the design, development, maintenance, and retirement of software.
========================================

Shape of section vectors matrix: (25, 500)

Please enter your question: What is software engineering?

Distances: [[0.153]]
Indices: [[0]]
Relevant Section Index: 0

Answer:
Software Engineering is a systematic approach to the design, development, maintenance, and retirement of software.
```

---

## File Structure

- `pdf_qa.py`: The main script for PDF text extraction and question answering.
- `ASE.pdf`: Example PDF document (replace with your own PDF).

---

## Future Improvements

1. **Enhanced Preprocessing**: Implement advanced text cleaning and noise reduction.
2. **Support for Multi-Section Answers**: Extend functionality to retrieve multiple relevant sections.
3. **Interactive UI**: Add a graphical user interface for better usability.
