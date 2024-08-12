import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re


pdf_path = "ASE.pdf"

#text from the PDF
doc = fitz.open(pdf_path)
text = ""
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    text += page.get_text("text")

# sections
sections = re.split(r'\n{2,}', text)
print(f"Number of sections: {len(sections)}")


for i, section in enumerate(sections[:5]):
    print(f"Section {i} preview:")
    print(section[:200])  #  first 200 characters 
    print("=" * 40)

# S TF-IDF vectors
vectorizer = TfidfVectorizer()
section_vectors = vectorizer.fit_transform(sections)
print(f"Shape of section vectors matrix: {section_vectors.shape}")

# Train the k-NN model
knn = NearestNeighbors(n_neighbors=1, metric='cosine')
knn.fit(section_vectors)


question = input("Please enter your question: ")
question_vector = vectorizer.transform([question])
print("Question vector shape:", question_vector.shape)

distances, indices = knn.kneighbors(question_vector)
print(f"Distances: {distances}")
print(f"Indices: {indices}")

relevant_section_index = indices[0][0]
print(f"Relevant Section Index: {relevant_section_index}")


answer = sections[relevant_section_index]
print("Answer:")
print(answer)
