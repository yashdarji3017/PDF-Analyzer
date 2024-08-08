import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")  
    return text

#Segment 
def segment_text(text):
    sections = re.split(r'\n{2,}', text)
    return sections

#Convert text to vectors
def convert_text_to_vectors(sections):
    vectorizer = TfidfVectorizer()
    section_vectors = vectorizer.fit_transform(sections)
    return vectorizer, section_vectors

#Train the k-NN model
def train_knn_model(section_vectors):
    knn = NearestNeighbors(n_neighbors=1, metric='cosine')
    knn.fit(section_vectors)
    return knn

#questions 
def find_relevant_section(question, vectorizer, knn):
    question_vector = vectorizer.transform([question])
    distances, indices = knn.kneighbors(question_vector)
    relevant_section_index = indices[0][0]
    return relevant_section_index

#display
def extract_answer(relevant_section_index, sections):
    answer = sections[relevant_section_index]
    return answer



pdf_path = "Temp2.pdf"  
pdf_text = extract_text_from_pdf(pdf_path)

#Segment the text
sections = segment_text(pdf_text)

# Convert text to vectors
vectorizer, section_vectors = convert_text_to_vectors(sections)


knn = train_knn_model(section_vectors)


question = input("Please enter your question: ")


relevant_section_index = find_relevant_section(question, vectorizer, knn)
answer = extract_answer(relevant_section_index, sections)


print("Answer:")
print(answer)
