import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv




load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment")

os.environ["OPENAI_API_KEY"] = api_key

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def chunk_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

def answer_question(vectorstore, question):
    docs = vectorstore.similarity_search(question, k=3)
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=question)

def main():
    pdf_path = input("Enter the full path to the PDF file: ")
    
    if not os.path.exists(pdf_path):
        print("Invalid PDF path. File not found.")
        return

    print("\nReading and processing PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print("No text could be extracted from the PDF.")
        return

    chunks = chunk_text(text)
    vectorstore = create_vectorstore(chunks)
    print("PDF processed. You can now ask questions!\n")

    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.strip().lower() == "exit":
            print("Goodbye!")
            break
        try:
            answer = answer_question(vectorstore, question)
            print("\nAnswer:", answer, "\n")
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
