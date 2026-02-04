from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        print(chunk)
    return chunks

def create_local_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        task_type="retrieval_document"
    )
    # Only index the first chunk
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("vector_store")

def get_conversation_chain():
    prompt_template = """
    Answer question based on the following context, and if you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context : \n {context} \n
    Question: \n{question} \n
    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    prompt = PromptTemplate(template = prompt_template, input_variables=["context", "question"])
    # Use the supported 'stuff' chain type (single-pass QA over provided docs)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_response(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        task_type="retrieval_document"
    )
    # We created this local FAISS index ourselves, so it's safe to allow pickle-based deserialization.
    vector_store = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    docs = vector_store.similarity_search(user_question)
    chain = get_conversation_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question})
    # Extract the output text from the response
    output_text = response.get('output_text', str(response))
    print(output_text)
    return response


def main():
    pdf_path = 'budget_speech.pdf'
    text = get_pdf_text(pdf_path)
    
    # # Write text to a file
    # output_file = 'budget_speech.txt'
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     f.write(text)
    
    chunks = get_chunks(text)
    create_local_vector_store(chunks)
    get_response("What are the highlights for Attracting global business and investment in the bugget")

if __name__ == '__main__':
    main()