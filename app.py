import os
import asyncio
import torch
from flask import Flask, request, render_template, jsonify
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = Flask(__name__)
loop = asyncio.get_event_loop()

retriever = None

# === Check for GPU ===
if torch.cuda.is_available():
    device = 0
    print("✅ CUDA available — using GPU.")
else:
    device = -1
    print("⚠️ CUDA not available — using CPU.")

# === Load Transformer Model (T5) ===
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=512
)
llm = HuggingFacePipeline(pipeline=pipe)

# === Local Embedding Model ===
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def extract_text_from_pdf(file_stream):
    reader = PdfReader(file_stream)
    return " ".join(page.extract_text() for page in reader.pages if page.extract_text())

async def process_pdf_text(text):
    global retriever

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    vector_store = await asyncio.to_thread(
        lambda: Chroma.from_documents(
            documents,
            embedding_function,
            persist_directory="./chroma_store"
        )
    )
    retriever = vector_store.as_retriever()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['pdf']
    if not file or not file.filename.endswith('.pdf'):
        return jsonify({"error": "Please upload a valid PDF file."}), 400

    async def handle_upload():
        text = await asyncio.to_thread(extract_text_from_pdf, file)
        await process_pdf_text(text)

    loop.run_until_complete(handle_upload())
    return jsonify({"message": "PDF uploaded and processed successfully!"})


@app.route('/chat', methods=['POST'])
def chat():
    question = request.json.get("message", "")
    if not retriever:
        return jsonify({"response": "Please upload a PDF first."})

    async def get_response():
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        return await asyncio.to_thread(qa_chain.run, question)

    answer = loop.run_until_complete(get_response())
    return jsonify({"response": answer})


if __name__ == '__main__':
    app.run(debug=True)
