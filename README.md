🧠 PDF Question Answering App
This is a Flask web application that allows users to upload a PDF document and interact with it using natural language questions. It utilizes LangChain, HuggingFace Transformers, and a local vector store (Chroma) for semantic search and retrieval-augmented question answering (QA).
🚀 Features
Upload any PDF file and extract text from it.

Chunk and embed the text using sentence-transformers.

Ask questions about the uploaded content.

Retrieve answers via a transformer-based QA pipeline (google/flan-t5-base).

GPU acceleration support (if available via CUDA).
📦 Dependencies
Ensure you have the following installed (see requirements.txt if available):

bash
Copy
Edit
flask
torch
transformers
PyPDF2
langchain
langchain-community
langchain-huggingface
sentence-transformers
chromadb
🛠️ Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/arryc2021/pdf-qa-app.git
cd pdf-qa-app
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
python app.py
🌐 Usage
Navigate to http://127.0.0.1:5000 in your browser.

Upload a PDF document.
📂 Project Structure
.
<pre><code>📁 pdf-qa-app/ ├── 📄 app.py # Main Flask application ├── 📁 templates/ # Folder for HTML templates │ └── 📄 index.html # Web UI for upload and chat ├── 📁 chroma_store/ # Chroma vector store (created at runtime) ├── 📄 requirements.txt # Python dependencies └── 📄 README.md # Project documentation </code></pre>
             # Project documentation

🧪 Example Workflow
Upload a PDF (e.g., research paper, manual).

System extracts and splits the content.

Embeddings are created and stored in a Chroma DB.

User asks a question like "What is the main conclusion?"

The app retrieves the most relevant chunk and generates an answer using FLAN-T5.
⚠️ Notes
Performance: GPU acceleration is used if available.

Security: This app doesn't currently sanitize inputs or enforce upload limits—not suitable for production as-is.

Persistence: Vector DB (chroma_store/) is persisted locally. You can delete this folder to reset stored embeddings.

📄 License
MIT License. See LICENSE for details.

Ask questions related to the content of the PDF.
