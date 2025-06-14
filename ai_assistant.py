import os
import fitz  # PyMuPDF
#from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import shutil
import stat

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

shutil.rmtree("faiss_index", onerror=remove_readonly)

# 1. Load PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return " ".join([page.get_text() for page in doc])

# 2. Split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

# 3. Embed and store using FAISS
def create_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model='gemma:2b')
    return FAISS.from_documents(chunks, embeddings)

# 4. Setup QA chain
def setup_qa_chain(vstore):
    retriever = vstore.as_retriever()
    llm = Ollama(model='gemma:2b')
    #return RetrievalQA.from_chain_type(
    #    llm=llm,
    #    retriever=retriever,
     #   return_source_documents=False  # ✅ This shows the actual PDF content used
    #)
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Answer the following question thoughtfully:\n\n{question}"
    )
    return LLMChain(llm=llm, prompt=prompt)

# 5. Main logic
def main():
    pdf_path = "C:\\Users\\mayan\\OneDrive\\Desktop\\Assistant\\assis_trial.pdf"  # Replace with your PDF file path
    text = extract_text_from_pdf(pdf_path)

    # Check if FAISS index already exists
    if os.path.exists("faiss_index/index.faiss"):
        print("🔁 Loading existing FAISS index...")
        embeddings = OllamaEmbeddings(model='gemma:2b')
        vstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        print("⚙️ Creating FAISS index for the first time...")
        chunks = split_text(text)
        vstore = create_vectorstore(chunks)
        vstore.save_local("faiss_index")

    # ✅ Always setup the QA chain after vstore is defined
    qa = setup_qa_chain()#vstore)

    print("\n✅ PDF loaded. You can now ask questions. Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break
        answer = qa.run({"question": query})
        print("Assistant:", answer)


if __name__ == "__main__":
    main()
