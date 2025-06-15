from langchain_groq.chat_models import ChatGroq
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import fitz  # PyMuPDF
import os

# Replace with your Groq API key
GROQ_API_KEY = "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"  # 🔐 Use your actual key securely

# Load LLaMA3 LLM from Groq
llm_shared = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama3-8b-8192"
)

# Prompt used with vector-based QA
custom_pdf_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use ONLY the information below to answer the question.

PDF Context:
{context}

Question: {question}

Answer:"""
)

# Fallback full-text prompt
fallback_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer the question using the following full PDF text as context:

Context:
{context}

Question: {question}

Answer:"""
)


# Extract full text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text() for page in doc])
    print(f"[DEBUG] Extracted PDF text length: {len(text)}")
    return text


# Split into manageable chunks for embedding
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.create_documents([text])
    print(f"[DEBUG] Number of chunks created: {len(chunks)}")
    return chunks


# Create or load FAISS vector store
def create_vectorstore(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)


# Set up RetrievalQA chain
def setup_pdf_qa_chain(vstore, llm):
    retriever = vstore.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": custom_pdf_prompt}
    )


# Set up fallback chain using full text
def setup_fallback_chain(llm, pdf_text):
    chain = LLMChain(llm=llm, prompt=fallback_prompt)
    return chain, pdf_text


def main():
    pdf_path = "C:\\Users\\mayan\\OneDrive\\Desktop\\Assistant\\CSE332.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found at: {pdf_path}")
        return

    # Step 1: Extract and embed PDF content
    text = extract_text_from_pdf(pdf_path)

    embeddings = OllamaEmbeddings(model='nomic-embed-text')

    # Step 2: Load or create vector index
    if os.path.exists("faiss_index/index.faiss"):
        print("[INFO] Loading existing FAISS index...")
        vstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        print("[INFO] Creating new FAISS index from PDF...")
        chunks = split_text(text)
        vstore = create_vectorstore(chunks, embeddings)
        vstore.save_local("faiss_index")

    # Step 3: Set up both primary and fallback QA chains
    pdf_qa = setup_pdf_qa_chain(vstore, llm_shared)
    fallback_qa, full_context = setup_fallback_chain(llm_shared, text)

    print("\n🚀 Groq-based PDF Assistant ready. Ask questions (type 'exit' to quit).\n")

    # Step 4: User loop
    while True:
        query = input("You: ")
        if query.strip().lower() in ["exit", "quit"]:
            print("👋 Exiting. Goodbye!")
            break

        # Try PDF-based QA first
        try:
            answer = pdf_qa.invoke({"query": query})["result"]
        except Exception as e:
            print("⚠️ Error from PDF QA chain:", e)
            answer = ""

        # Detect vague/empty answers and fallback
        vague_phrases = [
            "no answer found", "not in pdf", "i don't know", "context does not", "no relevant information"
        ]
        normalized = answer.strip().lower()

        if not normalized or any(vague in normalized for vague in vague_phrases):
            print("(⚠️ Fallback: PDF context did not help — using full LLM)")
            try:
                answer = fallback_qa.invoke({"context": full_context, "question": query})["text"]
            except Exception as e:
                answer = f"❌ Error from fallback LLM: {str(e)}"

        print("Assistant:", answer)


if __name__ == "__main__":
    main()
