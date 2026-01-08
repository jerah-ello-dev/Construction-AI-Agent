import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class NotebookLM_Agent:
    def __init__(self, pdf_folder_path):
        self.pdf_folder_path = pdf_folder_path
        self.db_persist_path = "./chroma_db"
        self.chat_history = []  # Stores conversation memory
        
        print("🚀 Initializing NotebookLM Style Agent...")
        
        # 1. Load Embedding Model (Keep this, it's excellent and free)
        print("📊 Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Initialize Vector DB
        self.db = self._initialize_database()
        self.retriever = self.db.as_retriever(search_kwargs={"k": 5}) # Fetch more context (k=5)
        
        # 3. Load LLM (The Upgrade: Llama 3 via Ollama)
        print("🤖 Connecting to Llama 3 (via Ollama)...")
        try:
            self.llm = ChatOllama(
                model="llama3.1", 
                temperature=0.3, # Low temperature for factual accuracy
                keep_alive="1h"  # Keep model loaded for faster chat
            )
        except Exception as e:
            print(f"❌ Error: Could not connect to Ollama. Make sure Ollama is running and you have pulled llama3.1")
            sys.exit(1)
            
        # 4. Define the Prompt Template (Gives the AI its "NotebookLM" personality)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a smart research assistant akin to NotebookLM. 
            You answer questions strictly based on the provided context. 
            If the answer is not in the documents, say 'I cannot find that information in the documents.'
            
            Context:
            {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        print("✅ Agent ready! Type 'quit' to exit.\n")
    
    def _initialize_database(self):
        if os.path.exists(self.db_persist_path) and os.listdir(self.db_persist_path):
            print("📂 Loading existing vector database...")
            return Chroma(persist_directory=self.db_persist_path, embedding_function=self.embedding_model)
        
        print("📁 Creating new vector database from PDFs...")
        return self._create_vector_db()
    
    def _create_vector_db(self):
        if not os.path.exists(self.pdf_folder_path):
            os.makedirs(self.pdf_folder_path)
            print(f"⚠️ Created folder '{self.pdf_folder_path}'. Please put your PDFs here and restart.")
            sys.exit()

        pdf_files = [f for f in os.listdir(self.pdf_folder_path) if f.endswith(".pdf")]
        if not pdf_files:
            print(f"⚠️ No PDFs found in '{self.pdf_folder_path}'. Add files and restart.")
            sys.exit()
            
        docs = []
        print(f"📚 Processing {len(pdf_files)} files...")
        for file in pdf_files:
            loader = PyPDFLoader(os.path.join(self.pdf_folder_path, file))
            docs.extend(loader.load())
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        db = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embedding_model, 
            persist_directory=self.db_persist_path
        )
        return db

    def chat(self):
        chain = self.prompt_template | self.llm
        
        print("💬 Chat started. Ask me anything about your documents!")
        print("="*50)
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("👋 Goodbye!")
                break
            
            # 1. Retrieve relevant chunks
            docs = self.retriever.invoke(user_input)
            context_text = "\n\n".join([d.page_content for d in docs])
            
            # 2. Generate Response with History
            response = chain.invoke({
                "context": context_text,
                "chat_history": self.chat_history,
                "input": user_input
            })
            
            print(f"🤖 Agent: {response.content}")
            
            # 3. Update Memory
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=response.content))

# Make the script portable
if __name__ == "__main__":
    # Uses a relative path so it works on anyone's computer
    DEFAULT_PDF_FOLDER = os.path.join(os.getcwd(), "PDF")
    
    agent = NotebookLM_Agent(DEFAULT_PDF_FOLDER)
    agent.chat()