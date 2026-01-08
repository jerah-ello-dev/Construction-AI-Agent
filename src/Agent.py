import os
import sys
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class NotebookLMAgent:
    def __init__(self):
        # Define paths clearly
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # If Agent.py is in src/, go up one level to find PDF folder
        self.project_root = os.path.dirname(self.base_dir) if os.path.basename(self.base_dir) == 'src' else self.base_dir
        self.pdf_folder_path = os.path.join(self.project_root, "data")
        self.db_persist_path = os.path.join(self.project_root, "chroma_db")
        
        self.chat_history = [] 
        
        print("\n" + "="*50)
        print("Initializing Local NotebookLM Agent")
        print("="*50)
        
        # 1. Load Embedding Model
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Check for Ollama
        print("Connecting to Llama 3.1...")
        try:
            self.llm = ChatOllama(
                model="llama3.1", 
                temperature=0.3, # Keeps answers grounded in fact
                keep_alive="1h"  # Keeps model loaded for speed
            )
        except Exception as e:
            print(f"Error: Make sure Ollama is running and you pulled llama3.1")
            sys.exit(1)
            
        # 3. Initialize Database
        self.db = self._initialize_database()
        self.retriever = self.db.as_retriever(search_kwargs={"k": 5})

        # 4. Define the Brain (Prompt)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful, smart research assistant.
            Use the following pieces of retrieved context to answer the question.
            If the answer is not in the context, say "I cannot find that information in the documents."
            
            Context:
            {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        print("\nAgent Ready! (Type 'quit' to exit)")

    def _initialize_database(self):
        # Create PDF folder if it doesn't exist
        if not os.path.exists(self.pdf_folder_path):
            os.makedirs(self.pdf_folder_path)
            print(f"Created folder: {self.pdf_folder_path}")
            print(f"Please put your PDF files in there and run this script again.")
            sys.exit()

        # Check if DB exists and is not empty
        if os.path.exists(self.db_persist_path) and os.listdir(self.db_persist_path):
            print("Loading existing vector database...")
            return Chroma(persist_directory=self.db_persist_path, embedding_function=self.embedding_model)
        
        # If no DB, build it
        print("Building new vector database...")
        return self._create_vector_db()

    def _create_vector_db(self):
        pdf_files = [f for f in os.listdir(self.pdf_folder_path) if f.endswith(".pdf")]
        
        if not pdf_files:
            print(f"No PDF files found in: {self.pdf_folder_path}")
            print("Please add some PDFs and try again.")
            sys.exit()
            
        docs = []
        print(f"Processing {len(pdf_files)} PDF files...")
        for file in pdf_files:
            file_path = os.path.join(self.pdf_folder_path, file)
            print(f"   - Loading {file}...")
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        print(f"Split into {len(chunks)} chunks. Indexing...")
        db = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embedding_model, 
            persist_directory=self.db_persist_path
        )
        return db

    def chat(self):
        chain = self.prompt_template | self.llm
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input: continue
                if user_input.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break
                
                # Retrieve relevant docs
                docs = self.retriever.invoke(user_input)
                context_text = "\n\n".join([d.page_content for d in docs])
                
                # Generate Answer
                print("Thinking...")
                response = chain.invoke({
                    "context": context_text,
                    "chat_history": self.chat_history,
                    "input": user_input
                })
                
                print(f"Agent: {response.content}")
                
                # Update Memory
                self.chat_history.append(HumanMessage(content=user_input))
                self.chat_history.append(AIMessage(content=response.content))
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break

if __name__ == "__main__":
    agent = NotebookLMAgent()
    agent.chat()