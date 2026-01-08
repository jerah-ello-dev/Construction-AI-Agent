# Local AI Agent (RAG) for Construction & Documents

A private, local AI agent that lets you "chat" with your PDF documents. Built with **Llama 3.1**, **LangChain**, and **Ollama**.

Unlike simple summary tools, this agent uses **Retrieval Augmented Generation (RAG)** to understand context, retain conversation history (memory), and answer complex questions based strictly on your provided data.

## Features
* **100% Local & Private**: Runs entirely on your machine. No data leaves your computer.
* **Conversational Memory**: Remembers previous questions (e.g., "Summarize this," then "Tell me more about the first point").
* **Smart Retrieval**: Uses Vector Embeddings (ChromaDB) to find the exact paragraph needed to answer your question.
* **Powered by Llama 3.1**: Uses Meta's latest open-source model for human-like reasoning.

## Tech Stack
* **LLM**: Meta Llama 3.1 (via Ollama)
* **Framework**: LangChain
* **Vector Database**: ChromaDB
* **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)

## Prerequisites
1.  **Install Python 3.10+**
2.  **Install Ollama**: Download from [ollama.com](https://ollama.com)
3.  **Pull the Model**: Open your terminal and run:
    ```bash
    ollama pull llama3.1
    ```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/jerah-ello-dev/Construction-AI-Agent.git](https://github.com/jerah-ello-dev/Construction-AI-Agent.git)
    cd Construction-AI-Agent
    ```

2.  **Set up Virtual Environment**:
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Add your Documents**:
    Place your PDF files into the `data/` folder.

2.  **Run the Agent**:
    ```bash
    python src/Agent.py
    ```

3.  **Chat**:
    Type your questions in the terminal. Type `quit` to exit.

## Important Notes

* **Updating Documents**: If you add new PDFs to the `data/` folder, you must **delete the `chroma_db` folder** before running the script again. The agent will detect the missing database and rebuild it with your new files automatically.
    ```powershell
    # Windows Command to clear DB
    Remove-Item -Recurse -Force chroma_db
    ```

## Contributing
Feel free to fork this repository and submit pull requests.

## License
MIT License