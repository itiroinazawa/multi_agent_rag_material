from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.text_splitter import CharacterTextSplitter

from langchain.schema import Document

import vacation_template 

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define function to create FAISS retriever
def create_retriever(documents, index):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=doc) for doc in text_splitter.split_text(documents)]
    vectorstore = FAISS.from_documents(docs, embedding_model)
    return vectorstore.as_retriever()

# Load and initialize agents' knowledge bases
vacation_data = """Data: John Doe: June 10-20, 2025\nJane Smith: July 5-15, 2025\nJorge Aragao: June 17-27\n..."""
# Create retrievers
vacation_retriever = create_retriever(vacation_data, "vacation")

# Initialize LLM
llm = ChatOllama(model="llama3.1")

vacation_docs_chain = create_stuff_documents_chain(llm, vacation_template.PromptTemplateImpl().generate())

# Define agent chains
vacation_agent = create_retrieval_chain(vacation_retriever, vacation_docs_chain)

# Function to interact with agents
def query_agent(agent, query):
    return agent.invoke({"input": query})

# Example queries
if __name__ == "__main__":
    print("Vacation Info:", query_agent(vacation_agent, "When is John Doe on vacation?"))
    print("Vacation Info:", query_agent(vacation_agent, "When is Jane Smith on vacation?"))
    print("Vacation Info:", query_agent(vacation_agent, "When is Joazinho on vacation?"))
    print("Vacation Info:", query_agent(vacation_agent, "Does any employee have overlapping vacation periods?"))