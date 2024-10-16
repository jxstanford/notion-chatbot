import streamlit as st
import openai
import tiktoken
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import organization

# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.organization = st.secrets["OPENAI_ORGANIZATION"]
embedding_model = st.secrets["OPENAI_EMBEDDING_MODEL"]

# OpenAI's pricing for 'text-embedding-ada-002' is $0.050 per 1M tokens as of 2024-10-16
cost_per_1m_tokens = 0.050 # as of 2024-10-16

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def estimate_embedding_cost(docs, model_name='text-embedding-ada-002'):

    # Get the encoding name for the model
    encoding_name = "cl100k_base"  # Used by 'text-embedding-ada-002'

    total_tokens = 0
    for doc in docs:
        if isinstance(doc, str):
            text = doc
        else:
            # If doc is a Document object (from LangChain), get the content
            text = doc.page_content

        num_tokens = num_tokens_from_string(text, encoding_name)
        total_tokens += num_tokens

    # Compute total cost
    total_cost = (total_tokens / 1_000_000) * cost_per_1m_tokens

    print(f"Total tokens: {total_tokens}")
    print(f"Estimated cost: ${total_cost:.4f}")

    return total_tokens, total_cost

# Load the notion content located in the notion_content folder
loader = NotionDirectoryLoader("notion_content")
documents = loader.load()

# Split Notion content into smaller chunks
markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["#","##", "###", "\n\n","\n","."],
    chunk_size=1500,
    chunk_overlap=100)
docs = markdown_splitter.split_documents(documents)

# Estimate the cost of embedding all chunks
total_tokens, total_cost = estimate_embedding_cost(docs)


# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# Convert all chunks into vectors embeddings using OpenAI embedding model
# Store all vectors in FAISS index and save locally to 'faiss_index'
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

print('Local FAISS index has been successfully saved.')