from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os 
import time

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load and process documents
extracted_data = load_pdf_file(data=r'c:\medical-chatbot\Data')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define the index name
index_name = "medichatbot"

# List existing indexes
existing_indexes = pc.list_indexes().names()

# Check if the index exists before creating it
if index_name not in existing_indexes:
    print(f"Creating index {index_name}...")
    # Create index with serverless configuration
    pc.create_index(
        name=index_name,
        dimension=384,  # Matches all-MiniLM-L6-v2 embedding size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
    # Wait for index to be ready
    while not pc.describe_index(index_name).status['ready']:
        print("Waiting for index to be ready...")
        time.sleep(5)

# Store documents in Pinecone
print("Storing documents in Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("✅ Documents successfully stored in Pinecone!")