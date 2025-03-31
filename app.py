from flask import Flask, jsonify, request, render_template
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Get API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# Initialize components
index_name = "medichatbot"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# System prompt for the chatbot
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieval context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise, yet detailed enough.\n\n"
    "{context}"
)

# Initialize Pinecone vector store
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Initialize Mistral LLM
llm = ChatMistralAI(
    model="mistral-medium",  # Using medium for better quality
    temperature=0.4,
    api_key=MISTRAL_API_KEY
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()  # Get JSON data
    user_message = data.get("message", "").strip()
    
    if not user_message:
        return jsonify({"response": "Please enter a message."})  # Handle empty input
    
    print("User query:", user_message)

    # Get AI-generated response
    response = rag_chain.invoke({"input": user_message})
    bot_response = response.get("answer", "Sorry, I couldn't understand that.")
    
    print("Response:", bot_response)
    
    return jsonify({"response": bot_response})  # Return JSON response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)