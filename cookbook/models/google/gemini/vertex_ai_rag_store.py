"""Vertex AI RAG Store with Gemini.

Vertex AI RAG Store allows Gemini to retrieve and generate responses
based on documents stored in your Vertex AI RAG corpora.

Prerequisites:
1. Set up Vertex AI RAG Store corpus in Google Cloud Console
2. Export environment variables:
   export GOOGLE_GENAI_USE_VERTEXAI="true"
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   export GOOGLE_CLOUD_LOCATION="your-location"

Run `pip install google-generativeai` to install dependencies.

Requirements:
- google-generativeai
- agno (this library)
- Google Cloud Project with Vertex AI enabled
- Vertex AI RAG Store corpus set up
"""

import os
from agno.agent import Agent
from agno.models.google import Gemini

# Set up environment variables for Vertex AI
# You can also set these as environment variables before running the script
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "stratto-backend"  # Replace with your actual project ID
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-east4"  # Replace with your actual location

# Validate required environment variables
required_vars = ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"]
for var in required_vars:
    if not os.getenv(var) or os.getenv(var) == "stratto-backend":
        raise ValueError(f"Please set the {var} environment variable with your actual values.")

# Set up Google Cloud authentication
# Option 1: Use service account key file
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/service-account-key.json"

# Option 2: Use Application Default Credentials (ADC)
# Make sure you're authenticated with: gcloud auth application-default login

# Alternative: Load from .env file if you prefer
# from dotenv import load_dotenv
# load_dotenv()

# Replace with your actual Vertex AI RAG Store corpus ID
# Format: "projects/{project_id}/locations/{location}/ragCorpora/{corpus_id}"
# )
rag_corpus_id = "projects/stratto-backend/locations/us-east4/ragCorpora/5188146770730811392"
rag_file_id1= "projects/945758462127/locations/us-east4/ragCorpora/5188146770730811392/ragFiles/5514461022525698996"
rag_file_ids= [rag_file_id1]
agent = Agent(
    model=Gemini(
        id="gemini-2.5-flash",
        vertexai_rag_store=True,
        vertexai_rag_store_corpus=rag_corpus_id,
        vertexai_rag_store_file_ids=rag_file_ids,
        vertexai=True,  # Use Vertex AI endpoint
    ),
    show_tool_calls=True,
    markdown=True,
)

# Ask questions that can be answered from your RAG corpus documents
try:
    print("Testing RAG Store with question 1...")
    agent.print_response("what is the experience of ala baccari?")

    print("\nTesting RAG Store with question 2...")
    agent.print_response("what is the experience of ahmed attafi?")
except Exception as e:
    print(f"Error occurred: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure your Google Cloud project ID and location are correct")
    print("2. Ensure you have proper authentication set up (service account or ADC)")
    print("3. Verify that the RAG corpus ID exists and is accessible")
    print("4. Check that you have the necessary Vertex AI permissions")