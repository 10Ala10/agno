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
"""

from agno.agent import Agent
from agno.models.google import Gemini

# Replace with your actual Vertex AI RAG Store corpus ID
# Format: "projects/{project_id}/locations/{location}/ragCorpora/{corpus_id}"
corpus_id = "projects/your-project-id/locations/your-location/ragCorpora/your-corpus-id"

agent = Agent(
    model=Gemini(
        id="gemini-2.5-flash",
        vertexai_rag_store=True,
        vertexai_rag_store_corpus=corpus_id,
        vertexai=True,  # Use Vertex AI endpoint
    ),
    show_tool_calls=True,
    markdown=True,
)

# Ask questions that can be answered from your RAG corpus documents
agent.print_response("What information can you provide about our products from the knowledge base?")