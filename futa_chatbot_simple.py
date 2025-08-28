import json
import numpy as np
from google import genai
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import List, Optional

# Load environment variables
load_dotenv()

# === CONFIGURATION ===
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found in environment variables")
    print("Please create a .env file with your API key")
    exit(1)

EMBEDDINGS_PATH = "futa_kb_embeddings.json"
TOP_K = 5  # Number of KB chunks to retrieve

client = genai.Client(api_key=GEMINI_API_KEY)

# === LOAD EMBEDDINGS ===
try:
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    print(f"✅ Loaded {len(kb)} knowledge base embeddings")
except FileNotFoundError:
    print(f"❌ Embeddings file {EMBEDDINGS_PATH} not found!")
    print("Please run futa_embedding_creator_simple.py first to create embeddings.")
    exit(1)

kb_vectors = np.array([entry["embedding"] for entry in kb])

# === EMBED USER QUERY ===
def embed_query(query):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    return np.array(result.embeddings[0].values)

# === COSINE SIMILARITY ===
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === RETRIEVE TOP KB CHUNKS ===
def retrieve_top_k(query_vec, k=TOP_K):
    sims = [cosine_similarity(query_vec, vec) for vec in kb_vectors]
    top_indices = np.argsort(sims)[-k:][::-1]
    return [kb[i] for i in top_indices]

# === STRUCTURED RESPONSE MODELS ===
class ListItem(BaseModel):
    item: str

class StructuredResponse(BaseModel):
    title: str
    introduction: str
    items: List[ListItem]
    conclusion: Optional[str] = None

# === FORMAT RESPONSE ===
def format_response(text):
    """
    Post-process the response to ensure proper list formatting
    """
    import re
    
    # Split into lines
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append('')
            continue
            
        # Check if this line contains a bullet point list
        if '•' in line:
            # Split by bullet points and format each item
            parts = line.split('•')
            for i, part in enumerate(parts):
                part = part.strip()
                if part:
                    if i == 0:  # First part (before first bullet)
                        formatted_lines.append(part)
                    else:  # Parts with bullets
                        formatted_lines.append(f'• {part}')
                        formatted_lines.append('')  # Add blank line after each item
        else:
            formatted_lines.append(line)
    
    # Join lines and clean up multiple blank lines
    result = '\n'.join(formatted_lines)
    
    # Clean up multiple consecutive blank lines
    result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
    
    return result

# === GENERATE AGENTIC ANSWER ===
def agentic_answer(user_q, history=None):
    query_vec = embed_query(user_q)
    top_chunks = retrieve_top_k(query_vec)
    context = "\n".join(f"[{c['section']}] {c['content']}" for c in top_chunks)

    # Build conversation history string
    history_str = ""
    if history:
        for turn in history[-5:]:  # Use last 5 turns
            history_str += f"User: {turn['user']}\nAgent: {turn['agent']}\n"

    # Check if the query is asking for a list
    list_keywords = ['faculty', 'faculties', 'department', 'departments', 'program', 'programs', 
                    'center', 'centers', 'facility', 'facilities', 'requirement', 'requirements',
                    'list', 'what are', 'how many', 'which', 'name', 'names']
    
    is_list_query = any(keyword in user_q.lower() for keyword in list_keywords)
    
    if is_list_query:
        # Use structured output for list queries
        prompt = f"""
You are FUTA, an agentic AI assistant for the Federal University of Technology Akure (FUTA). 
The user is asking for a list of items. Please provide a structured response with a title, introduction, list of items, and optional conclusion.

Use ONLY the knowledge provided below to answer.

Conversation so far:
{history_str}
User: {user_q}

Knowledge Base:
{context}
"""
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": StructuredResponse,
                }
            )
            
            # Parse the structured response
            structured_data = response.parsed
            
            # Format the structured response
            formatted_response = f"{structured_data.title}\n\n"
            formatted_response += f"{structured_data.introduction}\n\n"
            
            for item in structured_data.items:
                formatted_response += f"• {item.item}\n\n"
            
            if structured_data.conclusion:
                formatted_response += f"{structured_data.conclusion}"
            
            return formatted_response
            
        except Exception as e:
            # Fallback to regular text response if structured output fails
            print(f"Structured output failed, falling back to text: {e}")
    
    # Regular text response for non-list queries
    prompt = f"""
You are FUTA, an agentic AI assistant conversational bot for the Federal University of Technology Akure (FUTA). Your job is to help users by answering their questions using ONLY the knowledge provided below.

Instructions:
- If the user's question is incomplete, ambiguous, or unclear, politely ask them to clarify or complete their question before answering. Do NOT attempt to answer until the question is clear.
- If the answer is not found in the knowledge, say: "I'm sorry, I don't have enough information to answer that based on my current knowledge about FUTA. Could you clarify or ask something else?"
- If the question is clear and the answer is in the knowledge, provide a well-structured, easy-to-read answer.
- Use bullet points (•) for lists
- Use numbered lists (1., 2., 3.) for steps or sequences
- Use clear headings in CAPITAL LETTERS
- Break information into logical paragraphs
- Keep sentences concise and easy to understand
- Always be polite, professional, and conversational.
- If the user asks about something outside FUTA or the provided knowledge, politely let them know you are only able to answer questions about FUTA.
- Use the conversation history to provide context-aware responses and follow up on previous questions.
- Do NOT use markdown formatting like **bold** or # headings. Use plain text with clear structure.

Conversation so far:
{history_str}
User: {user_q}

Knowledge Base:
{context}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    # Format the response to ensure proper list formatting
    formatted_response = format_response(response.text)
    return formatted_response

def print_banner():
    """Print the chatbot banner"""
    print("=" * 60)
    print("🎓 FUTA Knowledge Assistant - Simple Version")
    print("Your AI guide to the Federal University of Technology Akure")
    print("=" * 60)
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("Type 'help' for sample questions")
    print("Type 'clear' to clear conversation history")
    print("Type 'history' to see recent conversation")
    print("=" * 60)

def print_help():
    """Print help information"""
    print("\n💡 Sample questions you can ask:")
    print("• What are the admission requirements for FUTA?")
    print("• What faculties does FUTA have?")
    print("• How can I contact FUTA?")
    print("• What research centers are available?")
    print("• What is the student population?")
    print("• What sports facilities are available?")
    print("• Tell me about the alumni network")
    print("• What international partnerships does FUTA have?")
    print("• What are the academic programs offered?")
    print("• What is the campus location and address?")

# === INTERACTIVE Q&A WITH MEMORY ===
def main():
    print_banner()
    print(f"📊 Loaded {len(kb)} knowledge sections")
    print("\n🚀 Ready to chat! Ask me anything about FUTA.\n")
    
    history = []
    while True:
        try:
            user_q = input("🤔 You: ").strip()
            
            # Check for exit commands
            if user_q.lower() in ("exit", "quit", "bye", "q"):
                print("\n👋 Thank you for using FUTA Knowledge Assistant! Goodbye!")
                break
            
            # Check for help command
            if user_q.lower() in ['help', 'h', '?']:
                print_help()
                continue
            
            # Check for clear command
            if user_q.lower() in ['clear', 'reset']:
                history = []
                print("🗑️ Conversation history cleared!")
                continue
            
            # Check for history command
            if user_q.lower() in ['history', 'chat']:
                if history:
                    print("\n📜 Conversation History:")
                    for i, turn in enumerate(history[-5:], 1):
                        print(f"{i}. User: {turn['user']}")
                        print(f"   Agent: {turn['agent'][:100]}...")
                else:
                    print("📜 No conversation history yet.")
                continue
            
            # Skip empty input
            if not user_q:
                continue
            
            # Generate answer
            print("🤖 FUTA Assistant: ", end="", flush=True)
            answer = agentic_answer(user_q, history)
            print(answer)
            print()  # Add spacing
            
            # Add to history
            history.append({"user": user_q, "agent": answer})
            
            # Keep only last 10 turns to prevent context from getting too long
            if len(history) > 10:
                history = history[-10:]
                
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using FUTA Knowledge Assistant! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again or type 'quit' to exit.\n")

if __name__ == "__main__":
    main()
