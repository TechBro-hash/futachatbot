from google import genai
import json
import time
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# === CONFIGURATION ===
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

KB_PATH = "futa_knowledge_base.txt"
EMBEDDINGS_PATH = "futa_kb_embeddings.json"

# === SETUP ===
client = genai.Client(api_key=GEMINI_API_KEY)

def extract_sections_from_text(text: str) -> List[Dict[str, str]]:
    """
    Extract structured sections from the knowledge base text
    """
    sections = []
    
    # Split by major sections (all caps headers)
    section_pattern = r'([A-Z\s]+)\n=+\n(.*?)(?=\n[A-Z\s]+\n=+\n|\Z)'
    matches = re.finditer(section_pattern, text, re.DOTALL)
    
    for match in matches:
        section_title = match.group(1).strip()
        section_content = match.group(2).strip()
        
        if section_content:
            # Split into smaller chunks if too long
            chunks = split_content_into_chunks(section_content)
            
            for i, chunk in enumerate(chunks):
                sections.append({
                    "section": f"{section_title} - Part {i+1}",
                    "content": chunk
                })
    
    return sections

def split_content_into_chunks(content: str, max_length: int = 800) -> List[str]:
    """
    Split content into manageable chunks
    """
    # Split by double newlines first
    parts = re.split(r'\n\n+', content)
    
    chunks = []
    current_chunk = ""
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # If adding this part would make chunk too long, start a new chunk
        if len(current_chunk + part) > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = part
        else:
            if current_chunk:
                current_chunk += "\n\n" + part
            else:
                current_chunk = part
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# === LOAD KNOWLEDGE BASE ===
print("Reading knowledge base...")
with open(KB_PATH, "r", encoding="utf-8") as f:
    kb_text = f.read()

print("Extracting sections...")
kb_sections = extract_sections_from_text(kb_text)
print(f"Found {len(kb_sections)} sections to embed")

# === EMBED EACH CHUNK ===
embeddings = []
for i, entry in enumerate(kb_sections):
    content = entry["content"]
    section = entry["section"]
    
    print(f"Creating embedding for section {i+1}/{len(kb_sections)}: {section}")
    
    # Call Gemini Embeddings API
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=content
    )
    embedding = result.embeddings[0].values
    embeddings.append({
        "section": section,
        "content": content,
        "embedding": embedding
    })
    time.sleep(0.2)  # Be polite to the API

# === SAVE EMBEDDINGS ===
with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
    json.dump(embeddings, f, indent=2)

print(f"✅ Saved {len(embeddings)} embeddings to {EMBEDDINGS_PATH}")
print("🎉 Embedding creation completed successfully!")
