import json
import os
from datetime import datetime

NOTES_FILE = "notes/notes_data.json"

def init_notes_store():
    """Ensure the notes JSON file exists."""
    if not os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def save_note(content: str):
    """Save a new note to the JSON file."""
    init_notes_store()
    
    with open(NOTES_FILE, "r", encoding="utf-8") as f:
        try:
            notes = json.load(f)
        except json.JSONDecodeError:
            notes = []
            
    new_note = {
        "timestamp": datetime.now().isoformat(),
        "content": content
    }
    notes.append(new_note)
    
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=4)
    
    return True

def get_all_notes() -> str:
    """Retrieve all saved notes formatted as a single string."""
    init_notes_store()
    
    with open(NOTES_FILE, "r", encoding="utf-8") as f:
        try:
            notes = json.load(f)
        except json.JSONDecodeError:
            return "No personal notes found."
            
    if not notes:
        return "No personal notes found."
        
    formatted_notes = ["Your saved notes:"]
    for i, note in enumerate(notes, 1):
        # Format timestamp to show just date and time cleanly
        dt = datetime.fromisoformat(note['timestamp'])
        date_str = dt.strftime("%Y-%m-%d %H:%M")
        formatted_notes.append(f"{i}. [{date_str}] {note['content']}")
        
    return "\n".join(formatted_notes)
