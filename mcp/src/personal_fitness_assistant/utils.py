import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
USER_DATA_DIR = os.path.join(DATA_DIR, "user_data")

# Ensure directories exist
os.makedirs(USER_DATA_DIR, exist_ok=True)


def save_user_query(user_id: str, query: str, response: str) -> None:
    """Save user query and response to a JSON file.
    
    Args:
        user_id: Unique identifier for the user
        query: User's query string
        response: Assistant's response to the query
    """
    # Create user directory if it doesn't exist
    user_dir = os.path.join(USER_DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # Create or load history file
    history_file = os.path.join(user_dir, "query_history.json")
    
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new query and response
    history.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response
    })
    
    # Save updated history
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def get_user_query_history(user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieve user query history.
    
    Args:
        user_id: Unique identifier for the user
        limit: Maximum number of history items to return (newest first)
        
    Returns:
        List of query history items
    """
    history_file = os.path.join(USER_DATA_DIR, user_id, "query_history.json")
    
    if not os.path.exists(history_file):
        return []
    
    with open(history_file, "r") as f:
        history = json.load(f)
    
    # Sort by timestamp (newest first)
    history.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Apply limit if specified
    if limit is not None:
        history = history[:limit]
    
    return history


def format_fitness_data(data: Dict[str, Any]) -> str:
    """Format fitness data for display.
    
    Args:
        data: Dictionary containing fitness data
        
    Returns:
        Formatted string representation of the data
    """
    formatted = []
    
    for key, value in data.items():
        if isinstance(value, dict):
            formatted.append(f"## {key.title()}")
            for sub_key, sub_value in value.items():
                formatted.append(f"* {sub_key.replace('_', ' ').title()}: {sub_value}")
        else:
            formatted.append(f"## {key.title()}")
            formatted.append(f"{value}")
        
        formatted.append("")
    
    return "\n".join(formatted)


def parse_fitness_metrics(text: str) -> Dict[str, Any]:
    """Parse fitness metrics from text.
    
    Args:
        text: Text containing fitness metrics
        
    Returns:
        Dictionary of parsed metrics
    """
    metrics = {}
    
    # Example parsing logic - this would be expanded based on specific needs
    if "BMI" in text or "body mass index" in text.lower():
        # Simple regex could be used here for more complex parsing
        metrics["bmi"] = "Mentioned but value not extracted"
    
    if "weight" in text.lower():
        metrics["weight"] = "Mentioned but value not extracted"
    
    if "calories" in text.lower():
        metrics["calories"] = "Mentioned but value not extracted"
    
    return metrics