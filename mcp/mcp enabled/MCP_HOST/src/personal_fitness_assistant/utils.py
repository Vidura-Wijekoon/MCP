import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Define paths - Updated for MCP architecture
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MCP_SERVER_DIR = os.path.join(BASE_DIR, '..', 'MCP_SERVER')
USER_DATA_DIR = os.path.join(MCP_SERVER_DIR, "data", "user_data")

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


def calculate_metrics(user_id: str) -> Dict[str, Any]:
    """Calculate usage metrics for a user.
    
    Args:
        user_id: Unique identifier for the user
        
    Returns:
        Dictionary containing usage metrics
    """
    history = get_user_query_history(user_id)
    
    if not history:
        return {
            "total_queries": 0,
            "avg_query_length": 0,
            "avg_response_length": 0,
            "first_query": None,
            "last_query": None
        }
    
    total_queries = len(history)
    total_query_length = sum(len(item["query"]) for item in history)
    total_response_length = sum(len(item["response"]) for item in history)
    
    # Get first and last query timestamps
    sorted_history = sorted(history, key=lambda x: x["timestamp"])
    first_query = sorted_history[0]["timestamp"]
    last_query = sorted_history[-1]["timestamp"]
    
    return {
        "total_queries": total_queries,
        "avg_query_length": total_query_length / total_queries if total_queries > 0 else 0,
        "avg_response_length": total_response_length / total_queries if total_queries > 0 else 0,
        "first_query": first_query,
        "last_query": last_query
    }