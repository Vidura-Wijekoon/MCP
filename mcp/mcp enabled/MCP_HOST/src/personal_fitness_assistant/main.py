import os
import re
import logging
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq

# ------------------------------- #
#         CONFIGURATION          #
# ------------------------------- #

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base Paths - Updated for MCP architecture
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MCP_SERVER_DIR = os.path.join(BASE_DIR, '..', 'MCP_SERVER')
VECTORSTORE_PATH = os.path.join(MCP_SERVER_DIR, "data", "fitness_articles", "vectorstore.parquet")

# Ensure data directory exists
os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)

# LLM instance
_llm = None

# ------------------------------- #
#         HELPER FUNCTIONS       #
# ------------------------------- #

def get_env_var(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} not set in environment.")
    return value


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    main_content = soup.find("article")
    content = main_content.get_text() if main_content else soup.text
    return re.sub(r"\n\n+", "\n\n", content).strip()


# ------------------------------- #
#           TOOLS                #
# ------------------------------- #

@tool
def fitness_query_tool(query: str) -> str:
    """
    Answers general fitness-related questions by retrieving relevant content from fitness articles.

    This tool uses a vector store built from reliable health and fitness websites 
    to return the most relevant information based on the user's query.

    Args:
        query (str): A general fitness-related question, such as 
                     "What are the benefits of daily exercise?"

    Returns:
        str: A formatted string with relevant information from the top-matching documents.
    """
    try:
        from MCP_SERVER.main import load_vectorstore, load_fitness_docs, split_documents, create_vectorstore
        
        if not os.path.exists(VECTORSTORE_PATH):
            logger.info("Vector store not found. Creating one...")
            docs = load_fitness_docs()
            splits = split_documents(docs)
            vectorstore = create_vectorstore(splits)
        else:
            vectorstore = load_vectorstore()

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)

        formatted_context = "\n\n".join(
            [f"==DOCUMENT {i + 1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)]
        )

        return formatted_context

    except Exception as e:
        logger.error(f"Error in fitness_query_tool: {e}")
        return "There was an error retrieving fitness information. Please try again later."


@tool
def exercise_search_tool(query: str) -> str:
    """
    Finds exercises based on user queries about specific muscle groups, exercise types, or difficulty levels.

    This tool analyzes the input query to extract relevant filters (e.g., muscle, type, difficulty)
    and fetches matching exercises from the external Exercise API.

    Args:
        query (str): A user search query such as "biceps workouts", "beginner stretching routine", or "cardio exercises".

    Returns:
        str: A formatted string listing up to 5 exercises, including details like name, type, muscle targeted, and instructions.
    """
    try:
        from MCP_SERVER.main import get_exercises
        
        muscles = ["abdominals", "abs", "biceps", "triceps", "chest", "forearms", "glutes", "hamstrings",
                   "lats", "lower_back", "middle_back", "neck", "quadriceps", "quads", "traps", "shoulders", "calves"]
        types = ["cardio", "olympic_weightlifting", "plyometrics", "powerlifting", "strength", "stretching", "strongman"]
        difficulties = ["beginner", "intermediate", "expert"]

        muscle = next((m for m in muscles if m in query.lower()), None)
        if muscle == "abs":
            muscle = "abdominals"

        exercise_type = next((t for t in types if t in query.lower()), None)
        difficulty = next((d for d in difficulties if d in query.lower()), None)
        name = query if not any([muscle, exercise_type, difficulty]) else None

        exercises = get_exercises(muscle=muscle, type=exercise_type, difficulty=difficulty, name=name)

        if not exercises:
            return "No exercises found matching your search. Try different terms."

        formatted = "Found the following exercises:\n\n"
        for i, ex in enumerate(exercises[:5], 1):
            formatted += f"""EXERCISE {i}:
Name: {ex.get('name')}
Type: {ex.get('type')}
Muscle: {ex.get('muscle')}
Equipment: {ex.get('equipment')}
Difficulty: {ex.get('difficulty')}

Instructions:
{ex.get('instructions')}\n\n"""

        return formatted

    except Exception as e:
        logger.error(f"Error in exercise_search_tool: {e}")
        return "There was an error searching for exercises. Please try again later."


# ------------------------------- #
#         LLM INTEGRATION        #
# ------------------------------- #

def initialize_fitness_assistant():
    """
    Initialize the fitness assistant by setting up the LLM with tools.
    """
    global _llm
    try:
        _llm = ChatGroq(
            groq_api_key=get_env_var("GROQ_API_KEY"),
            model_name="llama3-70b-8192"
        )
        _llm = _llm.bind_tools([fitness_query_tool, exercise_search_tool])
        logger.info("Fitness assistant initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize fitness assistant: {e}")
        raise


def query_fitness_assistant(query_text: str) -> str:
    """
    Process a user query and return a response from the fitness assistant.
    
    Args:
        query_text: The user's query text
        
    Returns:
        The assistant's response
    """
    if not _llm:
        logger.warning("Fitness assistant not initialized. Initializing now...")
        initialize_fitness_assistant()
    
    try:
        # Prepare system message with instructions
        system_message = """
        You are a helpful fitness assistant that provides accurate information about exercise, 
        nutrition, and general fitness topics. Use the available tools when appropriate to 
        provide the most relevant and accurate information.
        
        When responding to queries:
        1. For general fitness questions, use the fitness_query_tool to retrieve information
        2. For exercise recommendations, use the exercise_search_tool to find relevant exercises
        3. Always be encouraging and supportive of the user's fitness journey
        4. Provide concise, accurate information without making claims not supported by data
        5. If you don't know something, admit it rather than making up information
        """
        
        # Get response from LLM
        response = _llm.invoke([{"role": "system", "content": system_message}, 
                               {"role": "user", "content": query_text}])
        
        return response.content
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return "I'm sorry, I encountered an error processing your request. Please try again later."