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

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_PATH = os.path.join(DATA_DIR, "fitness_articles", "vectorstore.parquet")

# Ensure data directory exists
os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)


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


def load_fitness_docs(urls: List[str] = None) -> List[Document]:
    if urls is None:
        urls = [
            "https://www.healthline.com/nutrition/10-benefits-of-exercise",
            "https://www.webmd.com/fitness-exercise/guide/the-basics-of-fitness",
            "https://www.mayoclinic.org/healthy-lifestyle/fitness/in-depth/fitness/art-20048269"
        ]
    docs = []
    for url in urls:
        try:
            loader = RecursiveUrlLoader(url, max_depth=2, extractor=bs4_extractor)
            docs.extend(loader.load())
        except Exception as e:
            logger.warning(f"Failed to load {url}: {e}")
    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
    return splitter.split_documents(documents)


def create_vectorstore(splits: List[Document], persist: bool = True) -> SKLearnVectorStore:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=get_env_var("OPENAI_API_KEY")
    )
    store = SKLearnVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_path=VECTORSTORE_PATH,
        serializer="parquet"
    )
    if persist:
        store.persist()
    return store


def load_vectorstore() -> SKLearnVectorStore:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=get_env_var("OPENAI_API_KEY")
    )
    return SKLearnVectorStore(
        embedding=embeddings,
        persist_path=VECTORSTORE_PATH,
        serializer="parquet"
    )


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



def get_exercises(muscle: Optional[str] = None, type: Optional[str] = None,
                  difficulty: Optional[str] = None, name: Optional[str] = None) -> List[Dict[str, Any]]:
    params = {k: v for k, v in [('muscle', muscle), ('type', type),
                                ('difficulty', difficulty), ('name', name)] if v}
    headers = {'X-Api-Key': get_env_var("EXERCISES_API_KEY")}
    try:
        response = requests.get("https://api.api-ninjas.com/v1/exercises", headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return []


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
Instructions: {ex.get('instructions')}\n\n"""
        return formatted

    except Exception as e:
        logger.error(f"Error in exercise_search_tool: {e}")
        return "There was an error searching for exercises. Please try again later."


# ------------------------------- #
#        LLM WRAPPER             #
# ------------------------------- #

def create_augmented_llm():
    """Create an LLM augmented with fitness tools."""
    try:
        llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
            groq_api_key=get_env_var("GROQ_API_KEY")
        )
        return llm.bind_tools([fitness_query_tool, exercise_search_tool])
    except Exception as e:
        logger.warning(f"LLM unavailable: {e}")
        return None


def query_fitness_assistant(query: str) -> str:
    """Query the fitness assistant with a user question."""
    augmented_llm = create_augmented_llm()

    if augmented_llm is None:
        # Direct tool fallback if no Groq key
        if any(word in query.lower() for word in ["exercise", "workout", "routine", "biceps", "chest", "leg", "muscle"]):
            return exercise_search_tool.invoke({"query": query})
        else:
            return fitness_query_tool.invoke({"query": query})

    instructions = """
    You are a helpful assistant that can answer questions about fitness and exercise.
    Use the fitness_query_tool for general questions.
    Use the exercise_search_tool for looking up exercises by muscle group, type, or difficulty.
    """

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": query}
    ]

    try:
        # First LLM call to decide if a tool is needed
        response = augmented_llm.invoke(messages)
        logger.info(f"Raw response from LLM: {response}")

        # Handle tool call if it exists
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            logger.info(f"Tool call detected: {tool_name}({tool_args})")

            if tool_name == "fitness_query_tool":
                tool_output = fitness_query_tool.invoke(tool_args)
            elif tool_name == "exercise_search_tool":
                tool_output = exercise_search_tool.invoke(tool_args)
            else:
                tool_output = "Tool not recognized."

            logger.info(f"Tool output: {tool_output}")

            # Send tool response back to LLM
            follow_up_messages = [
                *messages,
                response,
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_output
                }
            ]

            final_response = augmented_llm.invoke(follow_up_messages)

            if hasattr(final_response, "content"):
                return final_response.content or "LLM returned no final message after tool execution."
            elif isinstance(final_response, str):
                return final_response
            else:
                return str(final_response)

        # If LLM gave an answer directly
        if hasattr(response, "content"):
            return response.content or "LLM returned no direct answer."
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return "Something went wrong processing your request."




def initialize_fitness_assistant():
    if not os.path.exists(VECTORSTORE_PATH):
        logger.info("Initializing fitness assistant vector store...")
        docs = load_fitness_docs()
        splits = split_documents(docs)
        create_vectorstore(splits)
        logger.info(f"Vector store created at {VECTORSTORE_PATH}")
    else:
        logger.info(f"Vector store loaded from {VECTORSTORE_PATH}")


# ------------------------------- #
#              MAIN              #
# ------------------------------- #

if __name__ == "__main__":
    initialize_fitness_assistant()

    doc_query = "What are the benefits of regular exercise?"
    logger.info(f"Document Query: {doc_query}")
    print(query_fitness_assistant(doc_query))

    exercise_query = "Show me beginner biceps exercises"
    logger.info(f"Exercise Query: {exercise_query}")
    print(query_fitness_assistant(exercise_query))
