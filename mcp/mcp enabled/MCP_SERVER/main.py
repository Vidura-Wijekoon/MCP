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
from langchain_openai import OpenAIEmbeddings

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
#       EXTERNAL API SERVICES    #
# ------------------------------- #

def get_exercises(muscle: Optional[str] = None, type: Optional[str] = None,
                  difficulty: Optional[str] = None, name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Connect to the Exercise API to fetch exercises based on filters.
    
    Args:
        muscle: Target muscle group (e.g., biceps, chest)
        type: Exercise type (e.g., strength, cardio)
        difficulty: Exercise difficulty level
        name: Exercise name to search for
        
    Returns:
        List of exercise dictionaries
    """
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


# ------------------------------- #
#       DOCUMENT LOADING         #
# ------------------------------- #

def load_fitness_docs_from_file(file_path: str) -> List[Document]:
    """Load fitness documents from a local file.
    
    Args:
        file_path: Path to the file containing fitness documents
        
    Returns:
        List of Document objects
    """
    # This is a placeholder for future implementation
    # Could be extended to load from various file formats
    pass