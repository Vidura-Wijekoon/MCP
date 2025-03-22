# Personal Fitness Assistant Documentation

The Personal Fitness Assistant is a tool for retrieving and analyzing fitness-related documents. It uses natural language processing and vector embeddings to provide relevant information about fitness topics.

## Features

- Load fitness-related documents from specified URLs
- Process and split documents into manageable chunks
- Create vector embeddings for efficient retrieval
- Query the assistant with natural language questions about fitness
- Save user query history for future reference

## Usage

### Basic Query

```python
from src.personal_fitness_assistant.main import query_fitness_assistant

# Ask a fitness-related question
response = query_fitness_assistant("What are the benefits of regular exercise?")
print(response)
```

### Initializing the Assistant

```python
from src.personal_fitness_assistant.main import initialize_fitness_assistant

# Initialize the assistant (creates vector store if it doesn't exist)
initialize_fitness_assistant()
```

### Saving User Queries

```python
from src.personal_fitness_assistant.utils import save_user_query

# Save a user query and response
user_id = "user123"
query = "What are the benefits of regular exercise?"
response = "Regular exercise has numerous benefits including..."
save_user_query(user_id, query, response)
```

### Retrieving User Query History

```python
from src.personal_fitness_assistant.utils import get_user_query_history

# Get user's query history (most recent 5 queries)
user_id = "user123"
history = get_user_query_history(user_id, limit=5)
print(history)
```

## Architecture

The Personal Fitness Assistant uses the following components:

1. **Document Loading**: Uses `RecursiveUrlLoader` to load fitness documents from specified URLs
2. **Text Processing**: Extracts and cleans text content using BeautifulSoup
3. **Document Splitting**: Splits documents into manageable chunks using `RecursiveCharacterTextSplitter`
4. **Vector Embeddings**: Creates vector embeddings using OpenAI's embedding model
5. **Vector Store**: Stores embeddings using `SKLearnVectorStore` for efficient retrieval
6. **Query Tool**: Provides a tool for querying fitness information
7. **Language Model**: Uses Claude to generate human-like responses based on retrieved information

## Dependencies

- langchain
- langchain-openai
- langchain-anthropic
- scikit-learn
- beautifulsoup4
- lxml
- pandas
- pyarrow