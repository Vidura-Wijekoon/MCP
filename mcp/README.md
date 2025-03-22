
# 🧠 Fitness Assistant – LLM + Tools (No MCP)

This is a standalone version of a personal fitness assistant powered by LLMs and custom tools, without the full MCP architecture.

---

## 📦 Folder Structure

```
project/
├── app.py                          # Flask server
├── templates/
│   └── index.html                  # Frontend (user query form)
├── src/
│   └── personal_fitness_assistant/
│       ├── main.py                 # Core logic: LLM + tool binding + query processing
│       ├── utils.py                # Save query history, format fitness data
│       └── __init__.py             # Module init
├── data/
│   ├── fitness_articles/           # Vectorstore for health content
│   └── user_data/                  # User-specific query logs
├── requirements.txt                # Project dependencies
└── .env                            # API keys (OpenAI, Exercises API)
```

---

## 🧰 Tools Available

- `fitness_query_tool`: Retrieves health & fitness facts from embedded articles  
- `exercise_search_tool`: Calls Exercise API to fetch workouts by muscle/type/difficulty

---

## 🚀 Getting Started

1. Clone this repo
2. Set your `.env` file with valid API keys:
```
OPENAI_API_KEY=your_openai_key
EXERCISES_API_KEY=your_exercise_api_key
```
3. Install requirements:
```bash
pip install -r requirements.txt
```
4. Run the assistant:
```bash
python app.py
```

---

## 🧠 Powered by

- OpenAI embeddings for vector search  
- LangChain + Groq for LLM orchestration  
- Flask for the API + frontend

---

## 🔮 Future Ideas

- Add calorie tracking and wearable integration  
- Track user progress across sessions  
- Enable speech input/output via Web APIs
