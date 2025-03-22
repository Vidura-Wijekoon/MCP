# MCP-Enabled Fitness Assistant

This is a restructured version of the personal fitness assistant following the MCP (Model-Client-Protocol) architecture pattern.

## Architecture Overview

### MCP_HOST
The user-facing interface - chatbot, smart device, or web assistant

### MCP_PROTOCOL
The bridge between the LLM and tools (services)

### MCP_SERVER
External service connectors (wearables, nutrition APIs, medical data, etc.)

## Folder Structure

```
MCP_HOST/
│
├── app.py                          → Flask server (Entry point)
├── templates/
│   └── index.html                  → Frontend (AI Assistant interface)
└── src/personal_fitness_assistant/
    ├── main.py                     → LLM logic, tool binding (MCP Client + Protocol)
    ├── utils.py                    → History + metric tracking (local state)
    └── tools/                      → (Optional) Decouple tools into separate files

MCP_SERVER/
├── data/fitness_articles/          → Vectorstore (embedded articles)
├── data/user_data/                 → Per-user query logs (JSON)
```

## Future Enhancements

- Move tools into src/tools/ directory
- Add real-time sync with wearable devices (e.g., steps, heart rate)
- Add secure authentication layer per user
- Connect to medical API for personalized health insights