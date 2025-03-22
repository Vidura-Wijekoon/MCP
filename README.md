# 🧠 MCP – Modular Control Plane for Fitness AI Assistant

This module powers the backend intelligence of the **MCP Pro** fitness assistant system.  
It is responsible for retrieving and processing external health, fitness, and exercise data — and exposing it to the AI agent via tool interfaces.

---

## 📦 Folder Structure


---

## 🔌 Responsibilities

- 🧠 Hosts vector retrieval for fitness articles  
- 🔍 Interfaces with external Exercise APIs (e.g. Ninja API)  
- 🧰 Offers tools: `fitness_query_tool`, `exercise_search_tool`  
- 📤 Serves as a Tool Provider for the LLM running in `mcp pro`  

---

## 🏗️ Usage

This repo is not run directly — it is imported by the LLM agent inside [`mcp pro`](https://github.com/your-org/mcp-pro).

Ensure `.env` is set correctly:


---

## 🔁 Used by

- `mcp pro` → AI assistant (Flask app with LLM + tools)  
- Connects to this MCP to delegate tool-based operations

---
