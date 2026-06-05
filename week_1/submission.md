# Week 1 Submission: Raw Terminal Chatbot (CSOT 2026 Agentic Track)

This document outlines the submission details for the Week 1 deliverable: building a raw, multi-turn terminal chatbot using primitive client libraries and manual state management.

---

## Submission Files Checklist
- [x] **`.env`**: Local environment variables configuration file containing the `OPENROUTER_API_KEY`.
- [x] **`.gitignore`**: Excludes `.env`, `.venv/`, `venv/`, and Python cache directories.
- [x] **`chatbot.py`**: Fully self-contained python script containing the `ChatAgent` class and interactive CLI loop.
- [x] **`submission.md`**: This summary and documentation file.

---

## Architecture & Implementation Details

### 1. Key API Primitives & Client Logic
We avoid high-level orchestration helper classes and frameworks (like LangChain or CrewAI). Instead, we initialize the raw `openai.OpenAI` client pointing to the OpenRouter endpoint:
- **Stateless Requests**: The OpenRouter API is completely stateless; it does not remember previous requests.
- **Manual Message State**: We explicitly maintain conversation history manually in a Python list of dictionaries:
  ```python
  self.messages = [
      {"role": "system", "content": "You are a helpful assistant."}
  ]
  ```
- **Context Preservation**: With every user input, the message is appended locally to `self.messages`, and the entire array is sent in the raw API request to keep conversation context alive.

### 2. Context Overflow Policies
To address token budgets and prevent context window issues, `chatbot.py` supports two manual policies:
1. **Drop Policy**: If the conversation exceeds the configured maximum turn limit, the oldest user-assistant turn (2 messages) is dropped.
2. **Summarize / Compaction Policy (Bonus)**: Automatically triggers an auxiliary stateless call to summarize the oldest turns. The original turns are then removed and replaced with a single `[System Note: Summary of previous turns: ...]` message.

### 3. Features & User Interactions
- **Streaming Response**: Tokens are printed in real-time as they stream back from OpenRouter.
- **Diagnostics Commands**:
  - `/exit` or `/quit`: Clean exit from loop.
  - `/reset`: Wipes the message history (demonstrates context loss).
  - `/tokens`: Prints accumulated input and output token counts.
  - `/compact`: Triggers history compaction manually.
  - `/history`: Displays the raw, manually managed messages array structure to view role alternation.
  - `/policy`: Dynamically toggles context management policies (Drop vs Summarize).

---

## How to Run & Verify

1. **Verify Dependencies**:
   Ensure `openai` and `python-dotenv` are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify API Configuration**:
   Ensure `.env` contains:
   ```env
   OPENROUTER_API_KEY=your_actual_key_here
   ```

3. **Launch the Chatbot**:
   Run the submission file directly:
   ```bash
   python chatbot.py
   ```

4. **Interact**:
   - Provide a model name (e.g. `deepseek/deepseek-v4-flash:free` or press enter for default).
   - Enter your prompt limit and choose overflow policies.
   - Run commands like `/history` and `/tokens` to inspect the internal state!
