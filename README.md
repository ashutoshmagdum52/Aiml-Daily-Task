# AI Daily Task — Setup & Run Guide

## Install all dependencies (run once inside venv)
```
pip install requests python-dotenv google-generativeai PyPDF2 sentence-transformers torch
```

## .env file — required keys
```
OPENWEATHER_API_KEY=your_key
NEWSDATA_API_KEY=your_key
OPENAI_API_KEY=your_groq_key   # used for Day 4, 5, 6, 8
```

---

## Day 1 — Weather Pipeline
```
cd "DAY 1"
python "weather api.py"
```
Output: `weather_data.json`

## Day 2 — News Classifier
```
cd "DAY 2"
python news_engine.py
```
Output: `news_data_day2.json`

## Day 3 — Keyword Search Engine
```
cd "DAY 3"
python search_engine.py
```

## Day 4 — Resume Analyzer (Prompt Engineering)
```
cd "DAY 4"
python resume_task.py
```

## Day 5 — Red-Team Adversarial Testing
```
cd "DAY 5"
python red_team_test.py
```
Output: `test_results.json` + review `failure_report.md`

## Day 6 — Stateful CLI Chatbot
```
cd "DAY 6"
python chatbot.py
```
Interactive CLI — type to chat, 'quit' to exit.

## Day 7 — Semantic Search (Embeddings)
```
cd "DAY 7"
python semantic_search.py
```
Note: downloads ~90MB model on first run.

## Day 8 — RAG Pipeline (Document Chatbot)
```
cd "DAY 8"
python rag_pipeline.py
```
Ask questions about NexusVortex OS. Interactive CLI.
