@echo off
echo Starting Personal AI Chatbot on port 8001...
uvicorn app:app --reload --port 8001
pause
