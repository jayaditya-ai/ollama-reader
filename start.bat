@echo off
echo ==========================================
echo  PDF RAG System - Starting Web UI
echo ==========================================
echo.

REM Check if Ollama is running
docker ps | find "ollama" >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Ollama...
    docker-compose up -d 2>nul
    if %errorlevel% neq 0 (
        echo Ollama container already exists, starting it...
        docker start ollama >nul 2>&1
    )
    timeout /t 5 /nobreak > nul
) else (
    echo Ollama is already running...
)

echo.
echo Checking Python packages...
python -m pip install -q streamlit PyPDF2 chromadb langchain-text-splitters requests

echo.
echo Checking AI models (first run will download ~2.3GB)...
ollama pull llama3.2
ollama pull nomic-embed-text

echo.
echo Starting web interface...
echo.
echo ==========================================
echo  Your browser will open automatically
echo  Or visit: http://localhost:8501
echo ==========================================
echo.

REM Disable Streamlit email prompt
set STREAMLIT_EMAIL=""

REM Start the app (browser.gatherUsageStats=false skips email prompt)
python -m streamlit run app.py --server.headless=true --browser.gatherUsageStats=false

pause
