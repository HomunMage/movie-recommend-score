# src/main.py

import os
import time
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from ServerTee import ServerTee

# --- Logging setup ---
today_date = datetime.now().strftime("%Y-%m-%d")
log_file_path = f"log/{today_date}.log"
tee = ServerTee(log_file_path)
print(f"Logging to: {log_file_path}")

# --- FastAPI initialization ---
app = FastAPI(title="AsyncPG FastAPI Server")


# Add CORS middleware
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Include your asyncpg-based data routes
# app.include_router(xxxx_router)

# --- Thread pool for blocking tasks ---
executor = ThreadPoolExecutor(max_workers=5)

# Example blocking function
def blocking_task(seconds: int):
    print(f"Start blocking task for {seconds} seconds")
    time.sleep(seconds)
    return f"Task finished after {seconds} seconds"

# Async endpoint that runs the blocking task in a background thread
@app.get("/run-task/{seconds}")
async def run_task(seconds: int):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, blocking_task, seconds)
    return {"result": result}

# Health check endpoint
@app.get("/status")
async def status():
    return {"status": "ok"}

# --- Entrypoint for local run ---
if __name__ == "__main__":
    import uvicorn
    backend_port = int(os.environ.get("BACKEND_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=backend_port, reload=True)
