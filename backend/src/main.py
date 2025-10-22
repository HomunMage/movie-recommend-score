# src/main.py

import os
import time
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx  # async HTTP client

# Local imports
from ServerTee import ServerTee

from route import health_router, movies_router


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

app.include_router(health_router)
app.include_router(movies_router)

# --- Thread pool for blocking tasks ---
executor = ThreadPoolExecutor(max_workers=5)


# Example blocking function
def blocking_task(seconds: int):
    print(f"Start blocking task for {seconds} seconds")
    time.sleep(seconds)
    return f"Task finished after {seconds} seconds"


@app.get("/run-task/{seconds}")
async def run_task(seconds: int):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, blocking_task, seconds)
    return {"result": result}


@app.get("/status")
async def status():
    return {"status": "ok"}


# --------------------------------------------------------------------
# 🧠 New route: call Bento API (internal port 3000)
# --------------------------------------------------------------------
@app.post("/encode_user")
async def encode_user(payload: dict):
    """
    Forward user_ratings to the Bento service (http://bento:3000/encode_user)
    """
    bento_url = os.getenv("BENTO_URL", "http://bento:3000/encode_user")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(bento_url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Bento service unreachable: {e}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


# --- Entrypoint for local run ---
if __name__ == "__main__":
    import uvicorn
    backend_port = int(os.environ.get("BACKEND_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=backend_port, reload=True)
