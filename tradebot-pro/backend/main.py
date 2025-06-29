from fastapi import FastAPI
from backend.api import bot

app = FastAPI(title="TradeBot Pro API")

app.include_router(bot.router, prefix="/api/bot")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # یا فقط ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)