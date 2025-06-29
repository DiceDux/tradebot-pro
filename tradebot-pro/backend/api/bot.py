from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from backend.models.models import Account, BotStatus
from backend.utils.db import get_db

router = APIRouter()

@router.get("/status")
def get_bot_status(db: Session = Depends(get_db)):
    status = db.query(BotStatus).first()
    return {"status": status.status if status else "unknown"}

@router.post("/status")
def set_bot_status(status: str, db: Session = Depends(get_db)):
    bot = db.query(BotStatus).first()
    if not bot:
        bot = BotStatus(status=status)
        db.add(bot)
    else:
        bot.status = status
    db.commit()
    return {"status": bot.status}

@router.get("/accounts")
def list_accounts(db: Session = Depends(get_db)):
    accounts = db.query(Account).all()
    return [ {"id": acc.id, "name": acc.name, "exchange": acc.exchange, "is_active": acc.is_active} for acc in accounts ]

@router.post("/accounts")
def add_account(name: str, exchange: str, api_key: str, api_secret: str, db: Session = Depends(get_db)):
    account = Account(name=name, exchange=exchange, api_key=api_key, api_secret=api_secret)
    db.add(account)
    db.commit()
    return { "ok": True, "id": account.id }