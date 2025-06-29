from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Account(Base):
    __tablename__ = 'accounts'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), unique=True)
    exchange = Column(String(32))
    api_key = Column(String(256))
    api_secret = Column(String(256))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Symbol(Base):
    __tablename__ = 'symbols'
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(64), unique=True)
    is_active = Column(Boolean, default=True)

class BotStatus(Base):
    __tablename__ = 'bot_status'
    id = Column(Integer, primary_key=True, index=True)
    status = Column(String(32), default="stopped")
    last_changed = Column(DateTime, default=datetime.utcnow)