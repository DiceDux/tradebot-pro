"""
راه‌اندازی کننده اصلی سیستم معاملاتی پیشرفته
"""
import argparse
import time
import os
import json
from datetime import datetime

from orchestrator.trading_orchestrator import TradingOrchestrator
from feature_store.feature_calculator import FeatureCalculator
from feature_store.feature_monitor import FeatureMonitor

def initialize_database():
    """ایجاد جداول مورد نیاز در دیتابیس"""
    from feature_store.feature_database import FeatureDatabase
    db = FeatureDatabase()
    return db

def start_feature_calculator(symbols=['BTCUSDT', 'ETHUSDT'], interval=1):
    """راه‌اندازی محاسبه‌کننده فیچر"""
    calculator = FeatureCalculator(symbols, update_interval=interval)
    calculator.run()
    return calculator

def start_feature_monitor(symbol='BTCUSDT', interval=5):
    """راه‌اندازی نمایشگر فیچر"""
    monitor = FeatureMonitor(symbol, refresh_rate=interval)
    monitor.run()
    return monitor

def start_trading_system(symbols=['BTCUSDT', 'ETHUSDT']):
    """راه‌اندازی سیستم معاملاتی"""
    orchestrator = TradingOrchestrator(symbols)
    orchestrator.initialize()
    orchestrator.start()
    return orchestrator

def main():
    parser = argparse.ArgumentParser(description="Enhanced Smart Trading Bot")
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'], help='Symbols to trade')
    parser.add_argument('--feature-calc', action='store_true', help='Run feature calculator only')
    parser.add_argument('--monitor', action='store_true', help='Run feature monitor')
    parser.add_argument('--trading', action='store_true', help='Run trading system')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol for feature monitor')
    parser.add_argument('--interval', type=int, default=1, help='Update interval in seconds')
    
    args = parser.parse_args()
    
    # ایجاد دایرکتوری‌های مورد نیاز
    os.makedirs('model', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("====== Enhanced Smart Trading Bot ======")
    print("Initializing database...")
    db = initialize_database()
    
    services = []
    
    if args.feature_calc:
        print(f"Starting feature calculator for {args.symbols} with {args.interval}s interval...")
        calculator = start_feature_calculator(args.symbols, args.interval)
        services.append(calculator)
    
    if args.monitor:
        print(f"Starting feature monitor for {args.symbol} with {args.interval}s refresh rate...")
        monitor = start_feature_monitor(args.symbol, args.interval)
        services.append(monitor)
    
    if args.trading:
        print(f"Starting trading system for {args.symbols}...")
        orchestrator = start_trading_system(args.symbols)
        services.append(orchestrator)
    
    if not (args.feature_calc or args.monitor or args.trading):
        # اجرای همه سرویس‌ها با تنظیمات پیش‌فرض
        print("Starting all services with default settings...")
        calculator = start_feature_calculator(args.symbols, args.interval)
        services.append(calculator)
        
        orchestrator = start_trading_system(args.symbols)
        services.append(orchestrator)
        
        monitor = start_feature_monitor(args.symbols[0]
