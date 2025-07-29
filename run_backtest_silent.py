#!/usr/bin/env python
"""
اجرای بک‌تست بدون نمایش هشدارها
"""
import sys
import os

# اضافه کردن مسیر ریشه پروژه به مسیرهای پایتون
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# وارد کردن فیلتر هشدارها
from silence_warnings import WarningFilter
sys.stdout = WarningFilter()

# وارد کردن اسکریپت اصلی
from smart_trader_cli import SmartTraderCLI

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" BACKTEST STARTING - ALL WARNINGS SILENCED ")
    print("="*50 + "\n")
    
    bot = SmartTraderCLI()
    results = bot.run_backtest()
    
    print("\n" + "="*50)
    print(" BACKTEST COMPLETED ")
    print("="*50 + "\n")
    
    # خلاصه نتایج
    total_profit = sum(r["profit_loss"] for r in results.values())
    profit_percent = sum(r["profit_percent"] for r in results.values()) / len(results) if results else 0
    
    print(f"Total profit: ${total_profit:.2f} ({profit_percent:.2f}%)")
    for symbol, result in results.items():
        print(f"{symbol}: ${result['profit_loss']:.2f} ({result['profit_percent']:.2f}%) - Trades: {result['total_trades']} - Win rate: {result['win_rate']*100:.1f}%")
