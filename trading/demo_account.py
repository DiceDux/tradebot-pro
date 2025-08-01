"""
کلاس مدیریت حساب معاملاتی دمو
"""
import logging
from datetime import datetime

logger = logging.getLogger("demo_account")

class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'

class DemoAccount:
    """کلاس مدیریت حساب معاملاتی دمو"""
    
    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.transactions = []
        self.start_time = datetime.now()
        logger.info(f"Demo account created with initial balance of {initial_balance} USDT")
    
    def deposit(self, amount):
        """واریز به حساب"""
        if amount <= 0:
            return False
            
        self.balance += amount
        self.transactions.append({
            'type': 'deposit',
            'amount': amount,
            'balance': self.balance,
            'timestamp': datetime.now()
        })
        
        logger.info(f"Deposit: +{amount:.4f} USDT, Balance: {self.balance:.4f} USDT")
        return True
    
    def withdraw(self, amount):
        """برداشت از حساب"""
        if amount <= 0 or amount > self.balance:
            logger.warning(f"Withdrawal failed: {amount:.4f} USDT (insufficient balance: {self.balance:.4f} USDT)")
            return False
            
        self.balance -= amount
        self.transactions.append({
            'type': 'withdraw',
            'amount': amount,
            'balance': self.balance,
            'timestamp': datetime.now()
        })
        
        logger.info(f"Withdraw: -{amount:.4f} USDT, Balance: {self.balance:.4f} USDT")
        return True
    
    def update_balance(self, profit_loss):
        """به‌روزرسانی موجودی با سود/ضرر"""
        self.balance += profit_loss
        
        if profit_loss >= 0:
            tx_type = 'profit'
            logger.info(f"Profit: +{profit_loss:.4f} USDT, Balance: {self.balance:.4f} USDT")
        else:
            tx_type = 'loss'
            logger.info(f"Loss: {profit_loss:.4f} USDT, Balance: {self.balance:.4f} USDT")
        
        self.transactions.append({
            'type': tx_type,
            'amount': profit_loss,
            'balance': self.balance,
            'timestamp': datetime.now()
        })
        
        return self.balance
    
    def get_balance(self):
        """دریافت موجودی فعلی"""
        return self.balance
    
    def get_profit_loss(self):
        """دریافت سود/ضرر کل"""
        return self.balance - self.initial_balance
    
    def get_profit_loss_percentage(self):
        """دریافت درصد سود/ضرر کل"""
        return (self.balance / self.initial_balance - 1) * 100
    
    def get_transaction_history(self):
        """دریافت تاریخچه تراکنش‌ها"""
        return self.transactions
    
    def print_status(self):
        """نمایش وضعیت حساب دمو"""
        profit_loss = self.get_profit_loss()
        profit_loss_pct = self.get_profit_loss_percentage()
        runtime = datetime.now() - self.start_time
        hours = runtime.total_seconds() / 3600
        
        # تعداد تراکنش‌های مختلف
        profit_count = len([tx for tx in self.transactions if tx['type'] == 'profit'])
        loss_count = len([tx for tx in self.transactions if tx['type'] == 'loss'])
        
        print("\n" + "="*50)
        print(f"{Colors.CYAN}DEMO ACCOUNT STATUS{Colors.RESET}")
        print(f"Current balance: {self.balance:.2f} USDT")
        
        if profit_loss >= 0:
            print(f"Total P&L: {Colors.GREEN}+{profit_loss:.2f} USDT ({profit_loss_pct:.2f}%){Colors.RESET}")
        else:
            print(f"Total P&L: {Colors.RED}{profit_loss:.2f} USDT ({profit_loss_pct:.2f}%){Colors.RESET}")
            
        if hours > 0:
            hourly_return = profit_loss_pct / hours
            print(f"Hourly return: {hourly_return:+.2f}% per hour")
            print(f"Annualized return: {hourly_return * 24 * 365:+.2f}% per year")
            
        print(f"Profitable trades: {profit_count}")
        print(f"Loss-making trades: {loss_count}")
        print(f"Runtime: {runtime}")
        print("="*50)
