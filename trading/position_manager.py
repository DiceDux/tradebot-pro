"""
کلاس مدیریت موقعیت‌های معاملاتی
"""
import logging
import uuid
from datetime import datetime
import json

logger = logging.getLogger("position_manager")

class PositionManager:
    """کلاس مدیریت موقعیت‌های معاملاتی"""
    
    def __init__(self):
        self.positions = {}  # سیمبل -> موقعیت
        self.position_history = []  # تاریخچه موقعیت‌ها
    
    def open_position(self, symbol, position_type, entry_price, size, tp_levels=None, tp_volumes=None, stop_loss=None):
        """
        باز کردن یک موقعیت جدید
        
        Args:
            symbol: نماد معاملاتی
            position_type: نوع موقعیت ('buy' یا 'sell')
            entry_price: قیمت ورود
            size: اندازه موقعیت
            tp_levels: سطوح تیک پرافیت (آرایه)
            tp_volumes: حجم هر سطح تیک پرافیت (آرایه)
            stop_loss: سطح استاپ لاس
            
        Returns:
            str: شناسه موقعیت
        """
        # بررسی پارامترها
        if position_type not in ['buy', 'sell']:
            logger.error(f"Invalid position type: {position_type}")
            return None
            
        if size <= 0:
            logger.error(f"Invalid position size: {size}")
            return None
            
        # بررسی وجود موقعیت قبلی برای این نماد
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return None
            
        # تنظیم مقادیر پیش‌فرض
        if tp_levels is None:
            # تنظیم پیش‌فرض بر اساس نوع موقعیت
            if position_type == 'buy':
                tp_levels = [entry_price * 1.02]  # سود 2%
            else:
                tp_levels = [entry_price * 0.98]  # سود 2%
                
        if tp_volumes is None:
            tp_volumes = [1.0] * len(tp_levels)  # کل موقعیت در یک سطح
            
        if stop_loss is None:
            # تنظیم پیش‌فرض بر اساس نوع موقعیت
            if position_type == 'buy':
                stop_loss = entry_price * 0.98  # ضرر 2%
            else:
                stop_loss = entry_price * 1.02  # ضرر 2%
                
        # ایجاد شناسه منحصر به فرد
        position_id = str(uuid.uuid4())
        
        # ایجاد موقعیت
        position = {
            'id': position_id,
            'symbol': symbol,
            'type': position_type,
            'entry_price': entry_price,
            'current_price': entry_price,
            'size': size,
            'initial_size': size,
            'tp_levels': tp_levels,
            'tp_volumes': tp_volumes,
            'stop_loss': stop_loss,
            'trailing_stop_active': False,
            'open_time': datetime.now(),
            'last_update': datetime.now(),
            'status': 'open'
        }
        
        # ذخیره موقعیت
        self.positions[symbol] = position
        
        # افزودن به تاریخچه
        self.position_history.append({
            'id': position_id,
            'symbol': symbol,
            'type': position_type,
            'entry_price': entry_price,
            'size': size,
            'action': 'open',
            'timestamp': datetime.now()
        })
        
        logger.info(f"Opened {position_type.upper()} position for {symbol} at {entry_price}, size: {size}")
        return position_id
    
    def close_position(self, symbol, close_price=None):
        """
        بستن یک موقعیت
        
        Args:
            symbol: نماد معاملاتی
            close_price: قیمت بستن (اگر ارائه نشود، از قیمت فعلی استفاده می‌شود)
            
        Returns:
            dict: اطلاعات موقعیت بسته شده
        """
        # بررسی وجود موقعیت
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return None
            
        # استفاده از قیمت فعلی اگر قیمت بستن ارائه نشده است
        position = self.positions[symbol]
        if close_price is None:
            close_price = position['current_price']
            
        # محاسبه سود/زیان
        if position['type'] == 'buy':
            pnl = (close_price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - close_price) * position['size']
            
        # بستن موقعیت
        position['close_price'] = close_price
        position['pnl'] = pnl
        position['close_time'] = datetime.now()
        position['status'] = 'closed'
        
        # افزودن به تاریخچه
        self.position_history.append({
            'id': position['id'],
            'symbol': symbol,
            'type': position['type'],
            'entry_price': position['entry_price'],
            'close_price': close_price,
            'size': position['size'],
            'pnl': pnl,
            'action': 'close',
            'timestamp': datetime.now()
        })
        
        # حذف از موقعیت‌های فعلی
        closed_position = self.positions.pop(symbol)
        
        logger.info(f"Closed {position['type'].upper()} position for {symbol} at {close_price}, PnL: {pnl:.4f}")
        return closed_position
    
    def update_market_price(self, symbol, price):
        """
        به‌روزرسانی قیمت بازار برای یک موقعیت
        
        Args:
            symbol: نماد معاملاتی
            price: قیمت فعلی
        """
        if symbol in self.positions:
            self.positions[symbol]['current_price'] = price
            self.positions[symbol]['last_update'] = datetime.now()
            
    def update_stop_loss(self, symbol, stop_loss, trailing_stop_active=False):
        """
        به‌روزرسانی استاپ لاس برای یک موقعیت
        
        Args:
            symbol: نماد معاملاتی
            stop_loss: سطح استاپ لاس جدید
            trailing_stop_active: آیا استاپ لاس تریلینگ فعال است
        """
        if symbol in self.positions:
            self.positions[symbol]['stop_loss'] = stop_loss
            self.positions[symbol]['trailing_stop_active'] = trailing_stop_active
            self.positions[symbol]['last_update'] = datetime.now()
            
            logger.info(f"Updated stop loss for {symbol} to {stop_loss:.4f}, trailing: {trailing_stop_active}")
    
    def execute_partial_tp(self, symbol, tp_index, tp_size, price):
        """
        اجرای تیک پرافیت جزئی برای یک موقعیت
        
        Args:
            symbol: نماد معاملاتی
            tp_index: شاخص سطح تیک پرافیت
            tp_size: اندازه تیک پرافیت
            price: قیمت اجرای تیک پرافیت
        """
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # بررسی معتبر بودن شاخص
            if tp_index >= len(position['tp_levels']) or tp_index >= len(position['tp_volumes']):
                logger.error(f"Invalid TP index {tp_index} for {symbol}")
                return False
                
            # بررسی اینکه این سطح قبلاً اجرا نشده باشد
            if position['tp_volumes'][tp_index] <= 0:
                logger.warning(f"TP{tp_index+1} for {symbol} already executed")
                return False
                
            # بررسی اندازه معتبر
            if tp_size <= 0 or tp_size > position['size']:
                logger.error(f"Invalid TP size {tp_size} for {symbol} (position size: {position['size']})")
                return False
                
            # محاسبه سود/زیان
            if position['type'] == 'buy':
                pnl = (price - position['entry_price']) * tp_size
            else:
                pnl = (position['entry_price'] - price) * tp_size
                
            # به‌روزرسانی موقعیت
            position['size'] -= tp_size
            position['tp_volumes'][tp_index] = 0.0  # این سطح اجرا شده است
            position['last_update'] = datetime.now()
            
            # افزودن به تاریخچه
            self.position_history.append({
                'id': position['id'],
                'symbol': symbol,
                'type': position['type'],
                'entry_price': position['entry_price'],
                'tp_price': price,
                'tp_index': tp_index,
                'tp_size': tp_size,
                'remaining_size': position['size'],
                'pnl': pnl,
                'action': 'partial_tp',
                'timestamp': datetime.now()
            })
            
            logger.info(f"Executed TP{tp_index+1} for {symbol} at {price}, size: {tp_size}, PnL: {pnl:.4f}")
            return True
        else:
            logger.warning(f"No position found for {symbol}")
            return False
    
    def has_position(self, symbol):
        """بررسی وجود موقعیت برای یک نماد"""
        return symbol in self.positions
    
    def get_position(self, symbol):
        """دریافت اطلاعات موقعیت برای یک نماد"""
        return self.positions.get(symbol)
    
    def get_all_positions(self):
        """دریافت همه موقعیت‌های فعال"""
        return self.positions
    
    def get_position_history(self):
        """دریافت تاریخچه موقعیت‌ها"""
        return self.position_history
    
    def get_position_stats(self):
        """دریافت آمار موقعیت‌ها"""
        # تعداد معاملات
        total_trades = len([p for p in self.position_history if p['action'] == 'close'])
        
        # معاملات سودده و زیان‌ده
        profitable_trades = len([p for p in self.position_history if p['action'] == 'close' and p.get('pnl', 0) > 0])
        losing_trades = len([p for p in self.position_history if p['action'] == 'close' and p.get('pnl', 0) <= 0])
        
        # نرخ برد
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # میانگین سود و زیان
        profits = [p.get('pnl', 0) for p in self.position_history if p['action'] == 'close' and p.get('pnl', 0) > 0]
        losses = [p.get('pnl', 0) for p in self.position_history if p['action'] == 'close' and p.get('pnl', 0) <= 0]
        
        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # ضریب سود/زیان
        profit_factor = abs(sum(profits) / sum(losses)) if sum(losses) != 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def export_history(self, file_path):
        """صدور تاریخچه موقعیت‌ها به فایل JSON"""
        try:
            history_data = []
            for item in self.position_history:
                data = item.copy()
                # تبدیل datetime به رشته
                data['timestamp'] = data['timestamp'].isoformat()
                history_data.append(data)
                
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=4, ensure_ascii=False)
                
            logger.info(f"Position history exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting position history: {e}")
            return False
