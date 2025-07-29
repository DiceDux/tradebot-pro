import warnings
import sys
import numpy as np
import logging
import os
from datetime import datetime

# تنظیم لاگر به حالت سکوت برای هشدارهای غیر مهم
logging.basicConfig(level=logging.ERROR)

# غیرفعال کردن تمام هشدارها
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# مسیردهی خروجی استاندارد برای اجرای بک‌تست
class WarningFilter:
    def __init__(self):
        # مسیر فایل لاگ
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.log_path = os.path.join(self.log_dir, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.log_file = open(self.log_path, "w")
        self.original_stdout = sys.stdout
    
    def write(self, message):
        # فیلتر پیام‌های ناخواسته
        unwanted_messages = [
            "Using cached feature selection",
            "UserWarning:",
            "DeprecationWarning:",
            "FutureWarning:",
            "pandas only supports SQLAlchemy"
        ]
        
        # اگر پیام حاوی متن‌های ناخواسته نباشد یا پیام مهم بک‌تست باشد، نمایش بده
        if not any(unwanted in message for unwanted in unwanted_messages):
            self.original_stdout.write(message)
            self.log_file.write(message)
            self.log_file.flush()
    
    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()

# فقط وقتی مستقیماً اجرا می‌شود فعال شود (اگر import شود فعال نشود)
if __name__ == "__main__":
    sys.stdout = WarningFilter()
