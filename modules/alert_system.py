import time

class AlertSystem:
    def __init__(self):
        self.alert_log = []

    def log_alert(self, message: str):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        alert = f"[{timestamp}] {message}"
        self.alert_log.append(alert)
        return alert
