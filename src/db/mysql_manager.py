"""
RESINET-IDAM: MySQL Database Manager (src/db/mysql_manager.py)
Implements strictly MySQL database connectivity for audit logging, detection alerts,
and benchmark archives. Adheres to policy: NEVER silently substitute SQLite.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    HAS_MYSQL_CONNECTOR = True
except ImportError:
    HAS_MYSQL_CONNECTOR = False
    MySQLError = Exception


class MySQLManager:
    """Manages direct MySQL connectivity for RESINET-IDAM."""

    def __init__(self):
        self.host = os.environ.get("MYSQL_HOST", "localhost")
        self.port = int(os.environ.get("MYSQL_PORT", 3306))
        self.user = os.environ.get("MYSQL_USER", "root")
        self.password = os.environ.get("MYSQL_PASSWORD", "")
        self.database = os.environ.get("MYSQL_DATABASE", "resinet_db")

        # In-memory buffer for alerts when MySQL service is unreachable
        self._memory_alert_buffer: List[Dict[str, Any]] = []
        self._memory_audit_buffer: List[Dict[str, Any]] = []

    def get_connection(self):
        """Attempts to establish a live connection to MySQL."""
        if not HAS_MYSQL_CONNECTOR:
            raise RuntimeError("mysql-connector-python is not installed.")
        return mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            connect_timeout=2,
        )

    def check_connection_status(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Tests whether MySQL is reachable on the configured host/port.
        Returns: (is_connected, status_message, config_info)
        """
        info = {
            "driver": "mysql.connector",
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "buffered_alerts": len(self._memory_alert_buffer),
            "buffered_audits": len(self._memory_audit_buffer),
        }

        if not HAS_MYSQL_CONNECTOR:
            return False, "Driver mysql-connector-python not found", info

        try:
            conn = self.get_connection()
            if conn.is_connected():
                cursor = conn.cursor()
                cursor.execute("SELECT VERSION()")
                db_version = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                info["version"] = db_version
                return True, f"Connected to MySQL {db_version} ({self.host}:{self.port}/{self.database})", info
        except Exception as e:
            return False, f"MySQL Standby / Unreachable ({self.host}:{self.port}) - {str(e)}", info

        return False, "Disconnected", info

    def log_audit(self, event_type: str, user_action: str, details: str = "", status: str = "SUCCESS") -> bool:
        """Logs user interactions or system events to MySQL, buffering if offline."""
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "user_action": user_action,
            "details": details,
            "status": status,
        }

        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            query = """
            INSERT INTO resinet_audit_logs (event_type, user_action, details, status)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (event_type, user_action, details, status))
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Exception:
            # Buffer in memory while MySQL is in standby (NO SQLite substitution!)
            self._memory_audit_buffer.append(record)
            if len(self._memory_audit_buffer) > 500:
                self._memory_audit_buffer.pop(0)
            return False

    def log_detection_alert(
        self,
        flow_id: str,
        detection_engine: str,
        predicted_class: str,
        confidence_score: float,
        anomaly_score: float,
        threat_severity: str,
        dest_port: int = 0,
    ) -> bool:
        """Stores detected malicious flows into MySQL, buffering if offline."""
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "flow_id": str(flow_id),
            "detection_engine": detection_engine,
            "predicted_class": predicted_class,
            "confidence_score": float(confidence_score),
            "anomaly_score": float(anomaly_score),
            "threat_severity": threat_severity,
            "dest_port": int(dest_port),
        }

        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            query = """
            INSERT INTO resinet_detection_alerts 
            (flow_id, detection_engine, predicted_class, confidence_score, anomaly_score, threat_severity, dest_port)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                record["flow_id"],
                record["detection_engine"],
                record["predicted_class"],
                record["confidence_score"],
                record["anomaly_score"],
                record["threat_severity"],
                record["dest_port"],
            ))
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Exception:
            self._memory_alert_buffer.append(record)
            if len(self._memory_alert_buffer) > 1000:
                self._memory_alert_buffer.pop(0)
            return False

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent alerts from MySQL or memory buffer."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM resinet_detection_alerts ORDER BY alert_id DESC LIMIT %s", (limit,))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            return rows
        except Exception:
            return list(reversed(self._memory_alert_buffer[-limit:]))

    def get_recent_audits(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent audit logs from MySQL or memory buffer."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM resinet_audit_logs ORDER BY log_id DESC LIMIT %s", (limit,))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            return rows
        except Exception:
            return list(reversed(self._memory_audit_buffer[-limit:]))


# Global Singleton Instance
db_manager = MySQLManager()

