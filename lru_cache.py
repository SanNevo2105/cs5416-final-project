import sqlite3
import json
import threading


class LRUCache:
    def __init__(self, capacity: int, db_path: str):
        self.capacity = capacity
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._create_table()

    def _create_table(self):
        with self.lock:
            with self.conn:
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        last_access TIMESTAMP
                    )
                """
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_last_access ON cache(last_access)"
                )

    def get(self, key):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT value FROM cache WHERE key = ?", (key,))
            row = cursor.fetchone()

            if row:
                # Update last_access using SQLite function for high precision timestamp
                with self.conn:
                    self.conn.execute(
                        "UPDATE cache SET last_access = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW') WHERE key = ?",
                        (key,),
                    )
                return json.loads(row[0])
            return None

    def set(self, key, value):
        # Serialize the data
        json_value = json.dumps(value)

        with self.lock:
            with self.conn:
                # Insert or replace the item
                self.conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, last_access) VALUES (?, ?, STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))",
                    (key, json_value),
                )

                # Check capacity and evict if necessary
                cursor = self.conn.execute("SELECT COUNT(*) FROM cache")
                count = cursor.fetchone()[0]

                if count > self.capacity:
                    # Evict the least recently used item
                    self.conn.execute(
                        "DELETE FROM cache WHERE key = (SELECT key FROM cache ORDER BY last_access ASC LIMIT 1)"
                    )

    def close(self):
        with self.lock:
            self.conn.close()
