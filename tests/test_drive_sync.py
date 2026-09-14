"""
Google Drive 同期モジュールのテスト
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import sqlite3

from src.drive_sync import GoogleDriveSync
from src.database import DatabaseManager


class TestGoogleDriveSync(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_alphasignal.db")
        # テスト用SQLite DB作成
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE test (id INT, val TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'hello')")
            conn.commit()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_init_default(self):
        sync = GoogleDriveSync(enabled=False)
        self.assertFalse(sync.enabled)
        self.assertFalse(sync.is_available())

    def test_sync_status_local_only(self):
        sync = GoogleDriveSync(enabled=False)
        status = sync.get_sync_status(self.db_path)
        self.assertFalse(status["enabled"])
        self.assertTrue(status["local_exists"])
        self.assertGreater(status["local_size"], 0)

    @patch("src.drive_sync.GoogleDriveSync._get_service")
    def test_get_remote_file_info(self, mock_get_service):
        mock_service = MagicMock()
        mock_files = MagicMock()
        mock_list = MagicMock()
        mock_list.execute.return_value = {
            "files": [{"id": "test_id_123", "name": "alphasignal.db", "size": "100", "modifiedTime": "2026-09-14T00:00:00Z"}]
        }
        mock_files.list.return_value = mock_list
        mock_service.files.return_value = mock_files
        mock_get_service.return_value = mock_service

        sync = GoogleDriveSync(enabled=True)
        info = sync.get_remote_file_info("alphasignal.db")
        self.assertIsNotNone(info)
        self.assertEqual(info["id"], "test_id_123")

    @patch("src.drive_sync.GoogleDriveSync._get_service")
    def test_push_update(self, mock_get_service):
        mock_service = MagicMock()
        mock_files = MagicMock()
        # ファイル検索で既存ファイルが見つかるケース
        mock_list = MagicMock()
        mock_list.execute.return_value = {
            "files": [{"id": "existing_id", "name": "test_alphasignal.db"}]
        }
        mock_files.list.return_value = mock_list
        # update が成功するケース
        mock_update = MagicMock()
        mock_update.execute.return_value = {"id": "existing_id", "name": "test_alphasignal.db"}
        mock_files.update.return_value = mock_update

        mock_service.files.return_value = mock_files
        mock_get_service.return_value = mock_service

        sync = GoogleDriveSync(enabled=True)
        result = sync.push(self.db_path)
        self.assertTrue(result)
        mock_files.update.assert_called_once()

    def test_database_manager_with_sync_disabled(self):
        # sync_drive=False で DatabaseManager が正常にローカル動作することを確認
        db = DatabaseManager(db_path=self.db_path, sync_drive=False)
        self.assertFalse(db.sync_drive_enabled)
        status = db.get_sync_status()
        self.assertFalse(status["enabled"])


if __name__ == "__main__":
    unittest.main()
