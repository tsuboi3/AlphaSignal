"""
Google Drive 同期モジュール
alphasignal.db を Google Drive 上の指定フォルダと自動同期（アップロード/ダウンロード）します。
"""

import os
import io
import warnings
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# httplib2 や googleapiclient の過剰な警告を抑制
warnings.filterwarnings("ignore", category=UserWarning, module="googleapiclient")

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_FOLDER_ID = "1nDozKuBzTzUk0GKQcD9xAXqAofLIkVcq"
DEFAULT_QUOTA_PROJECT = "kabutanproject"


class GoogleDriveSync:
    """Google Drive とローカル SQLite DB (alphasignal.db) を同期するマネージャー"""

    def __init__(
        self,
        folder_id: Optional[str] = None,
        quota_project: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.folder_id = folder_id or os.getenv("GOOGLE_DRIVE_FOLDER_ID", DEFAULT_FOLDER_ID)
        self.quota_project = quota_project or os.getenv("GOOGLE_DRIVE_QUOTA_PROJECT", DEFAULT_QUOTA_PROJECT)
        
        env_enabled = os.getenv("GOOGLE_DRIVE_SYNC", "true").lower() in ("true", "1", "yes")
        self.enabled = env_enabled if enabled is None else enabled
        
        self._service = None
        self._initialized = False

    def _get_service(self):
        """Google Drive API サービスを初期化・取得"""
        if self._service is not None:
            return self._service

        if not self.enabled:
            return None

        try:
            import google.auth
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            sa_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
            oauth_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRETS_FILE", "credentials.json")
            oauth_token = os.getenv("GOOGLE_OAUTH_TOKEN_FILE", "token.json")
            scopes = ["https://www.googleapis.com/auth/drive"]

            credentials = None

            # 1. サービスアカウントキー (JSON) が指定されている場合
            if sa_file and os.path.exists(sa_file):
                credentials = service_account.Credentials.from_service_account_file(sa_file, scopes=scopes)
            # 2. OAuth 2.0 トークンファイルが既に存在する場合
            elif os.path.exists(oauth_token):
                from google.oauth2.credentials import Credentials
                from google.auth.transport.requests import Request
                credentials = Credentials.from_authorized_user_file(oauth_token, scopes)
                if credentials and credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
            # 3. OAuth 2.0 クライアント認証 JSON が存在する場合 (初回ブラウザ認証)
            elif os.path.exists(oauth_secret):
                from google_auth_oauthlib.flow import InstalledAppFlow
                flow = InstalledAppFlow.from_client_secrets_file(oauth_secret, scopes)
                credentials = flow.run_local_server(port=0)
                with open(oauth_token, "w") as token:
                    token.write(credentials.to_json())
            # 4. Google Cloud ADC (gcloud auth や Cloud Shell 環境)
            else:
                credentials, _ = google.auth.default(scopes=scopes)
                if self.quota_project:
                    credentials = credentials.with_quota_project(self.quota_project)

            self._service = build("drive", "v3", credentials=credentials, cache_discovery=False)
            self._initialized = True
            return self._service
        except Exception as e:
            logger.warning(f"Google Drive APIの初期化に失敗しました (ローカルモードで続行します): {e}")
            self._initialized = False
            return None

    def is_available(self) -> bool:
        """Google Drive サービスが利用可能かどうか"""
        return self._get_service() is not None

    def get_remote_file_info(self, filename: str = "alphasignal.db") -> Optional[Dict[str, Any]]:
        """Google Drive 上のファイルのメタ情報を取得"""
        service = self._get_service()
        if not service:
            return None

        try:
            query = f"'{self.folder_id}' in parents and name = '{filename}' and trashed = false"
            results = service.files().list(
                q=query,
                fields="files(id, name, mimeType, size, modifiedTime)"
            ).execute()
            files = results.get("files", [])
            if files:
                return files[0]
            return None
        except Exception as e:
            logger.warning(f"Google Drive 上のファイル検索に失敗しました: {e}")
            return None

    def pull(self, local_path: str = "alphasignal.db", force: bool = False) -> bool:
        """
        Google Drive から alphasignal.db をダウンロードしてローカルを更新
        :param local_path: ダウンロード先のローカルパス
        :param force: 更新日時の比較を無視して強制ダウンロードするかどうか
        :return: 成功したか、ダウンロードが行われたかどうか
        """
        service = self._get_service()
        if not service:
            return False

        filename = os.path.basename(local_path)
        remote_info = self.get_remote_file_info(filename)
        if not remote_info:
            logger.info(f"Google Drive 上に {filename} が見つかりません。")
            return False

        try:
            from googleapiclient.http import MediaIoBaseDownload

            # 更新日時の比較
            if not force and os.path.exists(local_path):
                remote_mtime_str = remote_info.get("modifiedTime")
                if remote_mtime_str:
                    remote_mtime = datetime.fromisoformat(remote_mtime_str.replace("Z", "+00:00"))
                    local_mtime = datetime.fromtimestamp(os.path.getmtime(local_path), tz=timezone.utc)
                    if local_mtime >= remote_mtime:
                        return True

            file_id = remote_info["id"]
            request = service.files().get_media(fileId=file_id)

            temp_local_path = f"{local_path}.tmp_download"
            with open(temp_local_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()

            # アトミックにファイルを置換
            os.replace(temp_local_path, local_path)
            print(f"  ✓ Google Driveから最新の {filename} を同期（ダウンロード）しました。")
            return True

        except Exception as e:
            logger.error(f"Google Drive からのダウンロード中にエラーが発生しました: {e}")
            temp_local_path = f"{local_path}.tmp_download"
            if os.path.exists(temp_local_path):
                try:
                    os.remove(temp_local_path)
                except OSError:
                    pass
            return False

    def push(self, local_path: str = "alphasignal.db") -> bool:
        """
        ローカルの alphasignal.db を Google Drive へアップロード・更新
        :param local_path: アップロード元のローカルパス
        :return: 成功したかどうか
        """
        service = self._get_service()
        if not service:
            return False

        if not os.path.exists(local_path):
            logger.warning(f"アップロード対象のファイルが存在しません: {local_path}")
            return False

        filename = os.path.basename(local_path)
        remote_info = self.get_remote_file_info(filename)

        try:
            from googleapiclient.http import MediaFileUpload

            media = MediaFileUpload(local_path, mimetype="application/x-sqlite3", resumable=True)

            if remote_info:
                file_id = remote_info["id"]
                updated = service.files().update(
                    fileId=file_id,
                    media_body=media,
                    fields="id, name, modifiedTime, size"
                ).execute()
                print(f"  ✓ Google Drive 上の {filename} を更新しました (ID: {updated.get('id')})")
            else:
                file_metadata = {
                    "name": filename,
                    "parents": [self.folder_id]
                }
                created = service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields="id, name, modifiedTime, size"
                ).execute()
                print(f"  ✓ Google Drive フォルダに {filename} を新規設置しました (ID: {created.get('id')})")

            return True
        except Exception as e:
            logger.error(f"Google Drive へのアップロード中にエラーが発生しました: {e}")
            return False

    def get_sync_status(self, local_path: str = "alphasignal.db") -> Dict[str, Any]:
        """同期状態の詳細情報を取得"""
        filename = os.path.basename(local_path)
        status = {
            "enabled": self.enabled,
            "available": self.is_available(),
            "folder_id": self.folder_id,
            "quota_project": self.quota_project,
            "remote_exists": False,
            "remote_id": None,
            "remote_size": None,
            "remote_modified": None,
            "local_exists": os.path.exists(local_path),
            "local_size": os.path.getsize(local_path) if os.path.exists(local_path) else 0,
            "local_modified": None,
        }

        if status["local_exists"]:
            local_mtime = datetime.fromtimestamp(os.path.getmtime(local_path), tz=timezone.utc)
            status["local_modified"] = local_mtime.isoformat()

        if status["available"]:
            remote_info = self.get_remote_file_info(filename)
            if remote_info:
                status["remote_exists"] = True
                status["remote_id"] = remote_info.get("id")
                status["remote_size"] = remote_info.get("size")
                status["remote_modified"] = remote_info.get("modifiedTime")

        return status
