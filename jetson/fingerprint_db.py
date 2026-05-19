"""
Quản lý cơ sở dữ liệu vân tay đã đăng ký (enrolled fingerprints).
Lưu trữ ảnh vân tay dưới dạng file .npy trong thư mục enrolled_fingerprints/.
"""
import numpy as np
import os
import json
import time
from config import DB_DIR, MAX_ENROLLED


class FingerprintDB:
    """
    Cơ sở dữ liệu vân tay đã đăng ký.
    Cấu trúc thư mục:
        enrolled_fingerprints/
        ├── db.json                # Metadata: {user_id: {name, enrolled_at, num_samples}}
        ├── user_001/
        │   ├── sample_0.npy
        │   ├── sample_1.npy
        │   └── sample_2.npy
        └── user_002/
            └── ...
    """

    def __init__(self, db_dir=None):
        self.db_dir = db_dir or DB_DIR
        self.meta_path = os.path.join(self.db_dir, "db.json")
        self.metadata = {}
        self._ensure_dir()
        self._load_metadata()

    def _ensure_dir(self):
        os.makedirs(self.db_dir, exist_ok=True)

    def _load_metadata(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"[DB] Đã tải {len(self.metadata)} người dùng từ database.")
        else:
            self.metadata = {}

    def _save_metadata(self):
        with open(self.meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def get_user_count(self):
        return len(self.metadata)

    def list_users(self):
        """Liệt kê tất cả người dùng đã đăng ký."""
        users = []
        for uid, info in self.metadata.items():
            users.append({
                "user_id": uid,
                "name": info.get("name", ""),
                "num_samples": info.get("num_samples", 0),
                "enrolled_at": info.get("enrolled_at", "")
            })
        return users

    def enroll(self, user_id, name, fingerprint_images):
        """
        Đăng ký vân tay mới.

        Args:
            user_id: ID người dùng (str)
            name: Tên người dùng
            fingerprint_images: list các numpy array (ảnh grayscale)
        """
        if self.get_user_count() >= MAX_ENROLLED:
            print(f"[DB] Database đã đầy ({MAX_ENROLLED} người).")
            return False

        if user_id in self.metadata:
            print(f"[DB] User '{user_id}' đã tồn tại. Dùng update() để cập nhật.")
            return False

        user_dir = os.path.join(self.db_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)

        for i, img in enumerate(fingerprint_images):
            np.save(os.path.join(user_dir, f"sample_{i}.npy"), img)

        self.metadata[user_id] = {
            "name": name,
            "num_samples": len(fingerprint_images),
            "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_metadata()
        print(f"[DB] Đã đăng ký '{name}' (ID: {user_id}) với {len(fingerprint_images)} mẫu.")
        return True

    def delete(self, user_id):
        """Xóa vân tay của một người dùng."""
        if user_id not in self.metadata:
            print(f"[DB] Không tìm thấy user '{user_id}'.")
            return False

        import shutil
        user_dir = os.path.join(self.db_dir, user_id)
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)

        del self.metadata[user_id]
        self._save_metadata()
        print(f"[DB] Đã xóa user '{user_id}'.")
        return True

    def load_all_enrolled(self):
        """
        Tải tất cả ảnh vân tay đã đăng ký vào RAM.

        Returns:
            dict {user_id: [numpy_array, ...]}
        """
        enrolled = {}
        for user_id in self.metadata:
            user_dir = os.path.join(self.db_dir, user_id)
            if not os.path.isdir(user_dir):
                continue
            images = []
            num_samples = self.metadata[user_id].get("num_samples", 0)
            for i in range(num_samples):
                npy_path = os.path.join(user_dir, f"sample_{i}.npy")
                if os.path.exists(npy_path):
                    images.append(np.load(npy_path))
            enrolled[user_id] = images
        return enrolled

    def get_user_images(self, user_id):
        """Tải ảnh vân tay của 1 người dùng."""
        user_dir = os.path.join(self.db_dir, user_id)
        if not os.path.isdir(user_dir):
            return []
        images = []
        num_samples = self.metadata.get(user_id, {}).get("num_samples", 0)
        for i in range(num_samples):
            npy_path = os.path.join(user_dir, f"sample_{i}.npy")
            if os.path.exists(npy_path):
                images.append(np.load(npy_path))
        return images

    def get_user_name(self, user_id):
        """Lấy tên người dùng từ ID."""
        if user_id in self.metadata:
            return self.metadata[user_id].get("name", user_id)
        return None
