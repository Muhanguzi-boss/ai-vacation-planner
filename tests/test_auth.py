import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api.routes.auth import get_current_user


class AuthDependencyTests(unittest.TestCase):
    def test_get_current_user_returns_user_for_valid_bearer_token(self):
        fake_user = SimpleNamespace(id=1, username="alice", email="alice@example.com", hashed_password="hash")
        db = unittest.mock.MagicMock()
        db.query.return_value.filter.return_value.first.return_value = fake_user

        with patch("app.api.routes.auth.decode_token", return_value={"sub": "1"}):
            user = get_current_user(token="valid-token", db=db)

        self.assertEqual(user.id, 1)
        self.assertEqual(user.username, "alice")

    def test_get_current_user_supports_legacy_username_subjects(self):
        fake_user = SimpleNamespace(id=2, username="bob", email="bob@example.com", hashed_password="hash")
        db = unittest.mock.MagicMock()
        db.query.return_value.filter.return_value.first.return_value = fake_user

        with patch("app.api.routes.auth.decode_token", return_value={"sub": "bob"}):
            user = get_current_user(token="legacy-token", db=db)

        self.assertEqual(user.id, 2)
        self.assertEqual(user.username, "bob")


if __name__ == "__main__":
    unittest.main()
