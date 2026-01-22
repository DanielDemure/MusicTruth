
import unittest
from unittest.mock import MagicMock, patch
from src.ui.validators import SystemValidator, ValidationResult

class TestSystemValidator(unittest.TestCase):
    def setUp(self):
        self.validator = SystemValidator()

    def test_validate_llm_missing_key(self):
        config_dict = {}
        result = self.validator.validate_llm("openai", config_dict)
        self.assertFalse(result.passed)
        self.assertEqual(result.severity, "critical")

    def test_validate_llm_valid_key(self):
        config_dict = {'api_key': 'sk-test-key-1234567890'}
        result = self.validator.validate_llm("openai", config_dict)
        self.assertTrue(result.passed)

    @patch('os.path.exists')
    def test_validate_inputs_all_exist(self, mock_exists):
        mock_exists.return_value = True
        paths = ["C:/Music/Test.mp3", "C:/Music/Album"]
        result = self.validator.validate_inputs(paths)
        self.assertTrue(result.passed)
        self.assertIn("2 files verified", result.message)

    @patch('os.path.exists')
    def test_validate_inputs_missing_files(self, mock_exists):
        # Mock exist to return False for the second file
        mock_exists.side_effect = [True, False] 
        paths = ["C:/Music/Exists.mp3", "C:/Music/Missing.mp3"]
        result = self.validator.validate_inputs(paths)
        self.assertFalse(result.passed)
        self.assertIn("Missing.mp3", result.message)

if __name__ == '__main__':
    unittest.main()
