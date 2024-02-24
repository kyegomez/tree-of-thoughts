# BESTSearch

import pytest
import json
from tests.tot.montecarlosearch import mock_model
from tree_of_thoughts import BESTSearch
from unittest.mock import mock_open, patch

def test_save_tree_to_json_creates_correct_file(monkeypatch):
    test_model = mock_model()  # You should provide a suitable mock for the model
    tree = BESTSearch(test_model)
    monkeypatch.setattr('os.makedirs', lambda x, exist_ok=True: None)

    m = mock_open()
    with patch('builtins.open', m, create=True):
        tree.save_tree_to_json('/fake/dir/fake_tree.json')
        m.assert_called_once_with('/fake/dir/fake_tree.json', 'w')
        handle = m()
        expected_tree_json = json.dumps(tree.tree, indent=4)
        handle.write.assert_called_with(expected_tree_json)
