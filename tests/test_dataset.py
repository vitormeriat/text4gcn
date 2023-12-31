import pytest
from text4gcn.datasets import data


def test_gets_data_list_len():
    assert len(data.list()) == 3

