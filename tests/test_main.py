import pytest
from src.ignore_file import main

@pytest.mark.parametrize(
    ["inp", "expected"],
    [
        (1,2),
        [2,3],
        [3,4]
    ]
)
def test_main(inp:int, expected:int)->None:
    assert main(inp)==expected