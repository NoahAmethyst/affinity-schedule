from dataclasses import dataclass
from typing import Optional

import pytest

@dataclass
class Demo:
    value: Optional[int]

def test_demo():
    print(f'{Demo(1).__dict__}')
