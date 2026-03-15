from typing import Optional, Literal

from pydantic import BaseModel


class AlgoBase(BaseModel):
    name: str = None
    num_process: int = 5
    device: str = 'cpu'
    seed: Optional[int] = 0
    preprocessing: Optional[str] = None
