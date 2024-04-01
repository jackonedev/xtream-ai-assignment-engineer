from pydantic import BaseModel, ConfigDict

class Diamond(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float
    
    model_config = ConfigDict(extra='forbid')
