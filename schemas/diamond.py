from enum import Enum

from pydantic import BaseModel, ConfigDict


class Cut(str, Enum):
    IDEAL = 'Ideal'
    PREMIUM = 'Premium'
    VERY_GOOD = 'Very Good'
    GOOD = 'Good'
    FAIR = 'Fair'

class Color(str, Enum):
    D = 'D'
    E = 'E'
    F = 'F'
    G = 'G'
    H = 'H'
    I = 'I'
    J = 'J'

class Clarity(str, Enum):
    IF = 'IF'
    VVS1 = 'VVS1'
    VVS2 = 'VVS2'
    VS1 = 'VS1'
    VS2 = 'VS2'
    SI1 = 'SI1'
    SI2 = 'SI2'
    I1 = 'I1'

class Diamond(BaseModel):
    carat: float
    cut: Cut
    color: Color
    clarity: Clarity
    depth: float
    table: float
    x: float
    y: float
    z: float
    
    model_config = ConfigDict(extra='forbid')
