from pydantic import BaseModel
from datetime import datetime
import uuid

class Node_Measurement(BaseModel):
    uuid: uuid.UUID
    node_id: int
    timestamp: datetime
    temperature: float
    pH : float
    dissolved_oxygen: float

class Fish(BaseModel):
    uuid: uuid.UUID
    timestamp: datetime
    fish_id: int
    x_position: float
    y_position: float
    z_position: float
    x_velocity: float
    y_velocity: float
    z_velocity: float