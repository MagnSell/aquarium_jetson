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
