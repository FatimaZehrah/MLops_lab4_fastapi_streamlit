from pydantic import BaseModel


class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class IrisResponse(BaseModel):
    prediction: int
    confidence: float
