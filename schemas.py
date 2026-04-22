from pydantic import BaseModel, field_validator
class PredictRequest(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

    @field_validator("*", mode="before")
    @classmethod
    def check_finite(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("Must be a number")
        if v != v or abs(v) == float("inf"):
            raise ValueError("Must be a finite number")
        return float(v)

class RiskFactor(BaseModel):
    label: str
    value: str
    level: str

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    risk_factors: list[RiskFactor]