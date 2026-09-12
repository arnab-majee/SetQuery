from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural-language query for SatQuery."
    , min_length=1)
   
class AnalyzeRequest(BaseModel):
    query: str= Field(..., description="Natural-language remote-sensing analysis query.", min_length=1)