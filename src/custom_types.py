from pydantic import BaseModel

class LlmConfig(BaseModel):
    top_p: float 
    temperature: float 
    max_tokens: int
    llm: str 
