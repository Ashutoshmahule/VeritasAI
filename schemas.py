from typing import Annotated

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import List

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    extracted_info: list[AnyMessage]
    claim_reasoning: list[str]
    claims:str
    structured_evaluation: List[dict]
