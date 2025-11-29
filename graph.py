from langgraph.graph.state import END, StateGraph
from claim_identifier import claim_identifier
from deep_cross_reference import deep_cross_reference
from extraction_agent import extraction_agent
from schemas import State
from structured_output import evaluate_claims_node 

builder = StateGraph(State)

builder.set_entry_point("extract")
builder.add_node("extract", extraction_agent)
builder.add_node("claim_identifier", claim_identifier)
builder.add_node("evaluate_claim_node",evaluate_claims_node)
builder.add_node("deep_cross_reference", deep_cross_reference)

builder.add_edge("extract", "claim_identifier")
builder.add_edge("claim_identifier", "deep_cross_reference")
builder.add_edge("deep_cross_reference","evaluate_claim_node")
builder.add_edge("evaluate_claim_node", END)

graph = builder.compile()
