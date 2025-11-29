from dotenv import load_dotenv
from schemas import State
import operator
from typing import Annotated, List, Literal, TypedDict, Union
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

class FactCheckEntry(BaseModel):
   
    """Represents the evaluation of a single claim."""
    
    original_claiming_sentense: str = Field(..., description="The exact text of the claim extracted from the raw input, typically surrounded by ***."
    )
    fact_check_bullet_points: List[str] = Field(
        ..., 
        description="A list of specific evidence points from the Final Report that support, refute, or add context to the claim."
    )
    claim_fact_check_status: Literal["True", "False", "Misleading", "Lack of Evidence"] = Field(
        ..., 
        description="The final verdict of the claim based strictly on the provided report."
    )

class FactCheckOutput(BaseModel):
    """The container for the list of verified claims."""
    results: List[FactCheckEntry]




FACT_CHECK_SYSTEM_PROMPT = """You are a Senior Misinformation Analyst and Logic Auditor. 


Your Task:
You will be provided with two inputs:
1. A list of "Extracted Claims" (raw text where specific claims are highlighted with ***).
2. A "Final Verification Report" (authored by a research system).

Your Goal:
Map every claim found in the "Extracted Claims" to the evidence provided in the "Final Verification Report" and generate a structured evaluation object.

Guidelines:
1. **Source of Truth**: You must ONLY use facts present in the "Final Verification Report". Do not use external knowledge. If the report does not mention a claim, the status is "Lack of Evidence".
2. **Claim Identification**: Locate sentences in the "Extracted Claims" input that are surrounded by triple asterisks (e.g., ***Earth is flat***).
3. **Status Definitions**:
   - "True": The report explicitly confirms the claim with evidence.
   - "False": The report explicitly disproves the claim.
   - "Misleading": The claim contains grains of truth but is manipulated, lacks context, or draws an incorrect conclusion according to the report.
   - "Lack of Evidence": The report does not address this specific claim.

4. **Bullet Points**: Your `fact_check_bullet_points` must be concise summaries of the evidence found in the report.

Output strictly in the requested JSON structure.
"""


def evaluate_claims_node(state: State):
    
    llm = ChatOpenAI(model="gpt-5.1", temperature=0)
    final_report_content = state["messages"][-1].content
    
    claims_input = state["claims"]

    structured_llm = llm.with_structured_output(FactCheckOutput)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", FACT_CHECK_SYSTEM_PROMPT),
        ("human", """
        === EXTRACTED CLAIMS ===
        {claims_input}

        === FINAL VERIFICATION REPORT ===
        {report_content}
        
        Generate the structured evaluation.
        """)
    ])

    chain = prompt_template | structured_llm
    
    response: FactCheckOutput = chain.invoke({
        "claims_input": claims_input,
        "report_content": final_report_content
    })
    return {
        "structured_evaluation": [entry.model_dump() for entry in response.results]
    }