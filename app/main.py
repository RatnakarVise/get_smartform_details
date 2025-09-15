from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os, json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# --- Load environment ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

app = FastAPI(title="SAP SmartForm Node Field Mapping Explainer API")


# ---- Input Models ----
class Attribute(BaseModel):
    name: str
    value: str

class Node(BaseModel):
    id: int
    parentId: int
    depth: int
    path: str
    elemName: str
    elemNs: str
    nodeType: str
    attributes: List[Attribute] = []
    textPayload: str

class SmartFormInput(BaseModel):
    formName: str
    system: str
    client: str
    language: Optional[str] = ""
    sourceKind: str
    extractedat: str
    nodes: List[Node]


# ---- LLM Explainer ----
def build_chain(node: Node):
    SYSTEM_MSG = "You are an SAP SmartForm and ABAP expert. Always respond in strict JSON."

    USER_TEMPLATE = """
You are reviewing a SmartForm node.

Explain its **technical mapping and coding** in table format (JSON row).

Return ONLY strict JSON:
{{
  "elemName": "{elemName}",
  "path": "{path}",
  "nodeType": "{nodeType}",
  "attributes": {attributes_json},
  "mapping": "<SAP table.field or variable it is linked to>",
  "coding": "<ABAP code or condition behind it if any>",
  "usage": "<how it is used in the SmartForm page>"
}}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ])

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    parser = JsonOutputParser()
    return prompt | llm | parser


def llm_explain_node(node: Node):
    attrs_json = json.dumps([{"name": a.name, "value": a.value} for a in node.attributes], ensure_ascii=False)
    chain = build_chain(node)
    return chain.invoke({
        "elemName": node.elemName,
        "path": node.path,
        "nodeType": node.nodeType,
        "attributes_json": attrs_json,
    })


@app.post("/explain-smartform")
async def explain_smartform(payload: SmartFormInput):
    rows = []
    for node in payload.nodes:
        try:
            llm_result = llm_explain_node(node)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")
        rows.append(llm_result)

    # Final JSON "table"
    return {
        "formName": payload.formName,
        "system": payload.system,
        "client": payload.client,
        "field_table": rows
    }


@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
