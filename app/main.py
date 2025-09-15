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

app = FastAPI(title="SAP SmartForm Node Field Mapping Explainer API (Batch Optimized)")


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


# ---- LLM Batch Explainer ----
def llm_explain_nodes(nodes: List[Node]):
    SYSTEM_MSG = "You are an SAP SmartForm and ABAP expert. Always respond in strict JSON."

    USER_TEMPLATE = """
You are reviewing multiple SmartForm nodes. 
Group them into a hierarchy:
- At the top: PAGE (elemName=PAGE or NODETYPE=PA).
- Under PAGE: WINDOWS (elemName=WINDOW or NODETYPE=WI).
- Under WINDOW: all child nodes (depth > window depth).

For each element, include:
- elemName
- path
- nodeType
- attributes
- mapping (technical field mapping or SmartForm meaning)
- coding (ABAP / form logic if relevant)
- usage (business/functional purpose)

Return ONLY a strict JSON object like:
{{
  "pages": [
    {{
      "page": "Page Name / ID",
      "windows": [
        {{
          "window": "Window Name",
          "elements": [
            {{
              "elemName": "...",
              "path": "...",
              "nodeType": "...",
              "attributes": [...],
              "mapping": "...",
              "coding": "...",
              "usage": "..."
            }}
          ]
        }}
      ]
    }}
  ]
}}


Nodes:
{nodes_json}
"""
    # Prepare input for LLM
    nodes_data = [
        {
            "elemName": n.elemName,
            "path": n.path,
            "nodeType": n.nodeType,
            "depth": n.depth,
            "attributes": [{"name": a.name, "value": a.value} for a in n.attributes],
            "textPayload": n.textPayload,
        }
        for n in nodes
    ]
    nodes_json = json.dumps(nodes_data, ensure_ascii=False, indent=2)

    # Build chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ])

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    return chain.invoke({"nodes_json": nodes_json})


# ---- API Endpoint ----
@app.post("/explain-smartform")
async def explain_smartform(payload: SmartFormInput):
    try:
        rows = llm_explain_nodes(payload.nodes)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    return {
        "formName": payload.formName,
        "system": payload.system,
        "client": payload.client,
        "field_table": rows
    }


@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
