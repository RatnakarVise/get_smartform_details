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


# ---- Merge Logic for Consecutive ITEM Nodes ----
def merge_consecutive_items(nodes: List[Node]) -> List[Node]:
    merged_nodes = []
    buffer_node = None

    for node in nodes:
        if node.elemName.upper() == "ITEM":  # condition: ITEM nodes
            if buffer_node is None:
                # start a new buffer
                buffer_node = Node(**node.dict())
            else:
                # merge textPayload
                buffer_node.textPayload += "\n" + node.textPayload
                # merge attributes too
                buffer_node.attributes.extend(node.attributes)
        else:
            # flush buffer before adding non-ITEM node
            if buffer_node:
                merged_nodes.append(buffer_node)
                buffer_node = None
            merged_nodes.append(node)

    # flush last buffer if still active
    if buffer_node:
        merged_nodes.append(buffer_node)

    return merged_nodes


# ---- LLM Batch Explainer ----
def llm_explain_nodes(nodes: List[Node]):
    SYSTEM_MSG = "You are an SAP SmartForm and ABAP expert. Always respond in strict JSON."

    USER_TEMPLATE = """
You are reviewing multiple SmartForm nodes. 
Group them into a hierarchy:
- At the top: PAGE (elemName=PAGE or NODETYPE=PA, 
  or elemName=CAPTION/INAME that represent page captions/names).
- Under PAGE: WINDOWS (elemName=WINDOW or NODETYPE=WI).
- Under WINDOW: all child nodes (depth > window depth).

For each element, include:
- elemName
- path
- nodeType
- attributes
- textPayload
- mapping: Mention any SAP tables, structures, or fields referenced in this node. 
           If none, put "".
- coding: Mention all ABAP variables and code logic from textPayload. 
          Show them clearly (e.g., variable declarations, assignments, SELECTs). 
          If none, put "".
- usage: Business/functional purpose of this element with the variables and tables name explain.

⚠️ Important:
- If a PAGE name is not explicitly available in the node data, set "page": "" (empty string). 
-"page": "Use CAPTION/INAME textPayload if available, otherwise empty string."
- Do NOT invent or add a generic name like "Page 1" or "Unnamed Page".

Return ONLY a strict JSON object like:
{{
  "pages": [
    {{
      "page": "",   <-- leave blank if no page name
      "windows": [
        {{
          "window": "Window Name",
          "elements": [
            {{
              "elemName": "...",
              "textPayload": "...", 
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

    # --- PRE-PROCESS: Merge ITEM nodes ---
    merged_nodes = merge_consecutive_items(nodes)

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
        for n in merged_nodes
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
