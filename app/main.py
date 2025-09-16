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
        # treat "item" case-insensitively
        if node.elemName and node.elemName.strip().upper() == "ITEM":
            if buffer_node is None:
                buffer_node = Node(**node.dict())
            else:
                # merge textPayload (preserve order)
                buffer_node.textPayload = (buffer_node.textPayload or "") + "\n" + (node.textPayload or "")
                # merge attributes
                buffer_node.attributes.extend(node.attributes or [])
        else:
            # flush buffer before non-ITEM node
            if buffer_node:
                merged_nodes.append(buffer_node)
                buffer_node = None
            merged_nodes.append(node)

    # final flush
    if buffer_node:
        merged_nodes.append(buffer_node)

    return merged_nodes


# ---- Page Name Extractor ----
def extract_page_name(nodes: List[Node]) -> str:
    """
    Extract the correct page name:
     - Look for a PAGE marker (elemName == 'PAGE') OR a NODETYPE node whose textPayload == 'PA'.
     - When found, scan its subtree (subsequent nodes with depth > page_depth) for the first CAPTION or INAME with a non-empty textPayload.
     - If none found, return empty string.
    This prevents picking captions that belong to CODE blocks or other regions.
    """
    if not nodes:
        return ""

    n = len(nodes)
    for i, node in enumerate(nodes):
        try:
            elem_upper = (node.elemName or "").strip().upper()
            node_text = (node.textPayload or "").strip()
        except Exception:
            elem_upper = ""
            node_text = ""

        # Candidate page marker:
        is_page_marker = False
        if elem_upper == "PAGE":
            is_page_marker = True
        # Some payloads set elemName == "NODETYPE" and textPayload == "PA"
        elif elem_upper == "NODETYPE" and node_text.upper() == "PA":
            is_page_marker = True
        # Also allow explicit nodeType property == 'PA' (if populated)
        elif (node.nodeType or "").strip().upper() == "PA":
            is_page_marker = True

        if not is_page_marker:
            continue

        page_depth = node.depth or 0

        # Search forward for CAPTION / INAME within subtree (depth > page_depth)
        j = i + 1
        while j < n:
            nj = nodes[j]
            # if depth <= page_depth, we've left subtree
            if (nj.depth or 0) <= page_depth:
                break
            nelem = (nj.elemName or "").strip().upper()
            ntext = (nj.textPayload or "").strip()
            if nelem in ("CAPTION", "INAME") and ntext:
                return ntext
            j += 1

        # If PAGE node has attributes that carry the name, check them
        for attr in (node.attributes or []):
            if (attr.name or "").strip().lower() in ("name", "iname", "caption") and (attr.value or "").strip():
                return (attr.value or "").strip()

        # otherwise continue searching for next page marker

    # No explicit page marker with caption found → return empty string (do NOT invent)
    return ""


# ---- LLM Batch Explainer ----
def llm_explain_nodes(nodes: List[Node]):
    SYSTEM_MSG = "You are an SAP SmartForm and ABAP expert. Always respond in strict JSON."

    USER_TEMPLATE = """
You are reviewing multiple SmartForm nodes.
Group them into a hierarchy:
- At the top: PAGE (value already extracted and provided as "{page_name}").
- Under PAGE: WINDOWS (elemName=WINDOW or NODETYPE=WI).
- Under WINDOW: all child nodes (depth > window depth).

For each element, include:
- elemName
- path
- nodeType
- attributes
- textPayload
- mapping: Mention any SAP tables, structures, or fields referenced in this node. If none, put "".
- coding: Mention all ABAP variables and code logic from textPayload. Show them clearly (e.g., variable declarations, assignments, SELECTs). If none, put "".
- usage: Business/functional purpose of this element with the variables and tables name explained.

⚠️ Important:
- Always set "page": "{page_name}" exactly. If empty string is provided, keep it empty.
- Do NOT invent or add a generic name like "Page 1" or "Unnamed Page".

Return ONLY a strict JSON object with the structure:
{{
  "pages": [
    {{
      "page": "{page_name}",
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

Nodes (the preprocessed, merged nodes array):
{nodes_json}
"""

    # --- PRE-PROCESS: Merge ITEM nodes ---
    merged_nodes = merge_consecutive_items(nodes)

    # --- Extract page name (strictly from node structure) ---
    page_name = extract_page_name(merged_nodes)

    # Prepare input for LLM (flatten nodes to simple dicts)
    nodes_data = [
        {
            "elemName": n.elemName,
            "path": n.path,
            "nodeType": n.nodeType,
            "depth": n.depth,
            "attributes": [{"name": a.name, "value": a.value} for a in (n.attributes or [])],
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

    # pass both nodes_json and page_name into the template
    return chain.invoke({"nodes_json": nodes_json, "page_name": page_name})


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
