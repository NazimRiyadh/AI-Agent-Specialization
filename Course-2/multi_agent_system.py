"""
Multi-Agent Invoice Processing System with GPT-4o (Responses API)
- CoordinatorAgent orchestrates the workflow
- ExtractionAgent extracts invoice data
- ValidationAgent validates invoice data
- StorageAgent stores the invoice
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Callable, List
from dataclasses import dataclass, field
import openai
import pdfplumber
from docx import Document
import traceback

# ===================== CORE =====================

@dataclass
class ActionContext:
    storage: Dict[str, Any] = field(default_factory=dict)
    llm_client: Any = None
    model: str = "gpt-4o"
    agent_registry: Any = None

    def get(self, key: str, default=None):
        return self.storage.get(key, default)

    def set(self, key: str, value: Any):
        self.storage[key] = value

@dataclass
class Goal:
    name: str
    description: str

@dataclass
class Tool:
    name: str
    description: str
    function: Callable
    tags: List[str] = field(default_factory=list)

class PythonActionRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool:
        return self.tools.get(name)

_global_registry = PythonActionRegistry()

def register_tool(tags: List[str] = None):
    def decorator(func: Callable) -> Callable:
        tool = Tool(
            name=func.__name__,
            description=(func.__doc__ or func.__name__),
            function=func,
            tags=tags or []
        )
        _global_registry.register(tool)
        return func
    return decorator

# ===================== LLM HELPER =====================

def prompt_llm_for_json(action_context: ActionContext, schema: Dict, prompt: str) -> dict:
    """Use GPT-4o to extract structured JSON from invoice text"""
    try:
        client = action_context.llm_client
        response = client.chat.completions.create(
            model=action_context.model,
            messages=[
                {"role": "system", "content": "You are an expert invoice extraction assistant. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        result_text = response.choices[0].message.content.strip()
        # Remove markdown code fences if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        return json.loads(result_text)
    except Exception as e:
        print("❌ LLM extraction failed:", e)
        traceback.print_exc()
        return {}

# ===================== TOOLS =====================

@register_tool(tags=["document_processing", "invoices"])
def extract_invoice_data(action_context: ActionContext, document_text: str) -> dict:
    """Extract invoice data from raw text using GPT-4o"""
    invoice_schema = {
        "type": "object",
        "required": ["invoice_number", "date", "total_amount"],
        "properties": {
            "invoice_number": {"type": "string"},
            "date": {"type": "string"},
            "total_amount": {"type": "number"},
            "vendor": {"type": "object", "properties": {"name": {"type": "string"}, "address": {"type": "string"}}},
            "line_items": {"type": "array", "items": {"type": "object", "properties": {"description": {"type": "string"}, "quantity": {"type": "number"}, "unit_price": {"type": "number"}, "total": {"type": "number"}}}}
        }
    }

    prompt = f"""
You are an expert invoice analyzer. Extract all invoice information accurately.
Required fields: invoice_number, date, total_amount.
Optional: vendor name, vendor address, line_items.
If any required field is missing, put null.
Ensure numeric fields are numbers, not strings.

<invoice>
{document_text}
</invoice>
Return ONLY valid JSON matching this schema.
"""
    return prompt_llm_for_json(action_context, invoice_schema, prompt)

def reconcile_invoice(invoice_data: dict) -> dict:
    """Ensure total_amount matches the sum of line items"""
    line_items = invoice_data.get("line_items", [])
    if line_items:
        sum_total = sum(item.get("total", 0) for item in line_items)
        invoice_data["total_amount"] = sum_total
    return invoice_data

@register_tool(tags=["validation", "invoices"])
def validate_invoice(action_context: ActionContext, invoice_data: dict) -> dict:
    """Validate invoice totals and required fields"""
    errors = []
    if not invoice_data.get("invoice_number"):
        errors.append("Missing invoice_number")
    if invoice_data.get("total_amount") is None:
        errors.append("Missing total_amount")
    line_total = sum(item.get("total", 0) for item in invoice_data.get("line_items", []))
    if abs(line_total - invoice_data.get("total_amount", 0)) > 0.01:
        errors.append("Line item totals do not match total_amount")
    return {"valid": len(errors) == 0, "errors": errors}

@register_tool(tags=["storage", "invoices"])
def store_invoice(action_context: ActionContext, invoice_data: dict) -> dict:
    """Store invoice in context storage"""
    storage = action_context.get("invoice_storage", {})
    invoice_number = invoice_data.get("invoice_number")
    if not invoice_number:
        raise ValueError("Invoice missing number")
    storage[invoice_number] = invoice_data
    action_context.set("invoice_storage", storage)
    return {"status": "success", "invoice_number": invoice_number}

@register_tool(tags=["multi_agent"])
def call_agent(action_context: ActionContext, agent_name: str, task: dict) -> dict:
    """Call another agent from the registry with a task"""
    registry = action_context.agent_registry
    if not registry:
        raise ValueError("No agent registry found")
    agent_fn = registry.get_agent(agent_name)
    if not agent_fn:
        raise ValueError(f"Agent '{agent_name}' not found")
    # Run agent with isolated memory
    temp_context = ActionContext(storage=dict(action_context.storage), llm_client=action_context.llm_client)
    temp_context.agent_registry = registry
    return agent_fn(temp_context, tool=task.get("tool"), params=task.get("params", {}))

# ===================== AGENT CLASS =====================

class Agent:
    def __init__(self, name: str, goals: List[Goal], tools_registry: PythonActionRegistry, llm_client: Any):
        self.name = name
        self.goals = goals
        self.tools_registry = tools_registry
        self.context = ActionContext(llm_client=llm_client)
        self.context.agent_registry = None
        self.context.set("invoice_storage", {})

    def run(self, action_context: ActionContext, tool: str = None, params: dict = None) -> dict:
        tool_obj = self.tools_registry.get_tool(tool)
        if not tool_obj:
            raise ValueError(f"Tool '{tool}' not found for agent {self.name}")
        return tool_obj.function(action_context, **(params or {}))

# ===================== AGENT REGISTRY =====================

class AgentRegistry:
    def __init__(self):
        self.agents: dict = {}

    def register_agent(self, name: str, agent: Agent):
        self.agents[name] = agent

    def get_agent(self, name: str) -> Callable:
        agent = self.agents.get(name)
        return agent.run if agent else None

# ===================== FILE READING =====================

def read_invoice(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".txt":
        return file_path.read_text(encoding="utf-8")
    elif ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ===================== COORDINATOR =====================

def process_invoices_folder(coordinator: Agent, folder_path: str):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder_path}")
        return

    for file_path in folder.iterdir():
        if file_path.is_file():
            try:
                print(f"🧾 Processing {file_path.name}...")
                text = read_invoice(file_path)

                # Extract
                extracted = call_agent(coordinator.context, "ExtractionAgent", {
                    "tool": "extract_invoice_data", "params": {"document_text": text}
                })

                # Reconcile totals
                extracted = reconcile_invoice(extracted)

                # Validate
                validated = call_agent(coordinator.context, "ValidationAgent", {
                    "tool": "validate_invoice", "params": {"invoice_data": extracted}
                })
                if not validated.get("valid", False):
                    print(f"❌ Validation failed: {validated.get('errors')}")
                    continue

                # Store
                stored = call_agent(coordinator.context, "StorageAgent", {
                    "tool": "store_invoice", "params": {"invoice_data": extracted}
                })
                print(f"✅ Stored invoice {stored['invoice_number']}")

            except Exception as e:
                print(f"❌ Failed {file_path.name}: {e}")
                traceback.print_exc()

    print("\n📦 All stored invoices:")
    print(json.dumps(coordinator.context.get("invoice_storage", {}), indent=2))

# ===================== MAIN =====================

if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key missing")

    client = openai.OpenAI(api_key=api_key)

    # Create agents
    ExtractionAgent = Agent("ExtractionAgent", [], _global_registry, client)
    ValidationAgent = Agent("ValidationAgent", [], _global_registry, client)
    StorageAgent = Agent("StorageAgent", [], _global_registry, client)
    CoordinatorAgent = Agent("CoordinatorAgent", [], _global_registry, client)

    # Agent registry
    registry = AgentRegistry()
    for agent in [ExtractionAgent, ValidationAgent, StorageAgent, CoordinatorAgent]:
        registry.register_agent(agent.name, agent)
        agent.context.agent_registry = registry

    # Process invoices
    process_invoices_folder(CoordinatorAgent, "./invoices_to_process")
