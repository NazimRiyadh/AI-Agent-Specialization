"""
Invoice Expert Agent using Expert Tool Framework
Processes invoices using GPT-4o and structured Expert Tools.
"""

import os
import json
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable

import pdfplumber
from docx import Document
from openai import OpenAI

# =========================
# CORE FRAMEWORK COMPONENTS
# =========================

@dataclass
class ActionContext:
    storage: Dict[str, Any] = field(default_factory=dict)
    llm_client: Any = None
    model: str = "gpt-4o"

    def get(self, key: str, default=None):
        return self.storage.get(key, default)

    def set(self, key: str, value: Any):
        self.storage[key] = value


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
    """Decorator to register a function as an Expert Tool"""
    def decorator(func: Callable) -> Callable:
        tool = Tool(
            name=func.__name__,
            description=func.__doc__ or "",
            function=func,
            tags=tags or []
        )
        _global_registry.register(tool)
        return func
    return decorator

# =========================
# LLM HELPER
# =========================

def prompt_llm_for_json(action_context: ActionContext, schema: Dict, invoice_text: str) -> Dict:
    """Uses GPT-4o to extract structured JSON from invoice text"""
    try:
        client = action_context.llm_client
        schema_str = json.dumps(schema, indent=2)

        prompt = f"""
You are an expert invoice analyzer.
Extract the following structured data from the invoice.

Return only valid JSON matching this schema:
{schema_str}

<invoice>
{invoice_text}
</invoice>
"""
        response = client.chat.completions.create(
            model=action_context.model,
            messages=[
                {"role": "system", "content": "You are a precise data extraction assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        result_text = response.choices[0].message.content.strip()
        # Clean code fences if any
        for marker in ["```json", "```"]:
            if result_text.startswith(marker):
                result_text = result_text[len(marker):]
            if result_text.endswith(marker):
                result_text = result_text[:-len(marker)]

        return json.loads(result_text.strip())
    except Exception as e:
        print("❌ LLM extraction failed:", e)
        traceback.print_exc()
        return {}

# =========================
# INVOICE TOOLS
# =========================

@register_tool(tags=["document_processing", "invoices"])
def extract_invoice_data(action_context: ActionContext, document_text: str) -> dict:
    """
    Extract structured invoice data from text using GPT-4o.
    Returns invoice number, date, total_amount, vendor info, line_items.
    """
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
    return prompt_llm_for_json(action_context, invoice_schema, document_text)


@register_tool(tags=["storage", "invoices"])
def store_invoice(action_context: ActionContext, invoice_data: dict) -> dict:
    """
    Store invoice in memory dictionary by invoice_number.
    """
    storage = action_context.get("invoice_storage", {})
    invoice_number = invoice_data.get("invoice_number")
    if not invoice_number:
        raise ValueError("Invoice must have an invoice_number")
    storage[invoice_number] = invoice_data
    action_context.set("invoice_storage", storage)
    return {"status": "success", "invoice_number": invoice_number}

# =========================
# AGENT
# =========================

class Agent:
    def __init__(self, llm_client: Any):
        self.context = ActionContext(llm_client=llm_client)
        self.context.set("invoice_storage", {})
        self.registry = _global_registry

    def process_invoice_text(self, text: str) -> dict:
        extractor = self.registry.get_tool("extract_invoice_data")
        store = self.registry.get_tool("store_invoice")
        data = extractor.function(self.context, document_text=text)
        return store.function(self.context, invoice_data=data)

# =========================
# FILE READING
# =========================

def read_invoice(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".txt":
        return file_path.read_text(encoding="utf-8")
    elif ext == ".pdf":
        with pdfplumber.open(file_path.resolve()) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# =========================
# FOLDER PROCESSING
# =========================

def process_invoices_folder(agent: Agent, folder_path: str = "./invoices_to_process"):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder_path}")
        return

    for file_path in folder.iterdir():
        if file_path.is_file():
            print(f"🧾 Processing {file_path.name}...")
            try:
                text = read_invoice(file_path)
                result = agent.process_invoice_text(text)
                print(f"✅ Processed {file_path.name}: {result.get('invoice_number')}")
            except Exception as e:
                print(f"❌ Failed {file_path.name}: {e}")
                traceback.print_exc()

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Set OPENAI_API_KEY environment variable first.")
        exit(1)

    client = OpenAI(api_key=api_key)
    agent = Agent(llm_client=client)
    process_invoices_folder(agent)

    stored = agent.context.get("invoice_storage", {})
    print("\n📦 Stored Invoices:")
    print(json.dumps(stored, indent=2))
