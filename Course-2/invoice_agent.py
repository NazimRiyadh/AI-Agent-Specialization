"""
Invoice Processing Agent - Fully Automated
Processes invoices from a folder: extracts and stores them automatically.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field
import openai
import pdfplumber          # pip install pdfplumber
from docx import Document  # pip install python-docx

# ============================================================================ 
# CORE INFRASTRUCTURE - Agent Framework Components
# ============================================================================

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
class Goal:
    name: str
    description: str

@dataclass
class Tool:
    name: str
    description: str
    function: Callable
    parameters: Dict
    tags: List[str] = field(default_factory=list)

class PythonActionRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Tool:
        return self.tools.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        return list(self.tools.values())
    
    def to_openai_tools(self) -> List[Dict]:
        return [ {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        } for tool in self.tools.values() ]

class PythonEnvironment:
    def __init__(self):
        self.state = {}
    
    def update_state(self, key: str, value: Any):
        self.state[key] = value
    
    def get_state(self, key: str, default=None):
        return self.state.get(key, default)

class AgentFunctionCallingActionLanguage:
    def __init__(self):
        self.protocol = "function_calling"

# ============================================================================ 
# TOOL REGISTRATION DECORATOR
# ============================================================================

_global_registry = PythonActionRegistry()

def register_tool(tags: List[str] = None):
    def decorator(func: Callable) -> Callable:
        import inspect
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        
        parameters = {"type": "object", "properties": {}, "required": []}
        
        for param_name, param in sig.parameters.items():
            if param_name == "action_context":
                continue
            
            param_type = "string"
            if param.annotation == dict:
                param_type = "object"
            elif param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            
            parameters["properties"][param_name] = {"type": param_type, "description": f"Parameter {param_name}"}
            
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        tool = Tool(
            name=func.__name__,
            description=doc.split('\n\n')[0] if doc else func.__name__,
            function=func,
            parameters=parameters,
            tags=tags or []
        )
        
        _global_registry.register(tool)
        return func
    
    return decorator

# ============================================================================ 
# LLM HELPER FUNCTIONS
# ============================================================================

def prompt_llm_for_json(action_context: ActionContext, schema: Dict, prompt: str) -> Dict:
    client = action_context.llm_client
    schema_str = json.dumps(schema, indent=2)
    full_prompt = f"""{prompt}

Please extract the information and return it as valid JSON matching this schema:
{schema_str}

Return ONLY the JSON object, no additional text."""
    
    response = client.chat.completions.create(
        model=action_context.model,
        messages=[
            {"role": "system", "content": "You are a precise data extraction assistant. Return only valid JSON."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0
    )
    
    result_text = response.choices[0].message.content.strip()
    
    if result_text.startswith("```json"):
        result_text = result_text[7:]
    if result_text.startswith("```"):
        result_text = result_text[3:]
    if result_text.endswith("```"):
        result_text = result_text[:-3]
    
    return json.loads(result_text.strip())

# ============================================================================ 
# INVOICE PROCESSING TOOLS
# ============================================================================

@register_tool(tags=["document_processing", "invoices"])
def extract_invoice_data(action_context: ActionContext, document_text: str) -> dict:
    invoice_schema = {
        "type": "object",
        "required": ["invoice_number", "date", "total_amount"],
        "properties": {
            "invoice_number": {"type": "string"},
            "date": {"type": "string"},
            "total_amount": {"type": "number"},
            "vendor": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "string"}
                }
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit_price": {"type": "number"},
                        "total": {"type": "number"}
                    }
                }
            }
        }
    }
    
    extraction_prompt = f"""
You are an expert invoice analyzer. Extract invoice information accurately and 
thoroughly. Pay special attention to:
- Invoice numbers
- Dates
- Amounts
- Line items

Extract the invoice data from:

<invoice>
{document_text}
</invoice>
"""
    
    return prompt_llm_for_json(action_context, invoice_schema, extraction_prompt)

@register_tool(tags=["storage", "invoices"])
def store_invoice(action_context: ActionContext, invoice_data: dict) -> dict:
    storage = action_context.get("invoice_storage", {})
    invoice_number = invoice_data.get("invoice_number")
    if not invoice_number:
        raise ValueError("Invoice data must contain an invoice number")
    
    storage[invoice_number] = invoice_data
    action_context.set("invoice_storage", storage)
    
    return {"status": "success", "message": f"Stored invoice {invoice_number}", "invoice_number": invoice_number}

# ============================================================================ 
# AGENT CLASS
# ============================================================================

class Agent:
    def __init__(self, goals: List[Goal], agent_language: AgentFunctionCallingActionLanguage,
                 action_registry: PythonActionRegistry, environment: PythonEnvironment,
                 llm_client: Any):
        self.goals = goals
        self.agent_language = agent_language
        self.action_registry = action_registry
        self.environment = environment
        self.context = ActionContext(llm_client=llm_client)
        self.context.set("invoice_storage", {})
    
    def process_invoice_text(self, invoice_text: str) -> dict:
        extract_tool = self.action_registry.get_tool("extract_invoice_data")
        invoice_data = extract_tool.function(self.context, document_text=invoice_text)
        
        store_tool = self.action_registry.get_tool("store_invoice")
        result = store_tool.function(self.context, invoice_data=invoice_data)
        return result

# ============================================================================ 
# AGENT FACTORY
# ============================================================================

def create_invoice_agent(api_key: str = None) -> Agent:
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required.")
    
    client = openai.OpenAI(api_key=api_key)
    environment = PythonEnvironment()
    
    goals = [
        Goal(name="Persona", description="Invoice Processing Agent specialized in extracting and storing invoice data."),
        Goal(name="Process Invoices", description="Automatically extract data and store invoices in a database.")
    ]
    
    agent = Agent(goals, AgentFunctionCallingActionLanguage(), _global_registry, environment, client)
    return agent

# ============================================================================ 
# SAFE FILE READING (No textract)
# ============================================================================

def read_invoice(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".txt":
        return file_path.read_text(encoding="utf-8")
    elif ext == ".pdf":
            file_str = str(file_path.resolve())  # absolute path
            with pdfplumber.open(file_str) as pdf:
                return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ============================================================================ 
# FOLDER-BASED AUTOMATION
# ============================================================================

def process_invoices_folder(agent: Agent, folder_path: str = "./invoices_to_process"):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder_path}")
        return
    
    for file_path in folder.iterdir():
        if file_path.is_file():
            try:
                invoice_text = read_invoice(file_path)
                result = agent.process_invoice_text(invoice_text)
                print(f"✅ Processed {file_path.name}: {result['invoice_number']}")
            except Exception as e:
                print(f"❌ Failed {file_path.name}: {e}")

# ============================================================================ 
# MAIN
# ============================================================================

if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
    agent = create_invoice_agent(api_key)
    process_invoices_folder(agent, folder_path="./invoices_to_process")
    
    stored = agent.context.get("invoice_storage", {})
    print("\nSTORED INVOICES:")
    print(json.dumps(stored, indent=2))
