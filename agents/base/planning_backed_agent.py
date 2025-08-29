"""
Shared planning context loader and factory for IRONFORGE agents.

This implements the document-mediated communication protocol where agents
load their behavior, tools, and configuration from planning documents
rather than direct communication.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PlanningContext:
    """Structured planning context loaded from markdown documents."""
    
    # Core context documents
    requirements: str = ""
    behavior: str = ""
    tools: str = ""
    dependencies: str = ""
    
    # Parsed content
    parsed_requirements: Dict[str, Any] = None
    parsed_behavior: Dict[str, Any] = None
    parsed_tools: Dict[str, Any] = None
    parsed_dependencies: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parsed_requirements is None:
            self.parsed_requirements = {}
        if self.parsed_behavior is None:
            self.parsed_behavior = {}
        if self.parsed_tools is None:
            self.parsed_tools = {}
        if self.parsed_dependencies is None:
            self.parsed_dependencies = {}


class PlanningDocumentLoader:
    """Loads and parses planning documents for agents."""
    
    @staticmethod
    def load_agent_planning(agent_name: str, agents_root: Path = None) -> PlanningContext:
        """Load all planning documents for an agent."""
        if agents_root is None:
            agents_root = Path(__file__).parent.parent
        
        planning_path = agents_root / agent_name / "planning"
        
        context = PlanningContext()
        
        # Load raw documents
        for doc_name, attr_name in [
            ("INITIAL.md", "requirements"),
            ("prompts.md", "behavior"), 
            ("tools.md", "tools"),
            ("dependencies.md", "dependencies")
        ]:
            doc_path = planning_path / doc_name
            if doc_path.exists():
                setattr(context, attr_name, doc_path.read_text())
        
        # Parse structured content
        context.parsed_requirements = PlanningDocumentLoader._parse_requirements(context.requirements)
        context.parsed_behavior = PlanningDocumentLoader._parse_behavior(context.behavior)
        context.parsed_tools = PlanningDocumentLoader._parse_tools(context.tools)
        context.parsed_dependencies = PlanningDocumentLoader._parse_dependencies(context.dependencies)
        
        return context
    
    @staticmethod
    def _parse_requirements(content: str) -> Dict[str, Any]:
        """Parse INITIAL.md requirements document."""
        parsed = {
            "functional_requirements": [],
            "technical_requirements": [],
            "success_criteria": [],
            "assumptions": []
        }
        
        if not content:
            return parsed
        
        # Extract functional requirements
        fr_match = re.search(r'## Functional Requirements(.*?)(?=##|$)', content, re.DOTALL)
        if fr_match:
            fr_text = fr_match.group(1)
            parsed["functional_requirements"] = PlanningDocumentLoader._extract_requirements_list(fr_text)
        
        # Extract technical requirements
        tr_match = re.search(r'## Technical Requirements(.*?)(?=##|$)', content, re.DOTALL)
        if tr_match:
            tr_text = tr_match.group(1)
            parsed["technical_requirements"] = PlanningDocumentLoader._extract_requirements_list(tr_text)
        
        # Extract success criteria
        sc_match = re.search(r'## Success Criteria(.*?)(?=##|$)', content, re.DOTALL)
        if sc_match:
            sc_text = sc_match.group(1)
            parsed["success_criteria"] = PlanningDocumentLoader._extract_requirements_list(sc_text)
        
        return parsed
    
    @staticmethod
    def _parse_behavior(content: str) -> Dict[str, Any]:
        """Parse prompts.md behavior document."""
        parsed = {
            "system_prompt": "",
            "behavioral_guidelines": [],
            "decision_patterns": [],
            "communication_style": ""
        }
        
        if not content:
            return parsed
        
        # Extract system prompt
        prompt_match = re.search(r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', content, re.DOTALL)
        if prompt_match:
            parsed["system_prompt"] = prompt_match.group(1).strip()
        
        # Extract behavioral guidelines
        guidelines_match = re.search(r'Core Responsibilities:(.*?)(?=Available Tools:|$)', content, re.DOTALL)
        if guidelines_match:
            guidelines_text = guidelines_match.group(1)
            parsed["behavioral_guidelines"] = [
                line.strip().lstrip('1234567890. -')
                for line in guidelines_text.split('\n') 
                if line.strip() and not line.strip().startswith('#')
            ]
        
        return parsed
    
    @staticmethod
    def _parse_tools(content: str) -> Dict[str, Any]:
        """Parse tools.md tool specifications."""
        parsed = {
            "tools": [],
            "utility_functions": [],
            "parameters": {}
        }
        
        if not content:
            return parsed
        
        # Extract tool names from function definitions
        tool_matches = re.findall(r'async def (\w+)\(', content)
        parsed["tools"] = tool_matches
        
        # Extract parameter validation info
        if "Parameter Validation" in content:
            validation_section = re.search(r'## Parameter Validation(.*?)(?=##|$)', content, re.DOTALL)
            if validation_section:
                parsed["parameters"]["validation"] = validation_section.group(1).strip()
        
        return parsed
    
    @staticmethod
    def _parse_dependencies(content: str) -> Dict[str, Any]:
        """Parse dependencies.md configuration."""
        parsed = {
            "environment_variables": {},
            "settings": {},
            "dependencies": []
        }
        
        if not content:
            return parsed
        
        # Extract environment variables from bash code blocks
        env_matches = re.findall(r'```bash(.*?)```', content, re.DOTALL)
        for env_block in env_matches:
            for line in env_block.split('\n'):
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    parsed["environment_variables"][key.strip()] = value.strip()
        
        # Extract Python dependencies from requirements sections
        if "Python Packages" in content or "dependencies" in content.lower():
            deps_match = re.search(r'```txt(.*?)```', content, re.DOTALL)
            if deps_match:
                deps_text = deps_match.group(1)
                parsed["dependencies"] = [
                    line.strip() for line in deps_text.split('\n') 
                    if line.strip() and not line.startswith('#')
                ]
        
        return parsed
    
    @staticmethod
    def _extract_requirements_list(text: str) -> List[str]:
        """Extract requirements from markdown text."""
        requirements = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('###') and 'FR' in line:
                # Functional requirement header
                req_match = re.search(r'###\s*([^:]+):', line)
                if req_match:
                    requirements.append(req_match.group(1).strip())
            elif line.startswith('###') and 'TR' in line:
                # Technical requirement header
                req_match = re.search(r'###\s*([^:]+):', line)
                if req_match:
                    requirements.append(req_match.group(1).strip())
        return requirements


class PlanningBackedAgent(ABC):
    """Base mixin for agents that load behavior from planning documents."""
    
    def __init__(self, agent_name: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if agent_name is None:
            # Infer agent name from class name
            agent_name = self.__class__.__name__.lower().replace('agent', '')
            if not agent_name:
                agent_name = 'unknown_agent'
        
        self.agent_name = agent_name
        self.planning_context = None
        self._load_planning_context()
    
    def _load_planning_context(self):
        """Load planning context from documents."""
        try:
            self.planning_context = PlanningDocumentLoader.load_agent_planning(self.agent_name)
            self._apply_planning_context()
        except Exception as e:
            print(f"Warning: Could not load planning context for {self.agent_name}: {e}")
            self.planning_context = PlanningContext()
    
    def _apply_planning_context(self):
        """Apply planning context to agent configuration."""
        if not self.planning_context:
            return
        
        # Apply environment variables if available
        env_vars = self.planning_context.parsed_dependencies.get("environment_variables", {})
        for key, value in env_vars.items():
            if not hasattr(self, key.lower()) and value:
                try:
                    # Try to convert to appropriate type
                    if value.lower() in ['true', 'false']:
                        setattr(self, key.lower(), value.lower() == 'true')
                    elif value.replace('.', '').isdigit():
                        setattr(self, key.lower(), float(value) if '.' in value else int(value))
                    else:
                        setattr(self, key.lower(), value)
                except:
                    setattr(self, key.lower(), value)
    
    @classmethod
    def from_planning_documents(cls, agent_name: str = None, **kwargs):
        """Factory method to create agent from planning documents."""
        if agent_name is None:
            agent_name = cls.__name__.lower().replace('agent', '')
        
        return cls(agent_name=agent_name, **kwargs)
    
    # Document context accessor methods (matching the test interface)
    async def get_requirements_from_planning(self) -> str:
        """Get requirements context from INITIAL.md."""
        if not self.planning_context:
            return ""
        
        # Extract functional requirements text
        functional_reqs = self.planning_context.parsed_requirements.get("functional_requirements", [])
        return " | ".join(functional_reqs) if functional_reqs else self.planning_context.requirements
    
    async def get_behavior_from_planning(self) -> str:
        """Get behavioral context from prompts.md."""
        if not self.planning_context:
            return ""
        
        # Extract behavioral guidelines
        guidelines = self.planning_context.parsed_behavior.get("behavioral_guidelines", [])
        return " | ".join(guidelines) if guidelines else self.planning_context.behavior
    
    async def get_tools_from_planning(self) -> List[str]:
        """Get tool specifications from tools.md."""
        if not self.planning_context:
            return []
        
        return self.planning_context.parsed_tools.get("tools", [])
    
    async def get_dependencies_from_planning(self) -> Dict[str, Any]:
        """Get dependencies context from dependencies.md."""
        if not self.planning_context:
            return {}
        
        return self.planning_context.parsed_dependencies.get("environment_variables", {})
    
    # Abstract methods for concrete implementation
    @abstractmethod
    async def execute_primary_function(self, *args, **kwargs):
        """Execute the agent's primary function using planning context."""
        pass


# Factory functions for common agent creation patterns
def create_planning_backed_agent(agent_class, agent_name: str = None, **kwargs):
    """Create any agent with planning document support."""
    class PlanningBackedVersion(PlanningBackedAgent, agent_class):
        async def execute_primary_function(self, *args, **kwargs):
            # Delegate to original agent's primary method
            if hasattr(agent_class, 'execute'):
                return await agent_class.execute(self, *args, **kwargs)
            elif hasattr(agent_class, 'run'):
                return await agent_class.run(self, *args, **kwargs)
            else:
                raise NotImplementedError(f"Agent {agent_class} needs execute() or run() method")
    
    return PlanningBackedVersion.from_planning_documents(agent_name, **kwargs)


def load_agent_configuration(agent_name: str) -> Dict[str, Any]:
    """Load configuration for an agent from its dependencies.md."""
    context = PlanningDocumentLoader.load_agent_planning(agent_name)
    return {
        "environment_variables": context.parsed_dependencies.get("environment_variables", {}),
        "dependencies": context.parsed_dependencies.get("dependencies", []),
        "settings": context.parsed_dependencies.get("settings", {})
    }