#!/usr/bin/env python3
"""
TQE Send Command Implementation
Provides communication interface between TQE Orchestrator and agents
"""

import sys
import json
from typing import Dict, Any, Optional
from tqe_orchestrator import TQEOrchestrator

def tqe_send(agent_name: str, message: str, orchestrator: Optional[TQEOrchestrator] = None) -> Dict[str, Any]:
    """
    Send message to TQE agent via orchestrator
    
    Args:
        agent_name: Target agent ('pattern', 'data', 'validation', 'execution')
        message: Message to send to agent
        orchestrator: Optional orchestrator instance (creates new if None)
        
    Returns:
        Agent response
    """
    
    if orchestrator is None:
        orchestrator = TQEOrchestrator()
    
    return orchestrator.agent_send(agent_name, message)

def main():
    """Command line interface for tqe-send"""
    
    if len(sys.argv) < 3:
        print("Usage: python tqe_send.py <agent> '<message>'")
        print("Available agents: pattern, data, validation, execution")
        sys.exit(1)
    
    agent_name = sys.argv[1]
    message = sys.argv[2]
    
    try:
        orchestrator = TQEOrchestrator()
        response = tqe_send(agent_name, message, orchestrator)
        
        print("🎯 TQE Agent Communication")
        print("=" * 50)
        print(f"Agent: {agent_name}")
        print(f"Message: {message}")
        print(f"Response: {json.dumps(response, indent=2)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()