"""
BMAD-IRONFORGE Integration Bridge

This module provides seamless integration between BMAD (Bio-inspired Market
Archaeological Discovery) temporal metamorphosis workflows and the IRONFORGE
canonical pipeline.

Integration Features:
- Automatic BMAD workflow coordination during confluence scoring
- Real-time pattern metamorphosis detection and alerts
- Multi-agent coordination for enhanced pattern analysis
- Performance monitoring with sub-3-second processing targets
- Statistical validation and quality gates enforcement
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json

from ironforge.coordination.bmad_workflows import (
    BMadCoordinationWorkflow,
    AgentConsensusInput,
    CoordinationResults
)
from ironforge.monitoring import get_performance_tracker

try:
    from archon.python.src.agents.workflow.bmad_template import (
        create_bmad_workflow_instance,
        BMADPatternMetamorphosis
    )
    _ARCHON_AVAILABLE = True
except ImportError:
    _ARCHON_AVAILABLE = False

logger = logging.getLogger(__name__)


class BMADIronforgeIntegration:
    """
    Integration bridge between BMAD workflows and IRONFORGE pipeline.
    
    This class orchestrates BMAD temporal metamorphosis detection within
    the IRONFORGE canonical pipeline, providing:
    - Automatic workflow coordination during pattern analysis
    - Real-time metamorphosis alerts and notifications
    - Performance monitoring and optimization
    - Quality gates enforcement and statistical validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # BMAD coordination workflow
        self.bmad_workflow = BMadCoordinationWorkflow()
        
        # Integration state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.metamorphosis_events: List[Dict[str, Any]] = []
        self.performance_tracker = get_performance_tracker()
        
        # Configuration from BMAD research
        self.bmad_config = BMADPatternMetamorphosis(
            minimum_strength=self.config.get('metamorphosis_threshold', 0.213),
            strong_strength=self.config.get('strong_metamorphosis_threshold', 0.237),
            consensus_threshold=self.config.get('consensus_threshold', 0.75),
            targeting_completion_goal=1.0  # 100% targeting completion
        )
        
        self.logger.info("🧬 BMAD-IRONFORGE Integration initialized")
        self.logger.info(f"   Metamorphosis thresholds: {self.bmad_config.minimum_strength:.1%} - {self.bmad_config.strong_strength:.1%}")
        self.logger.info(f"   Consensus threshold: {self.bmad_config.consensus_threshold:.1%}")
    
    async def coordinate_pattern_analysis(
        self, 
        pattern_paths: List[str],
        session_context: Dict[str, Any],
        research_question: str = "How do temporal patterns evolve and transform across different market phases?"
    ) -> CoordinationResults:
        """
        Coordinate BMAD multi-agent analysis of patterns.
        
        Args:
            pattern_paths: List of pattern file paths for analysis
            session_context: Session metadata and configuration
            research_question: Research question for agent coordination
            
        Returns:
            CoordinationResults with comprehensive multi-agent analysis
        """
        
        session_id = session_context.get('session_id', f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        with self.performance_tracker.track_session_processing(session_id):
            self.logger.info(f"🤝 BMAD Multi-Agent Coordination: {session_id}")
            self.logger.info(f"   Patterns: {len(pattern_paths)}")
            self.logger.info(f"   Research: {research_question}")
            
            # Prepare agent consensus input
            consensus_input = AgentConsensusInput(
                research_question=research_question,
                hypothesis_parameters={
                    'pattern_evolution_metrics': ['transformation_strength', 'phase_adaptation', 'temporal_consistency'],
                    'metamorphosis_types': ['gradual_evolution', 'sudden_transformation', 'phase_adaptation'],
                    'temporal_windows': [300, 900, 1800, 3600],  # 5min, 15min, 30min, 1hr
                    'evolution_sensitivity': [0.3, 0.5, 0.7, 0.9],
                    'pattern_paths': pattern_paths
                },
                session_data={
                    'pattern_paths': pattern_paths,
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat(),
                    **session_context
                },
                analysis_objective='temporal_pattern_metamorphosis_detection',
                participating_agents=[
                    'data-scientist',
                    'adjacent-possible-linker', 
                    'knowledge-architect',
                    'scrum-master'
                ],
                consensus_threshold=self.bmad_config.consensus_threshold,
                targeting_completion_goal=self.bmad_config.targeting_completion_goal,
                minimum_confidence=0.7,
                statistical_significance_required=True,
                cross_agent_validation=True
            )\n            \n            # Execute BMAD coordination\n            coordination_results = await self.bmad_workflow.execute_bmad_coordination(consensus_input)\n            \n            # Store session state\n            self.active_sessions[session_id] = {\n                'timestamp': datetime.now().isoformat(),\n                'pattern_count': len(pattern_paths),\n                'coordination_results': coordination_results,\n                'session_context': session_context\n            }\n            \n            # Check for metamorphosis events\n            await self._detect_metamorphosis_events(coordination_results, session_id)\n            \n            # Log coordination summary\n            self.logger.info(f\"🎯 BMAD coordination completed: {session_id}\")\n            self.logger.info(f\"   Consensus achieved: {'✅' if coordination_results.consensus_achieved else '❌'}\")\n            self.logger.info(f\"   Targeting completion: {coordination_results.targeting_completion:.1%}\")\n            self.logger.info(f\"   Confidence score: {coordination_results.confidence_score:.1%}\")\n            \n            return coordination_results\n    \n    async def _detect_metamorphosis_events(\n        self, \n        coordination_results: CoordinationResults, \n        session_id: str\n    ):\n        \"\"\"Detect and log metamorphosis events from coordination results.\"\"\"\n        \n        with self.performance_tracker.track_metamorphosis_detection():\n            # Extract metamorphosis strength from agent analyses\n            metamorphosis_strength = 0.0\n            statistical_significance = 1.0\n            \n            # Check data-scientist analysis for statistical metrics\n            if 'data-scientist' in coordination_results.agent_analyses:\n                ds_analysis = coordination_results.agent_analyses['data-scientist']\n                if 'statistical_confidence' in ds_analysis:\n                    metamorphosis_strength = ds_analysis['statistical_confidence']\n                if 'p_value' in ds_analysis:\n                    statistical_significance = ds_analysis['p_value']\n            \n            # Check for metamorphosis threshold breach\n            if metamorphosis_strength >= self.bmad_config.minimum_strength:\n                event_type = 'strong_metamorphosis' if metamorphosis_strength >= self.bmad_config.strong_strength else 'metamorphosis'\n                \n                metamorphosis_event = {\n                    'event_id': f\"metamorphosis_{session_id}_{len(self.metamorphosis_events)}\",\n                    'session_id': session_id,\n                    'timestamp': datetime.now().isoformat(),\n                    'event_type': event_type,\n                    'strength': metamorphosis_strength,\n                    'statistical_significance': statistical_significance,\n                    'consensus_score': coordination_results.overall_consensus_score,\n                    'targeting_completion': coordination_results.targeting_completion,\n                    'breakthrough_insights': coordination_results.breakthrough_insights,\n                    'recommendations': coordination_results.consensus_recommendations\n                }\n                \n                self.metamorphosis_events.append(metamorphosis_event)\n                \n                # Record with performance tracker\n                self.performance_tracker.record_metamorphosis_detection(\n                    metamorphosis_strength, statistical_significance\n                )\n                \n                # Log metamorphosis detection\n                strength_pct = metamorphosis_strength * 100\n                if event_type == 'strong_metamorphosis':\n                    self.logger.warning(f\"🚨 STRONG METAMORPHOSIS DETECTED: {strength_pct:.1f}% strength\")\n                else:\n                    self.logger.info(f\"🔍 Metamorphosis detected: {strength_pct:.1f}% strength\")\n                \n                self.logger.info(f\"   Session: {session_id}\")\n                self.logger.info(f\"   Statistical significance: p={statistical_significance:.3f}\")\n                self.logger.info(f\"   Consensus: {coordination_results.overall_consensus_score:.1%}\")\n    \n    def get_session_analysis(self, session_id: str) -> Optional[Dict[str, Any]]:\n        \"\"\"Get analysis results for a specific session.\"\"\"\n        return self.active_sessions.get(session_id)\n    \n    def get_recent_metamorphosis_events(self, limit: int = 10) -> List[Dict[str, Any]]:\n        \"\"\"Get recent metamorphosis events.\"\"\"\n        return sorted(self.metamorphosis_events, key=lambda x: x['timestamp'], reverse=True)[:limit]\n    \n    def get_integration_status(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive integration status.\"\"\"\n        return {\n            'bmad_integration': {\n                'active_sessions': len(self.active_sessions),\n                'total_metamorphosis_events': len(self.metamorphosis_events),\n                'strong_metamorphosis_events': len([\n                    e for e in self.metamorphosis_events if e['event_type'] == 'strong_metamorphosis'\n                ]),\n                'archon_available': _ARCHON_AVAILABLE\n            },\n            'configuration': {\n                'minimum_strength_threshold': self.bmad_config.minimum_strength,\n                'strong_strength_threshold': self.bmad_config.strong_strength,\n                'consensus_threshold': self.bmad_config.consensus_threshold,\n                'targeting_completion_goal': self.bmad_config.targeting_completion_goal\n            },\n            'recent_sessions': list(self.active_sessions.keys())[-5:],  # Last 5 sessions\n            'performance_summary': self.performance_tracker.get_performance_summary()\n        }\n    \n    async def generate_metamorphosis_report(self, output_path: Optional[str] = None) -> str:\n        \"\"\"Generate comprehensive metamorphosis analysis report.\"\"\"\n        \n        report_data = {\n            'report_metadata': {\n                'timestamp': datetime.now().isoformat(),\n                'integration_version': '1.0.0',\n                'bmad_framework': 'temporal_metamorphosis_detection'\n            },\n            'integration_status': self.get_integration_status(),\n            'metamorphosis_events': self.metamorphosis_events,\n            'session_summaries': {\n                session_id: {\n                    'timestamp': data['timestamp'],\n                    'pattern_count': data['pattern_count'],\n                    'consensus_achieved': data['coordination_results'].consensus_achieved,\n                    'targeting_completion': data['coordination_results'].targeting_completion,\n                    'breakthrough_insights': data['coordination_results'].breakthrough_insights\n                }\n                for session_id, data in self.active_sessions.items()\n            }\n        }\n        \n        # Determine output path\n        if output_path is None:\n            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n            output_path = f\"data/processed/bmad_integration_report_{timestamp}.json\"\n        \n        # Ensure output directory exists\n        output_file = Path(output_path)\n        output_file.parent.mkdir(parents=True, exist_ok=True)\n        \n        # Write report\n        with open(output_file, 'w') as f:\n            json.dump(report_data, f, indent=2)\n        \n        self.logger.info(f\"📊 BMAD Integration Report generated: {output_path}\")\n        return str(output_file)\n\n\n# Global integration instance\n_global_integration: Optional[BMADIronforgeIntegration] = None\n\n\ndef get_bmad_integration(config: Optional[Dict[str, Any]] = None) -> BMADIronforgeIntegration:\n    \"\"\"Get global BMAD-IRONFORGE integration instance.\"\"\"\n    global _global_integration\n    if _global_integration is None:\n        _global_integration = BMADIronforgeIntegration(config)\n    return _global_integration\n\n\nasync def coordinate_bmad_analysis(\n    pattern_paths: List[str],\n    session_context: Dict[str, Any],\n    config: Optional[Dict[str, Any]] = None\n) -> CoordinationResults:\n    \"\"\"Convenience function for BMAD pattern analysis coordination.\"\"\"\n    integration = get_bmad_integration(config)\n    return await integration.coordinate_pattern_analysis(pattern_paths, session_context)\n\n\ndef initialize_bmad_integration(config: Optional[Dict[str, Any]] = None):\n    \"\"\"Initialize BMAD-IRONFORGE integration system.\"\"\"\n    integration = get_bmad_integration(config)\n    logger.info(\"🧬 BMAD-IRONFORGE Integration system initialized\")\n    logger.info(f\"   Integration status: {integration.get_integration_status()['bmad_integration']}\")\n\n\ndef shutdown_bmad_integration():\n    \"\"\"Shutdown BMAD-IRONFORGE integration system.\"\"\"\n    global _global_integration\n    if _global_integration:\n        # Generate final report\n        import asyncio\n        try:\n            loop = asyncio.get_event_loop()\n            if loop.is_running():\n                # Schedule report generation\n                loop.create_task(_global_integration.generate_metamorphosis_report())\n            else:\n                # Run report generation\n                loop.run_until_complete(_global_integration.generate_metamorphosis_report())\n        except Exception as e:\n            logger.warning(f\"Failed to generate final report: {e}\")\n        \n        _global_integration = None\n    \n    logger.info(\"🧬 BMAD-IRONFORGE Integration shutdown complete\")\n