"""
Test Suite for IRONFORGE Pipeline Performance Monitor Agent

Tests the main agent functionality, pipeline integration, and performance
monitoring across all IRONFORGE stages.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import (
    PipelinePerformanceMonitorAgent, 
    PipelineStageMetrics,
    PipelineHealthStatus,
    create_pipeline_monitor
)
from ironforge_config import IRONFORGEPerformanceConfig, StageThresholds
from contracts import ContractViolation, ContractSeverity


class TestPipelineStageMetrics:
    """Test suite for PipelineStageMetrics data class."""
    
    def test_initialization(self):
        """Test proper initialization of stage metrics."""
        metrics = PipelineStageMetrics("discovery")
        
        assert metrics.stage_name == "discovery"
        assert metrics.processing_times == []
        assert metrics.memory_snapshots == []
        assert metrics.quality_scores == []
        assert metrics.error_count == 0
        assert metrics.bottlenecks_detected == []
    
    def test_add_processing_time_within_threshold(self):
        """Test adding processing time within threshold."""
        metrics = PipelineStageMetrics("discovery")
        
        # Add time within threshold (no warning expected)
        with patch('builtins.print') as mock_print:
            metrics.add_processing_time(2.5, 3.0)
        
        assert len(metrics.processing_times) == 1
        assert metrics.processing_times[0] == 2.5
        # No warning should be printed
        mock_print.assert_not_called()
    
    def test_add_processing_time_exceeds_threshold(self):
        """Test adding processing time that exceeds threshold."""
        metrics = PipelineStageMetrics("discovery")
        
        # Add time exceeding threshold
        with patch('agents.pipeline_performance_monitor.agent.logger') as mock_logger:
            metrics.add_processing_time(4.5, 3.0)
        
        assert len(metrics.processing_times) == 1
        assert metrics.processing_times[0] == 4.5
        mock_logger.warning.assert_called_once()
    
    def test_get_average_time(self):
        """Test calculation of average processing time."""
        metrics = PipelineStageMetrics("discovery")
        
        # Empty list should return 0
        assert metrics.get_average_time() == 0.0
        
        # Add some times
        metrics.processing_times = [2.0, 3.0, 4.0]
        assert metrics.get_average_time() == 3.0
    
    def test_get_compliance_rate(self):
        """Test compliance rate calculation."""
        metrics = PipelineStageMetrics("discovery")
        
        # Empty list should return 1.0 (100% compliance)
        assert metrics.get_compliance_rate(3.0) == 1.0
        
        # Test with mixed compliance
        metrics.processing_times = [2.0, 2.5, 3.5, 4.0]  # 2 under, 2 over threshold of 3.0
        assert metrics.get_compliance_rate(3.0) == 0.5


class TestPipelineHealthStatus:
    """Test suite for PipelineHealthStatus data class."""
    
    def test_initialization(self):
        """Test proper initialization of health status."""
        health = PipelineHealthStatus(status="GREEN", overall_score=0.95)
        
        assert health.status == "GREEN"
        assert health.overall_score == 0.95
        assert health.stage_scores == {}
        assert health.active_issues == []
        assert health.trending == "STABLE"
        assert isinstance(health.last_assessment, datetime)


class TestPipelinePerformanceMonitorAgent:
    """Test suite for the main performance monitoring agent."""
    
    @pytest.fixture
    def config(self):
        """Fixture providing test configuration."""
        return IRONFORGEPerformanceConfig()
    
    @pytest.fixture
    def agent(self, config):
        """Fixture providing test agent instance."""
        return PipelinePerformanceMonitorAgent(config)
    
    def test_agent_initialization(self, agent):
        """Test proper agent initialization."""
        assert agent.config is not None
        assert len(agent.stage_metrics) == 4  # discovery, confluence, validation, reporting
        assert "discovery" in agent.stage_metrics
        assert "confluence" in agent.stage_metrics
        assert "validation" in agent.stage_metrics
        assert "reporting" in agent.stage_metrics
        
        assert agent.current_health.status == "GREEN"
        assert agent.current_health.overall_score == 1.0
        assert not agent.container_initialized
        assert not agent.monitoring_active
    
    @pytest.mark.asyncio
    async def test_agent_initialization_success(self, agent):
        """Test successful agent initialization."""
        with patch('agents.pipeline_performance_monitor.agent.initialize_ironforge_lazy_loading') as mock_init:
            mock_init.return_value = Mock()
            
            result = await agent.initialize()
            
            assert result is True
            assert agent.container_initialized is True
            assert agent.monitoring_active is True
            assert agent.container_init_time is not None
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_initialization_failure(self, agent):
        """Test agent initialization failure."""
        with patch('agents.pipeline_performance_monitor.agent.initialize_ironforge_lazy_loading') as mock_init:
            mock_init.side_effect = Exception("Initialization failed")
            
            result = await agent.initialize()
            
            assert result is False
            assert agent.container_initialized is False
    
    @pytest.mark.asyncio
    async def test_agent_initialization_timeout(self, agent):
        """Test agent initialization timeout."""
        # Configure very strict initialization threshold
        agent.config.stage_thresholds.initialization_seconds = 0.001
        
        with patch('agents.pipeline_performance_monitor.agent.initialize_ironforge_lazy_loading') as mock_init:
            mock_init.return_value = Mock()
            # Add a delay to simulate slow initialization
            def slow_init():
                time.sleep(0.01)  # 10ms delay, exceeds 1ms threshold
                return Mock()
            mock_init.side_effect = slow_init
            
            result = await agent.initialize()
            
            assert result is False  # Should fail due to timeout
    
    def test_monitoring_lifecycle(self, agent):
        """Test monitoring start/stop lifecycle."""
        assert not agent.monitoring_active
        
        # Start monitoring
        agent.start_monitoring()
        assert agent.monitoring_active is True
        assert agent.monitor_thread is not None
        assert agent.monitor_thread.is_alive()
        
        # Stop monitoring
        agent.stop_monitoring()
        assert agent.monitoring_active is False
        # Thread should stop within reasonable time
        agent.monitor_thread.join(timeout=3.0)
        assert not agent.monitor_thread.is_alive()
    
    def test_monitor_pipeline_stage_success(self, agent):
        """Test successful pipeline stage monitoring."""
        with agent.monitor_pipeline_stage("discovery", session_count=2) as metrics:
            # Simulate some work
            time.sleep(0.01)  # 10ms of work
            assert metrics.stage_name == "discovery"
        
        # Check that metrics were recorded
        discovery_metrics = agent.stage_metrics["discovery"]
        assert len(discovery_metrics.processing_times) == 1
        assert discovery_metrics.processing_times[0] > 0
        assert len(discovery_metrics.memory_snapshots) == 1
    
    def test_monitor_pipeline_stage_invalid_stage(self, agent):
        """Test monitoring invalid pipeline stage."""
        with pytest.raises(ValueError, match="Unknown stage"):
            with agent.monitor_pipeline_stage("invalid_stage"):
                pass
    
    def test_monitor_pipeline_stage_threshold_violation(self, agent):
        """Test stage monitoring with threshold violation."""
        # Set very strict threshold
        agent.config.stage_thresholds.discovery_seconds = 0.001  # 1ms
        
        with patch.object(agent, '_trigger_alert') as mock_alert:
            with agent.monitor_pipeline_stage("discovery") as metrics:
                time.sleep(0.01)  # 10ms, exceeds 1ms threshold
        
        # Should trigger timeout alert
        mock_alert.assert_called()
        call_args = mock_alert.call_args
        assert call_args[0][0] == "stage_timeout"
        assert call_args[0][1]["stage"] == "discovery"
    
    @pytest.mark.asyncio
    async def test_monitor_full_pipeline_run_success(self, agent):
        """Test successful full pipeline monitoring."""
        # Mock the stage execution functions
        with patch.object(agent, '_run_discovery_stage') as mock_discovery, \
             patch.object(agent, '_run_confluence_stage') as mock_confluence, \
             patch.object(agent, '_run_validation_stage') as mock_validation, \
             patch.object(agent, '_run_reporting_stage') as mock_reporting:
            
            # Configure successful mock responses
            mock_discovery.return_value = {
                "success": True, 
                "pattern_count": 10, 
                "authenticity_scores": [0.89, 0.91, 0.88]
            }
            mock_confluence.return_value = {
                "success": True, 
                "pattern_count": 10, 
                "average_score": 0.75
            }
            mock_validation.return_value = {
                "success": True, 
                "contracts_checked": 8, 
                "gates_passed": True
            }
            mock_reporting.return_value = {
                "success": True, 
                "dashboard_path": "/path/to/dashboard.html", 
                "export_completed": True
            }
            
            # Create mock config
            mock_config = Mock()
            
            result = await agent.monitor_full_pipeline_run("test_config.yml")
            
            # Verify results structure
            assert "stages" in result
            assert "overall_metrics" in result
            assert "quality_metrics" in result
            assert "contract_compliance" in result
            assert "optimization_opportunities" in result
            
            # Verify all stages were called
            assert all(stage in result["stages"] for stage in ["discovery", "confluence", "validation", "reporting"])
            
            # Verify successful execution
            for stage_result in result["stages"].values():
                assert stage_result["success"] is True
    
    def test_health_assessment(self, agent):
        """Test pipeline health assessment."""
        # Add some test data
        agent.stage_metrics["discovery"].processing_times = [2.5, 2.8, 2.6]  # Good performance
        agent.stage_metrics["confluence"].processing_times = [35.0, 40.0, 38.0]  # Poor performance (>30s target)
        
        # Trigger health assessment
        agent._assess_pipeline_health()
        
        # Check health status
        assert agent.current_health.status in ["GREEN", "YELLOW", "RED"]
        assert 0.0 <= agent.current_health.overall_score <= 1.0
        assert len(agent.current_health.stage_scores) > 0
    
    def test_performance_regression_detection(self, agent):
        """Test detection of performance regression."""
        # Simulate historical data with good performance
        for i in range(15):
            historical_profile = Mock()
            historical_profile.stage_timings = {"discovery": [2.0, 2.1, 1.9]}
            agent.performance_history.append(historical_profile)
        
        # Test current performance with regression
        current_metrics = {"discovery": [4.0, 4.2, 3.8]}  # 2x slower than historical
        
        regressions = agent._detect_performance_regression(current_metrics)
        
        assert len(regressions) > 0
        regression = regressions[0]
        assert regression.stage_name == "discovery"
        assert regression.bottleneck_type == "regression"
    
    def test_alert_system(self, agent):
        """Test performance alert system."""
        alert_triggered = False
        alert_data = None
        
        def test_callback(alert_type, data):
            nonlocal alert_triggered, alert_data
            alert_triggered = True
            alert_data = (alert_type, data)
        
        agent.add_alert_callback(test_callback)
        
        # Trigger an alert
        agent._trigger_alert("test_alert", {"test_key": "test_value"})
        
        assert alert_triggered is True
        assert alert_data[0] == "test_alert"
        assert alert_data[1]["data"]["test_key"] == "test_value"
    
    def test_performance_summary_generation(self, agent):
        """Test generation of performance summary."""
        # Add some test data
        agent.stage_metrics["discovery"].processing_times = [2.5, 2.8, 2.6]
        agent.current_health.status = "GREEN"
        agent.current_health.overall_score = 0.95
        
        summary = agent.get_performance_summary()
        
        assert "pipeline_health" in summary
        assert "stage_performance" in summary
        assert "system_performance" in summary
        assert "contract_compliance" in summary
        
        # Verify health information
        assert summary["pipeline_health"]["status"] == "GREEN"
        assert summary["pipeline_health"]["overall_score"] == 0.95
        
        # Verify stage performance information
        assert "discovery" in summary["stage_performance"]
        discovery_perf = summary["stage_performance"]["discovery"]
        assert "average_time" in discovery_perf
        assert "compliance_rate" in discovery_perf
        assert discovery_perf["total_executions"] == 3


class TestAgentIntegration:
    """Integration tests for agent with IRONFORGE components."""
    
    @pytest.fixture
    def mock_ironforge_components(self):
        """Mock IRONFORGE components for testing."""
        with patch('agents.pipeline_performance_monitor.agent.run_discovery') as mock_discovery, \
             patch('agents.pipeline_performance_monitor.agent.score_confluence') as mock_confluence, \
             patch('agents.pipeline_performance_monitor.agent.validate_run') as mock_validation, \
             patch('agents.pipeline_performance_monitor.agent.build_minidash') as mock_reporting:
            
            yield {
                'discovery': mock_discovery,
                'confluence': mock_confluence,
                'validation': mock_validation,
                'reporting': mock_reporting
            }
    
    @pytest.mark.asyncio
    async def test_discovery_stage_integration(self, mock_ironforge_components):
        """Test integration with IRONFORGE discovery stage."""
        config = IRONFORGEPerformanceConfig()
        agent = PipelinePerformanceMonitorAgent(config)
        
        # Configure mock discovery response
        mock_ironforge_components['discovery'].return_value = {
            "patterns_discovered": 15,
            "authenticity_scores": [0.89, 0.91, 0.88, 0.90]
        }
        
        # Test discovery stage execution
        mock_config = Mock()
        result = await agent._run_discovery_stage(mock_config)
        
        assert result["success"] is True
        assert result["pattern_count"] == 15
        assert len(result["authenticity_scores"]) == 4
        mock_ironforge_components['discovery'].assert_called_once_with(mock_config)
    
    @pytest.mark.asyncio
    async def test_confluence_stage_integration(self, mock_ironforge_components):
        """Test integration with IRONFORGE confluence stage."""
        config = IRONFORGEPerformanceConfig()
        agent = PipelinePerformanceMonitorAgent(config)
        
        # Configure mock confluence response
        mock_ironforge_components['confluence'].return_value = {
            "patterns_scored": 12,
            "average_confluence_score": 0.78
        }
        
        mock_config = Mock()
        result = await agent._run_confluence_stage(mock_config)
        
        assert result["success"] is True
        assert result["pattern_count"] == 12
        assert result["average_score"] == 0.78
    
    @pytest.mark.asyncio
    async def test_stage_failure_handling(self, mock_ironforge_components):
        """Test handling of stage execution failures."""
        config = IRONFORGEPerformanceConfig()
        agent = PipelinePerformanceMonitorAgent(config)
        
        # Configure mock to raise exception
        mock_ironforge_components['discovery'].side_effect = Exception("Discovery failed")
        
        mock_config = Mock()
        result = await agent._run_discovery_stage(mock_config)
        
        assert result["success"] is False
        assert "error" in result
        assert "Discovery failed" in result["error"]


class TestFactoryFunctions:
    """Test factory functions and module-level functionality."""
    
    def test_create_pipeline_monitor_default_config(self):
        """Test creating monitor with default configuration."""
        monitor = create_pipeline_monitor()
        
        assert isinstance(monitor, PipelinePerformanceMonitorAgent)
        assert monitor.config is not None
        assert len(monitor.alert_callbacks) == 1  # Default alert handler
    
    def test_create_pipeline_monitor_custom_config(self):
        """Test creating monitor with custom configuration."""
        config_dict = {
            "stage_thresholds": {
                "session_processing_seconds": 5.0,
                "memory_limit_mb": 150.0
            }
        }
        
        # Create temporary config file
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.safe_dump(config_dict, f)
            config_path = f.name
        
        try:
            monitor = create_pipeline_monitor(config_path)
            
            assert monitor.config.stage_thresholds.session_processing_seconds == 5.0
            assert monitor.config.stage_thresholds.memory_limit_mb == 150.0
        
        finally:
            # Clean up temporary file
            Path(config_path).unlink()


@pytest.mark.performance
class TestPerformanceRequirements:
    """Test performance requirements and contracts."""
    
    @pytest.fixture
    def agent(self):
        """Agent fixture for performance tests."""
        config = IRONFORGEPerformanceConfig()
        return PipelinePerformanceMonitorAgent(config)
    
    @pytest.mark.asyncio
    async def test_initialization_performance(self, agent):
        """Test that initialization meets <2s requirement."""
        with patch('agents.pipeline_performance_monitor.agent.initialize_ironforge_lazy_loading') as mock_init:
            mock_init.return_value = Mock()
            
            start_time = time.time()
            result = await agent.initialize()
            init_time = time.time() - start_time
            
            assert result is True
            assert init_time < 2.0  # Must initialize within 2 seconds
            assert agent.container_init_time < 2.0
    
    def test_monitoring_overhead(self, agent):
        """Test that monitoring overhead is sub-millisecond."""
        # Measure monitoring overhead
        start_time = time.perf_counter()
        
        with agent.monitor_pipeline_stage("discovery"):
            # Simulate minimal work
            pass
        
        overhead_time = time.perf_counter() - start_time
        
        # Monitoring overhead should be less than 1ms
        assert overhead_time < 0.001
    
    def test_memory_usage(self, agent):
        """Test that agent memory usage is minimal."""
        import psutil
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Initialize agent and run some operations
        with patch('agents.pipeline_performance_monitor.agent.initialize_ironforge_lazy_loading'):
            asyncio.run(agent.initialize())
        
        # Add some test data
        for i in range(100):
            with agent.monitor_pipeline_stage("discovery"):
                time.sleep(0.001)  # 1ms work
        
        # Measure final memory
        gc.collect()
        final_memory = process.memory_info().rss
        agent_memory_mb = (final_memory - initial_memory) / (1024 * 1024)
        
        # Agent should use less than 10MB additional memory
        assert agent_memory_mb < 10.0
    
    def test_concurrent_monitoring(self, agent):
        """Test concurrent monitoring performance."""
        import concurrent.futures
        
        def monitor_stage(stage_name, duration):
            with agent.monitor_pipeline_stage(stage_name):
                time.sleep(duration)
            return stage_name
        
        start_time = time.time()
        
        # Run multiple stages concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(monitor_stage, "discovery", 0.1),
                executor.submit(monitor_stage, "confluence", 0.05),
                executor.submit(monitor_stage, "validation", 0.02),
                executor.submit(monitor_stage, "reporting", 0.08)
            ]
            
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        # Should complete concurrently in reasonable time
        assert len(results) == 4
        assert total_time < 0.2  # Should be close to the longest individual time (0.1s)
        
        # All stages should have recorded metrics
        for stage in ["discovery", "confluence", "validation", "reporting"]:
            assert len(agent.stage_metrics[stage].processing_times) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])