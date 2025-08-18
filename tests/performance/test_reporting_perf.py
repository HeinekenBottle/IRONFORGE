"""Performance tests for ironforge.reporting package."""

import gc
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import psutil
import pytest

try:
    import numpy as np

    from ironforge.reporting.confluence import ConfluenceStripSpec, build_confluence_strip
    from ironforge.reporting.heatmap import TimelineHeatmapSpec, build_session_heatmap
    from ironforge.reporting.html import build_report_html
    from ironforge.reporting.writer import write_html, write_png
    from ironforge.sdk.cli import main
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="Dependencies not available")
class TestReportingPerformance:
    """Performance tests for Wave 5 reporting components."""

    def test_heatmap_generation_performance(self):
        """Test that heatmap generation meets performance budgets."""
        # Create large timeline data (240 minutes = 4 hours)
        minute_bins = np.arange(0, 240, 1)  # Every minute for 4 hours
        densities = np.random.random(len(minute_bins)) * 5.0
        
        spec = TimelineHeatmapSpec(width=1024, height=160)
        
        # Measure memory before
        gc.collect()  # Force garbage collection
        mem_before = get_memory_usage()
        
        start_time = time.time()
        
        # Generate heatmap
        heatmap = build_session_heatmap(minute_bins, densities, spec)
        
        end_time = time.time()
        
        # Measure memory after
        mem_after = get_memory_usage()
        memory_used = mem_after - mem_before
        
        # Performance assertions
        execution_time = end_time - start_time
        assert execution_time < 1.0, f"Heatmap generation took {execution_time:.3f}s, should be <1.0s"
        assert memory_used < 50, f"Heatmap used {memory_used:.1f}MB, should be <50MB"
        
        # Verify output quality
        assert heatmap.size == (1024, 160)
        assert heatmap.mode == "RGBA"

    def test_confluence_strip_performance(self):
        """Test that confluence strip generation meets performance budgets."""
        # Create large timeline data
        minute_bins = np.arange(0, 240, 1)
        scores_0_100 = np.random.random(len(minute_bins)) * 100
        marker_minutes = np.random.choice(minute_bins, 10, replace=False)  # 10 random markers
        
        spec = ConfluenceStripSpec(width=1024, height=54)
        
        # Measure memory before
        gc.collect()
        mem_before = get_memory_usage()
        
        start_time = time.time()
        
        # Generate confluence strip
        strip = build_confluence_strip(minute_bins, scores_0_100, marker_minutes, spec)
        
        end_time = time.time()
        
        # Measure memory after
        mem_after = get_memory_usage()
        memory_used = mem_after - mem_before
        
        # Performance assertions
        execution_time = end_time - start_time
        assert execution_time < 1.0, f"Confluence strip took {execution_time:.3f}s, should be <1.0s"
        assert memory_used < 50, f"Confluence strip used {memory_used:.1f}MB, should be <50MB"
        
        # Verify output quality
        assert strip.size == (1024, 54)
        assert strip.mode == "RGBA"

    def test_multi_session_performance(self):
        """Test performance with multiple sessions (5 sessions × 240 minutes)."""
        # Create data for 5 sessions, each 4 hours long
        sessions_data = {}
        for i in range(5):
            session_id = f"session_{i:02d}"
            minute_bins = np.arange(0, 240, 1)
            densities = np.random.random(len(minute_bins)) * 3.0
            confluence = np.random.random(len(minute_bins)) * 100
            markers = np.random.choice(minute_bins, 5, replace=False)
            
            sessions_data[session_id] = {
                "minute_bins": minute_bins,
                "densities": densities, 
                "confluence": confluence,
                "markers": markers
            }
        
        # Measure memory before
        gc.collect()
        mem_before = get_memory_usage()
        
        start_time = time.time()
        
        # Generate all reports
        images = []
        for session_id, data in sessions_data.items():
            heatmap = build_session_heatmap(
                data["minute_bins"],
                data["densities"]
            )
            strip = build_confluence_strip(
                data["minute_bins"],
                data["confluence"],
                data["markers"]
            )
            images.append((f"{session_id} — timeline", heatmap))
            images.append((f"{session_id} — confluence", strip))
        
        # Generate HTML
        html = build_report_html("Performance Test Report", images)
        
        end_time = time.time()
        
        # Measure memory after
        mem_after = get_memory_usage()
        memory_used = mem_after - mem_before
        
        # Performance assertions (budget: <2s, <150MB)
        execution_time = end_time - start_time
        assert execution_time < 2.0, f"Multi-session took {execution_time:.3f}s, should be <2.0s"
        assert memory_used < 150, f"Multi-session used {memory_used:.1f}MB, should be <150MB"
        
        # Verify output
        assert len(images) == 10  # 5 sessions × 2 images each
        assert "Performance Test Report" in html
        assert len(html) > 1000  # Should be substantial HTML

    @pytest.mark.slow
    def test_file_writing_performance(self):
        """Test file I/O performance with large reports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Generate test data
            minute_bins = np.arange(0, 240, 2)  # Every 2 minutes
            densities = np.random.random(len(minute_bins)) * 4.0
            
            heatmap = build_session_heatmap(minute_bins, densities)
            strip = build_confluence_strip(
                minute_bins, 
                np.random.random(len(minute_bins)) * 100
            )
            
            # Test PNG writing performance
            start_time = time.time()
            
            png_path1 = write_png(temp_path / "test_heatmap.png", heatmap)
            png_path2 = write_png(temp_path / "test_strip.png", strip)
            
            png_time = time.time() - start_time
            
            # Test HTML writing performance
            images = [("Test heatmap", heatmap), ("Test strip", strip)]
            html = build_report_html("Test Report", images)
            
            start_time = time.time()
            html_path = write_html(temp_path / "test_report.html", html)
            html_time = time.time() - start_time
            
            # Performance assertions
            assert png_time < 1.0, f"PNG writing took {png_time:.3f}s, should be <1.0s"
            assert html_time < 0.5, f"HTML writing took {html_time:.3f}s, should be <0.5s"
            
            # Verify files exist and have reasonable sizes
            assert png_path1.exists()
            assert png_path2.exists()
            assert html_path.exists()
            
            assert png_path1.stat().st_size > 1000  # Should be substantial PNG
            assert png_path2.stat().st_size > 1000
            assert html_path.stat().st_size > 10000  # HTML with embedded images

    def test_memory_cleanup_after_generation(self):
        """Test that memory is properly cleaned up after image generation."""
        initial_memory = get_memory_usage()
        
        # Generate and discard many images
        for _ in range(20):
            minute_bins = np.arange(0, 60, 1)
            densities = np.random.random(len(minute_bins))
            
            heatmap = build_session_heatmap(minute_bins, densities)
            strip = build_confluence_strip(minute_bins, densities * 100)
            
            # Force Python to release the images
            del heatmap, strip
        
        # Force garbage collection
        gc.collect()
        
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Should not leak significant memory
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB, should be <100MB"

    @patch("ironforge.sdk.cli.json.loads")
    @patch("ironforge.sdk.cli.build_session_heatmap")
    @patch("ironforge.sdk.cli.build_confluence_strip")
    @patch("ironforge.sdk.cli.write_png")
    @patch("ironforge.sdk.cli.write_html")
    @patch("ironforge.sdk.cli.build_report_html")
    def test_cli_end_to_end_performance(
        self, mock_build_html, mock_write_html, mock_write_png,
        mock_build_confluence, mock_build_heatmap, mock_json_loads
    ):
        """Test end-to-end CLI performance with mocked heavy operations."""
        # Create large dataset
        large_data = {}
        for i in range(5):
            session_id = f"large_session_{i}"
            large_data[session_id] = {
                "minute_bins": list(range(0, 240, 1)),  # 240 minutes
                "densities": list(np.random.random(240) * 5.0),
                "confluence": list(np.random.random(240) * 100),
                "markers": [10, 50, 100, 150, 200]
            }
        
        # Setup mocks to simulate realistic timing
        def slow_heatmap(*args, **kwargs):
            time.sleep(0.1)  # Simulate 100ms per heatmap
            return object()
        
        def slow_confluence(*args, **kwargs):
            time.sleep(0.05)  # Simulate 50ms per strip
            return object()
        
        mock_json_loads.return_value = large_data
        mock_build_heatmap.side_effect = slow_heatmap
        mock_build_confluence.side_effect = slow_confluence
        mock_write_png.return_value = Path("test.png")
        mock_build_html.return_value = "<html>test</html>"
        mock_write_html.return_value = Path("index.html")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = Path(temp_dir) / "large_data.json"
            json_file.write_text(json.dumps(large_data))
            
            # Measure performance
            start_time = time.time()
            
            with patch.object(Path, 'exists', return_value=True):
                result = main([
                    "report",
                    "--input-json", str(json_file),
                    "--out-dir", temp_dir,
                    "--html"
                ])
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should complete successfully
            assert result == 0
            
            # Should complete within budget (allowing for mocked delays)
            expected_max_time = (0.1 * 5) + (0.05 * 5) + 1.0  # Mocked delays + 1s buffer
            assert execution_time < expected_max_time, f"CLI took {execution_time:.3f}s, expected <{expected_max_time:.1f}s"
            
            # Verify all components were called
            assert mock_build_heatmap.call_count == 5
            assert mock_build_confluence.call_count == 5
            assert mock_write_png.call_count == 10  # 5 heatmaps + 5 strips

    def test_large_image_dimensions_performance(self):
        """Test performance with larger image dimensions."""
        minute_bins = np.arange(0, 120, 1)  # 2 hours
        densities = np.random.random(len(minute_bins)) * 3.0
        
        # Test with larger dimensions
        large_spec = TimelineHeatmapSpec(width=2048, height=320, pad=16)
        
        gc.collect()
        mem_before = get_memory_usage()
        start_time = time.time()
        
        heatmap = build_session_heatmap(minute_bins, densities, large_spec)
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
        execution_time = end_time - start_time
        memory_used = mem_after - mem_before
        
        # Larger images should still be reasonable
        assert execution_time < 2.0, f"Large heatmap took {execution_time:.3f}s, should be <2.0s"
        assert memory_used < 100, f"Large heatmap used {memory_used:.1f}MB, should be <100MB"
        
        # Verify correct dimensions
        assert heatmap.size == (2048, 320)

    @pytest.mark.skipif(psutil.cpu_count() < 2, reason="Requires multi-core system")
    def test_concurrent_generation_performance(self):
        """Test performance when generating multiple images concurrently."""
        import concurrent.futures
        
        def generate_session_images(session_id):
            """Generate images for a single session."""
            minute_bins = np.arange(0, 60, 1)
            densities = np.random.random(len(minute_bins)) * 2.0
            confluence = np.random.random(len(minute_bins)) * 100
            
            heatmap = build_session_heatmap(minute_bins, densities)
            strip = build_confluence_strip(minute_bins, confluence)
            
            return session_id, heatmap, strip
        
        start_time = time.time()
        
        # Generate 4 sessions concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(generate_session_images, f"session_{i}")
                for i in range(4)
            ]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Concurrent generation should be faster than sequential
        # (Though this depends on PIL's thread safety and GIL)
        assert execution_time < 3.0, f"Concurrent generation took {execution_time:.3f}s, should be <3.0s"
        assert len(results) == 4
        
        # Verify all images were generated successfully
        for session_id, heatmap, strip in results:
            assert heatmap.size == (1024, 160)
            assert strip.size == (1024, 54)


@pytest.mark.skipif(DEPS_AVAILABLE, reason="Testing performance without dependencies")
def test_performance_graceful_degradation():
    """Test that performance tests gracefully handle missing dependencies."""
    # This test should run when dependencies are NOT available
    with pytest.raises(ImportError):
        from ironforge.reporting.heatmap import build_session_heatmap
        build_session_heatmap(np.array([]), np.array([]))
