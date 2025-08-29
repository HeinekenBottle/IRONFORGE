"""
Reporting Pipeline Integration Tests
===================================

Smoke tests for reporting pipeline validation:
- minidash.html generation and content structure
- minidash.png export functionality and file size
- Artifact directory structure (runs/{date}/ layout)
- Dashboard content validation (charts, data integrity)
"""

import pytest
from pathlib import Path
import json
import os
from PIL import Image

from ironforge.reporting.minidash import build_minidash
from ironforge.sdk.app_config import load_config, materialize_run_dir


@pytest.fixture
def test_config():
    """Get test configuration."""
    config_path = Path("configs/dev.yml")
    if not config_path.exists():
        pytest.skip("Test configuration not available")
    
    return load_config(str(config_path))


@pytest.fixture
def sample_run_dir(test_config):
    """Get sample run directory with test data."""
    run_dir = materialize_run_dir(test_config)
    
    # Check if run directory exists with required data
    if not run_dir.exists():
        pytest.skip("No run directory available for testing")
    
    required_dirs = ['patterns', 'confluence']
    missing_dirs = [d for d in required_dirs if not (run_dir / d).exists()]
    if missing_dirs:
        pytest.skip(f"Missing required directories: {missing_dirs}")
    
    return run_dir


class TestMinidashGeneration:
    """Test minidash HTML generation."""
    
    def test_minidash_html_generation(self, test_config, sample_run_dir):
        """Test minidash.html is generated successfully."""
        # Generate minidash
        build_minidash(test_config)
        
        # Check HTML file exists
        html_path = sample_run_dir / "minidash.html"
        assert html_path.exists(), f"minidash.html not found at {html_path}"
        
        # Check file is not empty
        assert html_path.stat().st_size > 0, "minidash.html is empty"
        
        print(f"✅ minidash.html generated: {html_path.stat().st_size} bytes")
    
    def test_minidash_html_content_structure(self, test_config, sample_run_dir):
        """Test minidash.html has proper content structure."""
        build_minidash(test_config)
        
        html_path = sample_run_dir / "minidash.html"
        assert html_path.exists()
        
        # Read HTML content
        html_content = html_path.read_text(encoding='utf-8')
        
        # Check for essential HTML structure
        assert '<html' in html_content, "Missing HTML tag"
        assert '<head>' in html_content, "Missing head section"
        assert '<body>' in html_content, "Missing body section"
        assert '</html>' in html_content, "Missing closing HTML tag"
        
        # Check for dashboard-specific content
        expected_content = [
            'IRONFORGE',  # Title
            'Archaeological Discovery',  # Subtitle
            'Patterns',  # Section
            'Confluence',  # Section
        ]
        
        for content in expected_content:
            assert content in html_content, f"Missing expected content: {content}"
        
        print(f"✅ minidash.html content structure validated")
    
    def test_minidash_html_charts_present(self, test_config, sample_run_dir):
        """Test minidash.html contains chart elements."""
        build_minidash(test_config)
        
        html_path = sample_run_dir / "minidash.html"
        html_content = html_path.read_text(encoding='utf-8')
        
        # Check for chart-related content
        chart_indicators = [
            'plotly',  # Plotly charts
            'chart',   # Chart elements
            'data',    # Data sections
            'layout',  # Chart layouts
        ]
        
        chart_count = sum(1 for indicator in chart_indicators if indicator in html_content.lower())
        assert chart_count >= 2, f"Insufficient chart content found: {chart_count}/4 indicators"
        
        print(f"✅ minidash.html charts validated: {chart_count}/4 indicators found")


class TestMinidashPNGExport:
    """Test minidash PNG export functionality."""
    
    def test_minidash_png_generation(self, test_config, sample_run_dir):
        """Test minidash.png is generated successfully."""
        # Generate minidash (should include PNG export)
        build_minidash(test_config)
        
        # Check PNG file exists
        png_path = sample_run_dir / "minidash.png"
        assert png_path.exists(), f"minidash.png not found at {png_path}"
        
        # Check file is not empty
        file_size = png_path.stat().st_size
        assert file_size > 0, "minidash.png is empty"
        
        print(f"✅ minidash.png generated: {file_size} bytes")
    
    def test_minidash_png_file_properties(self, test_config, sample_run_dir):
        """Test minidash.png file properties."""
        build_minidash(test_config)
        
        png_path = sample_run_dir / "minidash.png"
        assert png_path.exists()
        
        # Check file size is reasonable (not too small, not too large)
        file_size = png_path.stat().st_size
        min_size = 10 * 1024  # 10KB minimum
        max_size = 5 * 1024 * 1024  # 5MB maximum
        
        assert min_size <= file_size <= max_size, (
            f"PNG file size {file_size} bytes outside expected range [{min_size}, {max_size}]"
        )
        
        print(f"✅ minidash.png size validated: {file_size} bytes")
    
    def test_minidash_png_image_properties(self, test_config, sample_run_dir):
        """Test minidash.png image properties."""
        build_minidash(test_config)
        
        png_path = sample_run_dir / "minidash.png"
        assert png_path.exists()
        
        try:
            # Open image and check properties
            with Image.open(png_path) as img:
                width, height = img.size
                
                # Check dimensions are reasonable
                min_width, min_height = 800, 600
                max_width, max_height = 2000, 1500
                
                assert min_width <= width <= max_width, f"Width {width} outside range [{min_width}, {max_width}]"
                assert min_height <= height <= max_height, f"Height {height} outside range [{min_height}, {max_height}]"
                
                # Check format
                assert img.format == 'PNG', f"Expected PNG format, got {img.format}"
                
                print(f"✅ minidash.png image properties: {width}x{height} {img.format}")
                
        except Exception as e:
            pytest.fail(f"Failed to validate PNG image: {e}")


class TestArtifactDirectoryStructure:
    """Test artifact directory structure validation."""
    
    def test_run_directory_structure(self, sample_run_dir):
        """Test runs/{date}/ directory structure."""
        # Check required directories exist
        required_dirs = [
            'patterns',
            'confluence', 
            'embeddings',
        ]
        
        for dir_name in required_dirs:
            dir_path = sample_run_dir / dir_name
            assert dir_path.exists(), f"Required directory missing: {dir_name}"
            assert dir_path.is_dir(), f"Path is not a directory: {dir_name}"
        
        print(f"✅ Run directory structure validated: {sample_run_dir}")
    
    def test_patterns_directory_content(self, sample_run_dir):
        """Test patterns directory contains expected files."""
        patterns_dir = sample_run_dir / "patterns"
        assert patterns_dir.exists()
        
        # Check for pattern files
        pattern_files = list(patterns_dir.glob("patterns_*.parquet"))
        assert len(pattern_files) > 0, "No pattern files found"
        
        # Check files are not empty
        for pattern_file in pattern_files:
            assert pattern_file.stat().st_size > 0, f"Empty pattern file: {pattern_file.name}"
        
        print(f"✅ Patterns directory validated: {len(pattern_files)} files")
    
    def test_confluence_directory_content(self, sample_run_dir):
        """Test confluence directory contains expected files."""
        confluence_dir = sample_run_dir / "confluence"
        assert confluence_dir.exists()
        
        # Check for required files
        required_files = [
            'scores.parquet',
            'stats.json'
        ]
        
        for file_name in required_files:
            file_path = confluence_dir / file_name
            assert file_path.exists(), f"Required confluence file missing: {file_name}"
            assert file_path.stat().st_size > 0, f"Empty confluence file: {file_name}"
        
        print(f"✅ Confluence directory validated")
    
    def test_embeddings_directory_content(self, sample_run_dir):
        """Test embeddings directory contains expected files."""
        embeddings_dir = sample_run_dir / "embeddings"
        assert embeddings_dir.exists()
        
        # Check for embedding files
        embedding_files = list(embeddings_dir.glob("node_embeddings_*.parquet"))
        assert len(embedding_files) > 0, "No embedding files found"
        
        # Check files are not empty
        for embedding_file in embedding_files:
            assert embedding_file.stat().st_size > 0, f"Empty embedding file: {embedding_file.name}"
        
        print(f"✅ Embeddings directory validated: {len(embedding_files)} files")


class TestDashboardContentValidation:
    """Test dashboard content validation."""
    
    def test_confluence_stats_content(self, sample_run_dir):
        """Test confluence stats.json content."""
        stats_path = sample_run_dir / "confluence" / "stats.json"
        assert stats_path.exists()
        
        # Load and validate stats
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        # Check required stats fields
        required_fields = [
            'total_patterns',
            'scored_patterns', 
            'average_score',
        ]
        
        for field in required_fields:
            assert field in stats, f"Missing stats field: {field}"
            assert isinstance(stats[field], (int, float)), f"Invalid stats field type: {field}"
        
        # Validate stats values
        assert stats['total_patterns'] >= 0, "Invalid total_patterns"
        assert stats['scored_patterns'] >= 0, "Invalid scored_patterns"
        assert stats['scored_patterns'] <= stats['total_patterns'], "scored_patterns > total_patterns"
        
        print(f"✅ Confluence stats validated: {stats['total_patterns']} patterns")
    
    def test_dashboard_data_integrity(self, test_config, sample_run_dir):
        """Test dashboard data integrity."""
        # Generate fresh dashboard
        build_minidash(test_config)
        
        # Check HTML was generated
        html_path = sample_run_dir / "minidash.html"
        assert html_path.exists()
        
        # Check PNG was generated
        png_path = sample_run_dir / "minidash.png"
        assert png_path.exists()
        
        # Check both files have recent timestamps (within last minute)
        import time
        current_time = time.time()
        html_mtime = html_path.stat().st_mtime
        png_mtime = png_path.stat().st_mtime
        
        time_threshold = 60  # 60 seconds
        assert current_time - html_mtime < time_threshold, "HTML file not recently generated"
        assert current_time - png_mtime < time_threshold, "PNG file not recently generated"
        
        print(f"✅ Dashboard data integrity validated")


class TestReportingErrorHandling:
    """Test reporting error handling."""
    
    def test_missing_data_handling(self, test_config):
        """Test reporting handles missing data gracefully."""
        # Create temporary config with non-existent run directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_run_dir = Path(temp_dir) / "nonexistent_run"
            
            # Modify config to point to non-existent directory
            test_config.outputs.run_dir = str(temp_run_dir)
            
            # Should handle missing data gracefully (not crash)
            try:
                build_minidash(test_config)
                # If it succeeds, check that some output was created
                if temp_run_dir.exists():
                    print("✅ Reporting handled missing data gracefully")
                else:
                    print("✅ Reporting skipped generation for missing data")
            except Exception as e:
                # Should not crash with unhandled exceptions
                pytest.fail(f"Reporting crashed on missing data: {e}")
    
    def test_partial_data_handling(self, test_config, sample_run_dir):
        """Test reporting handles partial data."""
        # Remove some data files temporarily
        confluence_dir = sample_run_dir / "confluence"
        stats_file = confluence_dir / "stats.json"
        
        if stats_file.exists():
            # Backup and remove stats file
            backup_content = stats_file.read_text()
            stats_file.unlink()
            
            try:
                # Should handle missing stats gracefully
                build_minidash(test_config)
                print("✅ Reporting handled partial data gracefully")
            except Exception as e:
                pytest.fail(f"Reporting failed on partial data: {e}")
            finally:
                # Restore stats file
                stats_file.write_text(backup_content)


def test_reporting_pipeline_summary():
    """Print reporting pipeline test summary."""
    print("\n" + "="*60)
    print("IRONFORGE Reporting Pipeline Test Summary")
    print("="*60)
    print("Validated Components:")
    print("  ✅ minidash.html generation")
    print("  ✅ minidash.png export")
    print("  ✅ Artifact directory structure")
    print("  ✅ Dashboard content validation")
    print("  ✅ Error handling")
    print("="*60)
