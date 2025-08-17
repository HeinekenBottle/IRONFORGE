"""Tests for ironforge.reporting.heatmap module."""

from io import BytesIO

import numpy as np
import pytest

try:
    from PIL import Image

    from ironforge.reporting.heatmap import TimelineHeatmapSpec, _normalize, build_session_heatmap

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
class TestTimelineHeatmap:
    """Test timeline heatmap generation."""

    def test_heatmap_spec_defaults(self):
        """Test default heatmap specification."""
        spec = TimelineHeatmapSpec()
        assert spec.width == 1024
        assert spec.height == 160
        assert spec.pad == 8
        assert spec.colormap == "viridis"

    def test_heatmap_spec_custom(self):
        """Test custom heatmap specification."""
        spec = TimelineHeatmapSpec(width=512, height=80, pad=4, colormap="plasma")
        assert spec.width == 512
        assert spec.height == 80
        assert spec.pad == 4
        assert spec.colormap == "plasma"

    def test_normalize_function(self):
        """Test the _normalize utility function."""
        # Test normal case
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = _normalize(x)
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0

        # Test constant array
        x_const = np.array([2.0, 2.0, 2.0])
        normalized_const = _normalize(x_const)
        assert np.all(normalized_const == 0.0)

        # Test single value
        x_single = np.array([5.0])
        normalized_single = _normalize(x_single)
        assert normalized_single[0] == 0.0

    def test_build_session_heatmap_basic(self):
        """Test basic heatmap generation."""
        # Create simple test data (T=60 minutes)
        minute_bins = np.array([0, 1, 2, 5, 10, 15, 30, 45, 59])
        densities = np.array([0.0, 2.0, 0.5, 1.8, 3.2, 1.1, 0.8, 2.5, 0.3])

        # Generate heatmap
        hm = build_session_heatmap(minute_bins, densities)

        # Verify it's a PIL Image
        assert isinstance(hm, Image.Image)

        # Check dimensions
        assert hm.size[0] == 1024  # width
        assert hm.size[1] == 160  # height

        # Check mode
        assert hm.mode == "RGBA"

        # Verify non-empty alpha channel (should have content)
        alpha_channel = np.array(hm)[:, :, 3]
        assert np.any(alpha_channel > 0)

    def test_build_session_heatmap_custom_spec(self):
        """Test heatmap generation with custom specification."""
        minute_bins = np.array([0, 1, 2, 3, 4])
        densities = np.array([1.0, 2.0, 0.5, 1.5, 0.8])

        spec = TimelineHeatmapSpec(width=512, height=80, pad=4)
        hm = build_session_heatmap(minute_bins, densities, spec)

        assert hm.size == (512, 80)
        assert hm.mode == "RGBA"

    def test_build_session_heatmap_empty_data(self):
        """Test heatmap generation with empty data."""
        minute_bins = np.array([])
        densities = np.array([])

        hm = build_session_heatmap(minute_bins, densities)
        assert isinstance(hm, Image.Image)
        assert hm.size == (1024, 160)

    def test_build_session_heatmap_single_point(self):
        """Test heatmap generation with single data point."""
        minute_bins = np.array([5])
        densities = np.array([2.5])

        hm = build_session_heatmap(minute_bins, densities)
        assert isinstance(hm, Image.Image)
        assert hm.size == (1024, 160)

    def test_build_session_heatmap_large_timeline(self):
        """Test heatmap generation with large timeline (240 minutes)."""
        minute_bins = np.arange(0, 240, 5)  # Every 5 minutes for 4 hours
        densities = np.random.random(len(minute_bins)) * 3.0

        hm = build_session_heatmap(minute_bins, densities)
        assert isinstance(hm, Image.Image)
        assert hm.size == (1024, 160)

    def test_build_session_heatmap_png_serializable(self):
        """Test that generated heatmap can be saved as PNG."""
        minute_bins = np.array([0, 10, 20, 30])
        densities = np.array([1.0, 2.0, 1.5, 0.8])

        hm = build_session_heatmap(minute_bins, densities)

        # Try to save to bytes buffer
        buf = BytesIO()
        hm.save(buf, format="PNG")

        # Verify we have PNG data
        assert len(buf.getvalue()) > 0

        # Verify PNG header
        buf.seek(0)
        png_header = buf.read(8)
        expected_png_header = b"\x89PNG\r\n\x1a\n"
        assert png_header == expected_png_header

    def test_build_session_heatmap_assertions(self):
        """Test input validation assertions."""
        # Test mismatched array shapes
        minute_bins = np.array([0, 1, 2])
        densities = np.array([1.0, 2.0])  # Wrong length

        with pytest.raises(AssertionError):
            build_session_heatmap(minute_bins, densities)

        # Test wrong dimensions
        minute_bins_2d = np.array([[0, 1], [2, 3]])
        densities_1d = np.array([1.0, 2.0])

        with pytest.raises(AssertionError):
            build_session_heatmap(minute_bins_2d, densities_1d)


@pytest.mark.skipif(PIL_AVAILABLE, reason="Testing import error handling")
def test_import_error_handling():
    """Test that proper ImportError is raised when PIL not available."""
    # This test runs when PIL is NOT available
    from ironforge.reporting.heatmap import build_session_heatmap

    minute_bins = np.array([0, 1, 2])
    densities = np.array([1.0, 2.0, 1.5])

    with pytest.raises(ImportError, match="Wave 5 reporting requires numpy and Pillow"):
        build_session_heatmap(minute_bins, densities)
