"""Tests for ironforge.reporting.confluence module."""

from io import BytesIO

import numpy as np
import pytest

try:
    from PIL import Image, ImageDraw

    from ironforge.reporting.confluence import ConfluenceStripSpec, build_confluence_strip

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
class TestConfluenceStrip:
    """Test confluence strip generation."""

    def test_confluence_spec_defaults(self):
        """Test default confluence specification."""
        spec = ConfluenceStripSpec()
        assert spec.width == 1024
        assert spec.height == 54
        assert spec.pad == 6
        assert spec.marker_radius == 3

    def test_confluence_spec_custom(self):
        """Test custom confluence specification."""
        spec = ConfluenceStripSpec(width=512, height=32, pad=4, marker_radius=2)
        assert spec.width == 512
        assert spec.height == 32
        assert spec.pad == 4
        assert spec.marker_radius == 2

    def test_build_confluence_strip_basic(self):
        """Test basic confluence strip generation."""
        minute_bins = np.array([0, 5, 10, 15, 20])
        scores_0_100 = np.array([25.0, 50.0, 75.0, 100.0, 60.0])

        strip = build_confluence_strip(minute_bins, scores_0_100)

        # Verify it's a PIL Image
        assert isinstance(strip, Image.Image)

        # Check dimensions
        assert strip.size[0] == 1024  # width
        assert strip.size[1] == 54  # height

        # Check mode
        assert strip.mode == "RGBA"

    def test_build_confluence_strip_with_markers(self):
        """Test confluence strip with event markers."""
        minute_bins = np.array([0, 10, 20, 30])
        scores_0_100 = np.array([30.0, 60.0, 80.0, 40.0])
        marker_minutes = np.array([5, 25])  # Two markers

        strip_without_markers = build_confluence_strip(minute_bins, scores_0_100)
        strip_with_markers = build_confluence_strip(minute_bins, scores_0_100, marker_minutes)

        # Both should be valid images
        assert isinstance(strip_without_markers, Image.Image)
        assert isinstance(strip_with_markers, Image.Image)

        # Convert to arrays for pixel comparison
        pixels_without = np.array(strip_without_markers)
        pixels_with = np.array(strip_with_markers)

        # Images should be different (markers add red pixels)
        assert not np.array_equal(pixels_without, pixels_with)

        # Check that red markers are present in the version with markers
        red_pixels_without = np.sum(pixels_without[:, :, 0] > 200)  # High red values
        red_pixels_with = np.sum(pixels_with[:, :, 0] > 200)
        assert red_pixels_with > red_pixels_without

    def test_build_confluence_strip_custom_spec(self):
        """Test confluence strip with custom specification."""
        minute_bins = np.array([0, 1, 2, 3])
        scores_0_100 = np.array([0.0, 25.0, 75.0, 100.0])

        spec = ConfluenceStripSpec(width=256, height=32, pad=2, marker_radius=1)
        strip = build_confluence_strip(minute_bins, scores_0_100, spec=spec)

        assert strip.size == (256, 32)
        assert strip.mode == "RGBA"

    def test_build_confluence_strip_score_clamping(self):
        """Test that scores are properly clamped to 0-100 range."""
        minute_bins = np.array([0, 1, 2, 3])
        scores_extreme = np.array([-50.0, 25.0, 150.0, 75.0])  # Out of range values

        # Should not raise error and produce valid image
        strip = build_confluence_strip(minute_bins, scores_extreme)
        assert isinstance(strip, Image.Image)
        assert strip.size == (1024, 54)

    def test_build_confluence_strip_empty_data(self):
        """Test confluence strip with empty data."""
        minute_bins = np.array([])
        scores_0_100 = np.array([])

        strip = build_confluence_strip(minute_bins, scores_0_100)
        assert isinstance(strip, Image.Image)
        assert strip.size == (1024, 54)

    def test_build_confluence_strip_single_point(self):
        """Test confluence strip with single data point."""
        minute_bins = np.array([10])
        scores_0_100 = np.array([65.0])

        strip = build_confluence_strip(minute_bins, scores_0_100)
        assert isinstance(strip, Image.Image)
        assert strip.size == (1024, 54)

    def test_build_confluence_strip_no_markers(self):
        """Test confluence strip with None markers."""
        minute_bins = np.array([0, 10, 20])
        scores_0_100 = np.array([30.0, 60.0, 90.0])

        strip = build_confluence_strip(minute_bins, scores_0_100, marker_minutes=None)
        assert isinstance(strip, Image.Image)

    def test_build_confluence_strip_empty_markers(self):
        """Test confluence strip with empty marker array."""
        minute_bins = np.array([0, 10, 20])
        scores_0_100 = np.array([30.0, 60.0, 90.0])
        empty_markers = np.array([])

        strip = build_confluence_strip(minute_bins, scores_0_100, empty_markers)
        assert isinstance(strip, Image.Image)

    def test_build_confluence_strip_marker_positioning(self):
        """Test that markers are positioned correctly."""
        minute_bins = np.array([0, 10, 20, 30, 40])
        scores_0_100 = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
        marker_minutes = np.array([15])  # Single marker at minute 15

        spec = ConfluenceStripSpec(width=100, height=30, pad=0, marker_radius=2)
        strip = build_confluence_strip(minute_bins, scores_0_100, marker_minutes, spec)

        # Convert to array and check for red pixels (markers)
        pixels = np.array(strip)
        red_channel = pixels[:, :, 0]

        # Should have red pixels somewhere in the middle region
        middle_region = red_channel[:, 30:70]  # Approximate middle of image
        assert np.any(middle_region > 200)  # High red values indicate marker

    def test_build_confluence_strip_multiple_markers(self):
        """Test confluence strip with multiple markers."""
        minute_bins = np.array([0, 10, 20, 30, 40, 50])
        scores_0_100 = np.array([10.0, 30.0, 50.0, 70.0, 90.0, 100.0])
        marker_minutes = np.array([5, 25, 45])  # Three markers

        strip = build_confluence_strip(minute_bins, scores_0_100, marker_minutes)

        # Convert to array and count red pixels
        pixels = np.array(strip)
        red_pixels = np.sum(pixels[:, :, 0] > 200)

        # Should have significant red content from multiple markers
        assert red_pixels > 50  # Arbitrary threshold for "significant"

    def test_build_confluence_strip_png_serializable(self):
        """Test that generated strip can be saved as PNG."""
        minute_bins = np.array([0, 15, 30, 45])
        scores_0_100 = np.array([25.0, 50.0, 75.0, 90.0])

        strip = build_confluence_strip(minute_bins, scores_0_100)

        # Try to save to bytes buffer
        buf = BytesIO()
        strip.save(buf, format="PNG")

        # Verify we have PNG data
        assert len(buf.getvalue()) > 0

        # Verify PNG header
        buf.seek(0)
        png_header = buf.read(8)
        expected_png_header = b"\x89PNG\r\n\x1a\n"
        assert png_header == expected_png_header

    def test_build_confluence_strip_gradient_effect(self):
        """Test that different scores produce different grayscale values."""
        # Test with extreme scores
        minute_bins_low = np.array([0])
        scores_low = np.array([0.0])

        minute_bins_high = np.array([0])
        scores_high = np.array([100.0])

        strip_low = build_confluence_strip(minute_bins_low, scores_low)
        strip_high = build_confluence_strip(minute_bins_high, scores_high)

        pixels_low = np.array(strip_low)
        pixels_high = np.array(strip_high)

        # Average grayscale values should be different
        avg_gray_low = np.mean(pixels_low[:, :, 0])  # Red channel (grayscale)
        avg_gray_high = np.mean(pixels_high[:, :, 0])

        # High score should produce brighter pixels
        assert avg_gray_high > avg_gray_low

    def test_build_confluence_strip_assertions(self):
        """Test input validation assertions."""
        # Test mismatched array shapes
        minute_bins = np.array([0, 1, 2])
        scores_0_100 = np.array([50.0, 75.0])  # Wrong length

        with pytest.raises(AssertionError):
            build_confluence_strip(minute_bins, scores_0_100)

        # Test wrong dimensions
        minute_bins_2d = np.array([[0, 1], [2, 3]])
        scores_1d = np.array([50.0, 75.0])

        with pytest.raises(AssertionError):
            build_confluence_strip(minute_bins_2d, scores_1d)


@pytest.mark.skipif(PIL_AVAILABLE, reason="Testing import error handling")
def test_confluence_import_error_handling():
    """Test that proper ImportError is raised when PIL not available."""
    # This test runs when PIL is NOT available
    from ironforge.reporting.confluence import build_confluence_strip

    minute_bins = np.array([0, 1, 2])
    scores_0_100 = np.array([25.0, 50.0, 75.0])

    with pytest.raises(ImportError, match="Wave 5 reporting requires numpy and Pillow"):
        build_confluence_strip(minute_bins, scores_0_100)
