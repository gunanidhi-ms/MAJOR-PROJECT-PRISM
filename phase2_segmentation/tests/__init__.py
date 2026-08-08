"""
phase2_segmentation/tests — Test suite for Package 2: Volumetric Analysis & Segmentation

Test modules:
    test_volume_builder      — NIfTI assembly validation and SimpleITK integration
    test_segment_runner      — TotalSegmentator subprocess wrapper and statistics parsing  
    test_lifecycle_manager   — RAM monitoring, orchestration, and multiprocessing entry
    test_integration         — End-to-end pipeline tests (marked as @pytest.mark.integration)
    test_run_commands        — Manual testing utilities for real dependency verification
    conftest                 — Shared fixtures and pytest configuration

Running tests:
    pytest phase2_segmentation/tests/                    # All tests
    pytest phase2_segmentation/tests/ -m "not slow"      # Skip TotalSegmentator tests  
    pytest phase2_segmentation/tests/ -m integration     # Integration tests only
    pytest phase2_segmentation/tests/test_run_commands.py -v -s  # Manual verification
"""
