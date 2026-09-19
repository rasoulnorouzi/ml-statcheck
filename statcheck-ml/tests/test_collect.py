"""Tests for collect module."""
import json
import tempfile
from pathlib import Path
import pytest
import importlib.util

# Import collect from pipeline/03_collect.py using importlib (filename starts with digit)
spec = importlib.util.spec_from_file_location(
    "collect_module",
    str(Path(__file__).parent.parent / "pipeline" / "03_collect.py")
)
collect_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect_module)
collect = collect_module.collect


@pytest.fixture
def temp_collect_env():
    """Create a temporary environment for collect testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Get real test data paths
        sample_dir = Path(__file__).parent.parent / "data" / "sample_round2" / "chunks"
        chunk_file = sample_dir / "chunk_00.json"

        if not chunk_file.exists():
            pytest.skip("Test data not found")

        # Create batch and output directories
        batches_dir = tmpdir / "batches"
        outputs_dir = tmpdir / "outputs"
        batches_dir.mkdir()
        outputs_dir.mkdir()

        # Load chunk file and use first two windows
        with open(chunk_file, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)

        batch_data = all_chunks[:2]

        # Write batch file
        with open(batches_dir / "batch_000.json", 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=1)
            f.write('\n')

        # Create valid v2 output: first window has no result, second has one result
        output_data = [
            {
                "window_id": batch_data[0]["window_id"],
                "contains_result": False,
                "results": []
            },
            {
                "window_id": batch_data[1]["window_id"],
                "contains_result": True,
                "results": [{
                    "test_type": "F",
                    "statistic": "6.14",
                    "df1": "2",
                    "df2": "89",
                    "n": None,
                    "p_operator": "<",
                    "p_value": ".01",
                    "quote": "F(2, 89) = 6.14, p < .01",
                    "damaged": False,
                    "confidence": "high"
                }]
            }
        ]

        with open(outputs_dir / "batch_000.json", 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=1)
            f.write('\n')

        # Get guideline and agent paths
        guideline_path = Path(__file__).parent.parent / "docs" / "GUIDELINE.md"
        agent_path = Path(__file__).parent.parent.parent / ".claude" / "agents" / "annotator-haiku.md"

        if not guideline_path.exists():
            pytest.skip("GUIDELINE.md not found")
        if not agent_path.exists():
            pytest.skip("annotator-haiku.md not found")

        yield {
            'tmpdir': tmpdir,
            'batches_dir': batches_dir,
            'outputs_dir': outputs_dir,
            'guideline_path': guideline_path,
            'agent_path': agent_path,
        }


def test_collect_success(temp_collect_env):
    """Test successful collection and validation."""
    env = temp_collect_env
    out_file = env['tmpdir'] / "collected.json"

    records, failures = collect(
        env['batches_dir'],
        env['outputs_dir'],
        'haiku',
        str(env['agent_path']),
        str(env['guideline_path']),
    )

    # Should have no failures
    assert failures == []

    # Should have records
    assert len(records) > 0

    # Each record should have provenance
    for record in records:
        assert 'provenance' in record
        assert record['provenance']['rater'] == 'haiku'
        assert record['provenance']['model_alias'] == 'haiku'
        assert record['provenance']['model_id'] == 'claude-haiku-4-5-20251001'
        assert 'guideline_sha' in record['provenance']
        assert 'agent_sha' in record['provenance']
        assert record['provenance']['agent_kind'] == 'agent'
        assert record['provenance']['batch_id'] == 'batch_000'
        assert 'collected_at' in record['provenance']
        assert record['provenance']['retries'] == 0

    # Each result should have span and aligned flag
    for record in records:
        for result in record['results']:
            assert 'span' in result
            assert 'aligned' in result
            assert isinstance(result['aligned'], bool)
            if result['aligned']:
                assert result['span'] is not None
            else:
                assert result['span'] is None


def test_collect_missing_output(temp_collect_env):
    """Test collection fails when output file is missing."""
    env = temp_collect_env

    # Delete the output file
    output_file = env['outputs_dir'] / "batch_000.json"
    output_file.unlink()

    records, failures = collect(
        env['batches_dir'],
        env['outputs_dir'],
        'haiku',
        str(env['agent_path']),
        str(env['guideline_path']),
    )

    # Should have failures
    assert len(failures) == 1
    assert failures[0]['batch_id'] == 'batch_000'
    assert 'not found' in failures[0]['errors'][0]

    # Should have no records
    assert records == []
