from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from recipe_sandbox.extensions import (
    load_extensions,
    materialize_extension_operator_catalog,
    run_extension_precomputations,
)
from recipe_sandbox.operators.registry import OperatorRegistry
from recipe_sandbox.schema.enums import Role
from recipe_sandbox.schema.types import CanonicalSample, Message, Target


EXTENSION_MODULE = "examples.extensions.dummy_extension"


def _sample(sample_id: str, text: str) -> CanonicalSample:
    return CanonicalSample(
        sample_id=sample_id,
        source_name="unit",
        messages=[Message(role=Role.USER, content=text)],
        target=Target(text=None),
    )


class ExtensionSmokeTest(unittest.TestCase):
    def test_dummy_extension_registers_precomputes_and_filters(self) -> None:
        registry = OperatorRegistry()
        hooks = load_extensions(EXTENSION_MODULE, registry=registry)
        self.assertIn("example_length_filter", registry.names())
        self.assertEqual(1, len(hooks))

        samples = [
            _sample("short", "short"),
            _sample("long", "this sample is intentionally longer"),
        ]
        results = run_extension_precomputations(
            EXTENSION_MODULE,
            samples=samples,
            context={"pool_size": len(samples)},
        )
        self.assertEqual(1, len(results))
        self.assertEqual(5, samples[0].metadata.extra["example_length"]["chars"])

        operator = registry.create("example_length_filter", max_chars=10)
        output = operator.apply(samples)
        self.assertEqual(["short"], [sample.sample_id for sample in output])

    def test_dummy_extension_catalog_patch_materializes_prompt_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "operator_catalog.extended.yaml"
            catalog_path = materialize_extension_operator_catalog(
                "examples/recipes/operator_catalog.yaml",
                str(output_path),
                EXTENSION_MODULE,
            )
            self.assertEqual(str(output_path), catalog_path)
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("example_length_filter", text)
            self.assertIn("max_chars", text)


if __name__ == "__main__":
    unittest.main()
