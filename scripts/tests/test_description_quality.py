import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_seo_build import description_quality_issues


class DescriptionQualityTests(unittest.TestCase):
    def test_rejects_real_openvla_cutoff(self):
        self.assertTrue(description_quality_issues("利用参数高效微调（Lo。"))

    def test_rejects_english_cutoff_and_stray_closing_parenthesis(self):
        self.assertTrue(description_quality_issues("并行推理 (Parallel Rea。"))
        self.assertTrue(description_quality_issues("说明LoRA）微调。"))

    def test_accepts_complete_nested_and_mixed_width_explanations(self):
        for text in ("OpenVLA支持参数高效微调（LoRA）。", "方法（说明（含示例））完整。", "说明（LLM)的定义。"):
            self.assertEqual([], description_quality_issues(text))

    def test_numeric_intervals_do_not_look_like_cutoff_explanations(self):
        self.assertEqual([], description_quality_issues("概率位于[0, 1)，温度处于(-1, 2.5]，范围为(0, +∞)。"))


if __name__ == "__main__":
    unittest.main()
