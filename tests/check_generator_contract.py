"""Native candidate filtering must preserve tactic/score pairs and templates."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class GeneratorTests(unittest.TestCase):
    def test_filter_alignment_and_qwen_template(self):
        source = Path(__file__).resolve().parents[2] / 'REALPROVER/manager/thirdparty/generator.py'
        tree = ast.parse(source.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'get_lean_tactics')
        seen = []

        def prompt(state, premises, template):
            seen.append(template)
            return 'prompt'

        namespace = {'SamplingParams': lambda **kwargs: kwargs,
            'ModelManage': SimpleNamespace(contain_gemini=lambda x: False, contain_local=lambda x: True),
            'LeanSearch': SimpleNamespace(get_related_theorem=lambda x: []),
            'PromptManage': SimpleNamespace(build_local_prompt_str=prompt)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        for texts, expected, scores in [(['sorry', 'rfl'], ['rfl'], [-2.]),
                                       (['rfl', 'sorry'], ['rfl'], [-1.]),
                                       (['sorry', 'sorry'], [], [])]:
            candidates = [SimpleNamespace(text=text, token_ids=[1, 2], cumulative_logprob=-2. * (i + 1))
                          for i, text in enumerate(texts)]
            fake = SimpleNamespace(model_list=['local'], sampling_params={}, _init_model=lambda: None,
                llm=SimpleNamespace(generate=lambda *args, **kwargs: [SimpleNamespace(outputs=candidates)]))
            actual, values, unused = namespace['get_lean_tactics'](fake, 'goal', 2, template='qwen')
            self.assertEqual(actual, expected)
            self.assertEqual(values, scores)
        self.assertEqual(seen, ['qwen'] * 3)


if __name__ == '__main__':
    unittest.main()
