"""CPU checks for MaskFlow condition dropout and Qwen/FLUX CFG, without model downloads."""
import dataclasses
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

from pipelines.cfg import BRANCHES, cfg_coefficients, combine_predictions, required_branches, training_probabilities
from pipelines.qwenimage.qwenimage_maskflow import QwenImageMaskFlow, QwenMaskFlowPreprocessOutput
from pipelines.flux2.flux2_maskflow import Flux2MaskFlow, Flux2MaskFlowPreprocessOutput
from schedulers.mask_flow import MaskFlowScheduler
from trainer.lora import MaskFlowTrainer
import inference


def qwen_fixture():
    pipe = object.__new__(QwenImageMaskFlow)
    pipe.device, pipe.dtype = torch.device('cpu'), torch.float32
    pipe.generator = torch.Generator().manual_seed(10)
    pipe.vae = SimpleNamespace(config=SimpleNamespace(temperal_downsample=[], z_dim=1))
    pipe.scheduler = MaskFlowScheduler(use_dynamic_shifting=False)
    pipe.enable_poisson_train = pipe.enable_poisson_infer = False
    pipe.enable_pixel_blend = False
    pipe.poisson_num_iter = 2
    pipe.prompt_calls = []

    def encode_prompt(prompt, conditions):
        pipe.prompt_calls.append((list(prompt), tuple(conditions)))
        return torch.ones(1, 2, 2) * bool(prompt[0]), torch.ones(1, 2)

    pipe.encode_prompt = encode_prompt
    pipe.encode_image = lambda image, mode: image.clone()
    pipe.decode_image = lambda image: image.squeeze(2)
    source = torch.ones(1, 1, 1, 4, 4) * 2
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., :2] = 1
    data = QwenMaskFlowPreprocessOutput(
        prompt=['edit'], negative_prompt=['avoid'], target=source + 1, height=4, width=4,
        raw_source=source.squeeze(2), mask=mask,
        dit_conditions={'source': source, 'mask': mask.unsqueeze(2)},
        vlm_conditions={'source': source.squeeze(2), 'mask': mask},
    )
    return pipe, data


def flux_fixture():
    pipe = object.__new__(Flux2MaskFlow)
    pipe.device, pipe.dtype = torch.device('cpu'), torch.float32
    pipe.generator = torch.Generator().manual_seed(10)
    pipe.vae = SimpleNamespace(config=SimpleNamespace(latent_channels=4))
    pipe.scheduler = MaskFlowScheduler(use_dynamic_shifting=False)
    pipe.enable_poisson_train = pipe.enable_poisson_infer = False
    pipe.enable_pixel_blend = False
    pipe.rescale_cfg = True
    pipe.poisson_num_iter = 2
    pipe.prompt_calls = []
    pipe.encode_image = lambda image, *args: image.clone()
    pipe.encode_mask = lambda mask: mask.clone()
    pipe.decode_image = lambda image: image

    def encode_prompt(prompt):
        pipe.prompt_calls.append(list(prompt))
        return torch.ones(1, 2, 2) * bool(prompt[0]), torch.zeros(1, 2, 4)

    def prepare_latents(**kwargs):
        noise = torch.randn(1, 4, 4, generator=kwargs['generator'])
        ids = Flux2Pipeline._prepare_image_ids([torch.zeros(1, 4, 2, 2)])
        ids[..., 0] = 0
        return noise, ids

    pipe.encode_prompt = encode_prompt
    pipe.text_pipeline = SimpleNamespace(prepare_latents=prepare_latents)
    source = torch.ones(1, 4, 2, 2) * 2
    mask = torch.zeros_like(source)
    mask[..., 0] = 1
    data = Flux2MaskFlowPreprocessOutput(
        prompt=['edit'], negative_prompt=['avoid'], target=source + 1, height=2, width=2,
        raw_source=source, mask=mask, dit_conditions={'source': source, 'mask': mask},
    )
    return pipe, data


class CFGAlgebraTests(unittest.TestCase):
    def test_legacy_text_guidance_and_rescaling(self):
        torch.manual_seed(12)
        pred = {key: torch.randn(2, 3, 4) for key in BRANCHES}
        expected = pred['nm'] + 4 * (pred['pm'] - pred['nm'])
        torch.testing.assert_close(combine_predictions(pred, 4, rescale=False), expected)
        expected *= pred['pm'].norm(dim=-1, keepdim=True) / expected.norm(dim=-1, keepdim=True)
        torch.testing.assert_close(combine_predictions(pred, 4), expected)

    def test_corners_joint_guidance_and_interaction(self):
        pred = {key: torch.tensor([float(i + 1)]) for i, key in enumerate(BRANCHES)}
        for text, mask, interaction, branch in [(1, 1, None, 'pm'), (0, 1, None, 'nm'),
                                               (0, 0, None, 'nn'), (1, 0, 0, 'pn')]:
            torch.testing.assert_close(combine_predictions(pred, text, mask, interaction, False), pred[branch])
        joint = pred['nn'] + 3 * (pred['pm'] - pred['nn'])
        torch.testing.assert_close(combine_predictions(pred, 3, 3, None, False), joint)
        expected = pred['nn'] + 2*(pred['pn']-pred['nn']) + 3*(pred['nm']-pred['nn'])
        expected += 4*(pred['pm']-pred['pn']-pred['nm']+pred['nn'])
        torch.testing.assert_close(combine_predictions(pred, 2, 3, 4, False), expected)
        self.assertAlmostEqual(sum(cfg_coefficients(2, 3, 4).values()), 1)

    def test_only_required_branches_and_validation(self):
        self.assertEqual(required_branches(4), ['pm', 'nm'])
        self.assertEqual(required_branches(2, 3, 5), list(BRANCHES))
        self.assertEqual(required_branches(0, 0, rescale=False), ['nn'])
        with self.assertRaisesRegex(ValueError, 'Missing'):
            combine_predictions({'pm': torch.ones(1)}, 4)
        with self.assertRaises(ValueError):
            cfg_coefficients(float('nan'))
        self.assertTrue(torch.isfinite(combine_predictions({'pm': torch.zeros(1, 2)})).all())


class TrainerTests(unittest.TestCase):
    def test_probability_validation_and_legacy_fallback(self):
        self.assertEqual(training_probabilities(None, .1, 0), dict(pm=.9, pn=0., nm=.1, nn=0.))
        for value in [dict(pm=1), dict(pm=.9, pn=.1, nm=.1, nn=.1),
                      dict(pm=1, pn=-1, nm=1, nn=0), dict(pm=float('nan'), pn=0, nm=0, nn=0)]:
            with self.assertRaises(ValueError):
                training_probabilities(value)

    def test_trainer_selects_branch_without_deleting_data(self):
        probabilities = {name: .25 for name in BRANCHES}
        with patch('trainer.lora.PromptSampler') as sampler:
            trainer = MaskFlowTrainer(prompt_sampler_cfgs={}, cfg_branch_probabilities=probabilities)
        trainer.prompt_sampler = sampler.return_value
        trainer.prompt_sampler.sample_batch.return_value = ['sampled', 'sampled']
        original_mask = torch.ones(2, 1, 2, 2)
        for index, branch in enumerate(BRANCHES):
            trainer.rng = Mock(random=Mock(return_value=(index + .5) / 4))
            batch = {'prompt': ['p', 'p'], 'edit_instruction': ['e', 'e'],
                     'conditions': {'mask': original_mask}}
            result = trainer.preprocess_train_batch(batch, 1)
            self.assertEqual(result['cfg_branch'], branch)
            self.assertEqual(result['prompt'], ['sampled', 'sampled'])
            self.assertIs(result['conditions']['mask'], original_mask)


class PipelineTests(unittest.TestCase):
    def test_dropout_preserves_path_target_and_source_for_both_models(self):
        for factory in (qwen_fixture, flux_fixture):
            for poisson in (False, True):
                pipe, data = factory()
                pipe.enable_poisson_train = poisson
                baseline = None
                for branch in BRANCHES:
                    with self.subTest(model=type(pipe).__name__, branch=branch, poisson=poisson):
                        pipe.generator.manual_seed(10)
                        pipe.prompt_calls.clear()
                        inputs = pipe.prepare_forward_inputs(dataclasses.replace(data, cfg_branch=branch))
                        self.assertEqual(len(pipe.prompt_calls), 1)
                        self.assertEqual(inputs.cfg_branch, branch)
                        self.assertEqual(len(inputs.conditions), 2 if branch.endswith('m') else 1)
                        for field in ('noise', 'noised_target', 'ground_truth', 'source_latents', 'mask_latents'):
                            if baseline is not None:
                                torch.testing.assert_close(getattr(inputs, field), getattr(baseline, field))
                        baseline = inputs
                        self.assertTrue(bool(inputs.prompt_embeds.any()) == branch.startswith('p'))
                        if factory is qwen_fixture:
                            prompt, keys = pipe.prompt_calls[-1]
                            self.assertEqual(prompt, ['edit'] if branch.startswith('p') else [''])
                            self.assertEqual('mask' in keys, branch.endswith('m'))
                            self.assertEqual(len(inputs.image_shapes[0]), len(inputs.conditions) + 1)
                        else:
                            self.assertEqual(inputs.image_ids.shape[1], inputs.noise.shape[1] + sum(c.shape[1] for c in inputs.conditions))
                            if branch.startswith('n'):
                                self.assertEqual(pipe.prompt_calls[-1], [''])

    def test_missing_mask_branches_supervise_background(self):
        for factory in (qwen_fixture, flux_fixture):
            pipe, data = factory()
            for branch in BRANCHES:
                with self.subTest(model=type(pipe).__name__, branch=branch):
                    selected = dataclasses.replace(data, cfg_branch=branch)
                    pipe.preprocess_inputs = lambda batch: selected
                    predictions = []
                    if factory is qwen_fixture:
                        def denoise(**kwargs):
                            pred = torch.zeros_like(kwargs['hidden_states'][:, :kwargs['img_seq_len']], requires_grad=True)
                            predictions.append(pred)
                            return pred
                    else:
                        def denoise(xt, timestep, inputs):
                            pred = torch.zeros_like(xt, requires_grad=True)
                            predictions.append(pred)
                            return pred
                    pipe.denoise = denoise
                    pipe.generator.manual_seed(10)
                    inputs = pipe.prepare_forward_inputs(selected)
                    pipe.generator.manual_seed(10)
                    pipe.forward_step({})['loss'].backward()
                    outside = inputs.mask_latents == 0
                    background_gradient = predictions[0].grad[outside]
                    self.assertEqual(bool(background_gradient.abs().sum() > 0), branch.endswith('n'))

    def test_eval_encoding_has_no_mask_leak_and_correct_ids(self):
        for factory in (qwen_fixture, flux_fixture):
            pipe, data = factory()
            inputs = pipe.prepare_eval_inputs(data, 2, 3, 5)
            self.assertEqual(list(inputs.cfg_branches), list(BRANCHES))
            for name, branch in inputs.cfg_branches.items():
                self.assertEqual(len(branch.conditions), 2 if name.endswith('m') else 1)
            if factory is qwen_fixture:
                self.assertEqual(pipe.prompt_calls, [(['edit'], ('source', 'mask')), (['edit'], ('source',)),
                                                     (['avoid'], ('source', 'mask')), (['avoid'], ('source',))])
            else:
                self.assertEqual(pipe.prompt_calls, [['edit'], ['avoid']])
                pm, pn = inputs.cfg_branches['pm'], inputs.cfg_branches['pn']
                torch.testing.assert_close(pn.image_ids, pm.image_ids[:, :8])
                self.assertEqual(pn.image_ids.shape[1], 8)
                self.assertEqual(pm.image_ids.shape[1], 12)
            pipe.prompt_calls.clear()
            inputs = pipe.prepare_eval_inputs(data, 4)
            self.assertEqual(list(inputs.cfg_branches), ['pm', 'nm'])

    def test_four_branch_sampling_shares_state_and_preserves_background(self):
        for factory in (qwen_fixture, flux_fixture):
            pipe, data = factory()
            pipe.preprocess_inputs = lambda batch: data
            seen = []
            if factory is qwen_fixture:
                def denoise(branch, xt, timestep):
                    seen.append((xt.clone(), timestep.clone(), len(branch.conditions)))
                    return torch.ones_like(xt)
                pipe.denoise_cfg_branch = denoise
            else:
                def denoise(xt, timestep, branch):
                    seen.append((xt.clone(), timestep.clone(), len(branch.conditions)))
                    return torch.ones_like(xt)
                pipe.denoise = denoise
            output = pipe.eval_step({}, num_inference_steps=2, text_cfg_scale=2, mask_cfg_scale=3, interaction_cfg_scale=5)
            self.assertEqual(len(seen), 8)
            for offset in (0, 4):
                for current in seen[offset:offset + 4]:
                    torch.testing.assert_close(current[0], seen[offset][0])
                    torch.testing.assert_close(current[1], seen[offset][1])
            outside = data.mask == 0
            torch.testing.assert_close(output['output'][outside], data.raw_source[outside])

    def test_native_transformer_receives_aligned_conditions(self):
        for factory in (qwen_fixture, flux_fixture):
            pipe, data = factory()
            calls = []

            class Transformer(torch.nn.Module):
                config = SimpleNamespace(guidance_embeds=True, patch_size=2)

                def forward(self, **kwargs):
                    calls.append(kwargs)
                    return (torch.ones_like(kwargs['hidden_states']),)

            pipe.transformer = Transformer()
            inputs = pipe.prepare_eval_inputs(data, 2, 3, 5)
            if factory is qwen_fixture:
                predictions = {
                    name: pipe.denoise_cfg_branch(branch, inputs.noise, torch.tensor([.5]))
                    for name, branch in inputs.cfg_branches.items()
                }
                combine_predictions(predictions, 2, 3, 5, pipe.rescale_cfg)
            else:
                pipe.predict_velocity(inputs.noise, torch.tensor([.5]), inputs, 2, 3, 5)
            self.assertEqual(len(calls), 4)
            for call, name in zip(calls, BRANCHES):
                self.assertEqual(call['hidden_states'].shape[1], 12 if name.endswith('m') else 8)
                if factory is qwen_fixture:
                    self.assertEqual(len(call['img_shapes'][0]), 3 if name.endswith('m') else 2)
                else:
                    self.assertEqual(call['img_ids'].shape[1], call['hidden_states'].shape[1])
                    torch.testing.assert_close(call['guidance'], torch.tensor([pipe.guidance_scale]))

    def test_config_composition_for_both_models_and_inference(self):
        config_dir = str(Path(__file__).resolve().parents[1] / 'configs')
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            for name in ('sft_maskflow', 'sft_flux2_maskflow'):
                cfg = compose(config_name=name)
                self.assertEqual(set(cfg.trainer.cfg_branch_probabilities), set(BRANCHES))
                training_probabilities(cfg.trainer.cfg_branch_probabilities)
            for overrides in ([], ['pipeline=flux2_maskflow']):
                cfg = compose(config_name='inference', overrides=overrides)
                self.assertEqual(cfg.runtime.mask_cfg_scale, 1.)
                self.assertIn('maskflow', cfg.pipeline._target_)

    def test_inference_entry_passes_cfg_and_saves_result(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = str(Path(directory) / 'input.png')
            output_path = str(Path(directory) / 'output.png')
            Image.new('RGB', (4, 4), 'white').save(image_path)
            cfg = OmegaConf.create({
                'input': {'source': image_path, 'mask': image_path, 'prompt': 'edit'},
                'runtime': {'device': 'cpu', 'dtype': 'float32', 'num_inference_steps': 2,
                            'text_cfg_scale': 2., 'mask_cfg_scale': 3., 'interaction_cfg_scale': 5.},
                'output': {'path': output_path},
            })
            pipe = Mock()
            pipe.eval_step.return_value = {'output': torch.ones(1, 3, 4, 4)}
            with patch.object(inference, 'build_pipeline', return_value=pipe):
                inference.main.__wrapped__(cfg)
            args, kwargs = pipe.eval_step.call_args
            self.assertEqual(args[0]['prompt'], ['edit'])
            self.assertEqual(kwargs['mask_cfg_scale'], 3.)
            self.assertEqual(kwargs['interaction_cfg_scale'], 5.)
            self.assertTrue(Path(output_path).is_file())


if __name__ == '__main__':
    unittest.main()
