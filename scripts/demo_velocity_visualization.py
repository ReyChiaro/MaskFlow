"""Two synthetic latent trajectories for layout QA; NOT a Qwen model result."""
import argparse

import torch

from pipelines.maskflow_utils import poisson_refine
from pipelines.velocity_visualization import VelocityTrace
from schedulers.mask_flow import MaskFlowScheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/velocity-demo')
    args = parser.parse_args()
    h, w = 80, 96
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1.2, 1.2, w), indexing='ij')
    radius = torch.sqrt((x / .8) ** 2 + (y / .75) ** 2)
    mask = ((1.05 - radius) / .25).clamp(0, 1)[None, None]
    source = torch.stack([.75 + .12*x, .78 + .1*y, .82 - .1*x])[None].clamp(0, 1)
    source_latents = torch.stack([.4*x, .4*y, .2*x*y, .1*torch.cos(x*3)])[None]
    driving = torch.stack([-x + .15*torch.sin(3*y), -.8*y, .15*x*y, .1*torch.cos(x*3)])[None]
    noise = source_latents + driving
    states = {'off': noise.clone(), 'on': noise.clone()}
    scheduler = MaskFlowScheduler(use_dynamic_shifting=False)
    trace = VelocityTrace(args.output, source, mask, 12, grid=22,
                          names=['Synthetic trajectories - NOT a model result'],
                          metadata={'synthetic': True, 'description': __doc__})
    with scheduler.inference(12) as inferencer:
        for step, (t, sigma, next_sigma, rate) in enumerate(inferencer, start=1):
            predictions, next_states = {}, {}
            for name, state in states.items():
                velocity = .35 * state + driving * (.5 + sigma)
                if name == 'on':
                    x0 = scheduler.predict_x0(state, sigma, rate, velocity, source_latents, mask, noise)
                    refined = poisson_refine(x0, source_latents, mask > 0, mask, 1., 1., 50, .1, disable_progress_bar=True)
                    velocity = (state - refined) / sigma
                predictions[name] = velocity
                next_states[name] = scheduler.step(state, velocity, sigma, next_sigma, rate, source_latents, mask, noise)
            trace.record_step(step, sigma, next_sigma, True, predictions['off'], predictions['on'],
                              next_states['off'], next_states['on'])
            trace.capture(step, t, sigma, True, predictions['off'], predictions['on'],
                          state_off=states['off'], state_on=states['on'])
            states = next_states
    for path in trace.save():
        print(path)


if __name__ == '__main__':
    main()
