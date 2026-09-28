"""Spatial diagnostics of latent velocities, not optical flow or pixel displacement.

The pipeline independently evolves two states; this recorder runs diagnostics on CPU.
A common, zero-preserving channel projection is fitted over BOTH sides and ALL
captured steps of one sample. Full-channel norms remain available independently.
"""

import html
import json
import os
import re
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def project_fields(velocity_off, velocity_on):
    """[T,C,H,W] -> [T,H,W,2], using uncentered principal directions.

    Centering would turn a zero velocity into a nonzero arrow. Fit in channel
    space using the second moment instead; pin signs for reproducible exports.
    """
    channels = velocity_off.shape[1]
    matrix = np.concatenate([np.moveaxis(v, 1, -1).reshape(-1, channels) for v in (velocity_off, velocity_on)])
    gram = matrix.astype(np.float64).T @ matrix
    values, vectors = np.linalg.eigh(gram)
    basis = vectors[:, np.argsort(values)[::-1][:2]]
    for column in range(basis.shape[1]):
        pivot = np.argmax(np.abs(basis[:, column]))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    if channels == 1:
        basis = np.pad(basis, ((0, 0), (0, 1)))
    total = max(float(np.trace(gram)), 0.0)
    retained = float(np.maximum(values, 0)[-2:].sum() / total) if total > 0 else 0.0
    fields = [np.einsum("tchw,ck->thwk", v, basis).astype(np.float32) for v in (velocity_off, velocity_on)]
    return *fields, basis.astype(np.float32), retained


def channel_rms(field):
    return np.sqrt(np.mean(np.square(field.astype(np.float64)), axis=1)).astype(np.float32)


def display_limit(values):
    """One robust scale per sample across all frames, with a zero-safe floor."""
    return max(float(np.percentile(values, 99)), 1e-8)


class VelocityTrace:
    def __init__(self, directory, source, mask, total_steps, *, steps=None,
                 max_samples=1, grid=24, names=None, metadata=None):
        if total_steps < 1 or max_samples < 1 or grid < 2:
            raise ValueError("total_steps/max_samples must be positive; grid must be >= 2.")
        if steps is not None and any(
            isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= total_steps for s in steps
        ):
            raise ValueError("Visualization steps must be 1-based integers within the inference run.")
        self.steps = set(steps if steps is not None else np.linspace(1, total_steps, min(6, total_steps)).round().astype(int))
        self.directory = Path(directory)
        self.sample_count = min(max_samples, len(source))
        # Bound the backdrop resolution; velocity arrays retain the full latent grid.
        backdrop = source[:self.sample_count].detach().float()
        scale = min(1.0, 768 / max(backdrop.shape[-2:]))
        size = [max(1, round(s * scale)) for s in backdrop.shape[-2:]]
        self.source = F.interpolate(backdrop, size=size, mode="bilinear", align_corners=False).cpu().numpy()
        self.mask = mask[:self.sample_count].detach().float().mean(1, keepdim=True).cpu()
        self.names = [str(names[i]) if names is not None else f"sample-{i:03d}" for i in range(self.sample_count)]
        self.metadata = metadata or {}
        self.grid = grid
        self.frames = []
        self.history = []
        self.run_id = f"rank-{os.environ.get('RANK', '0')}-{uuid.uuid4().hex[:10]}"

    def wants(self, step):
        return step in self.steps

    def record_step(self, step, sigma, next_sigma, applied, velocity_off, velocity_on, next_off, next_on):
        """Small per-sample summaries at EVERY step, including unsaved field frames.

        Inputs may be packed: global RMS is invariant to the latent permutation.
        State divergence is measured AFTER integration and background projection.
        """
        def rms_difference(left, right):
            delta = right[:self.sample_count].float() - left[:self.sample_count].float()
            return delta.square().flatten(1).mean(1).sqrt().detach().cpu().tolist()

        self.history.append({
            "step": int(step), "sigma": float(sigma), "next_sigma": float(next_sigma),
            "poisson_applied_on": bool(applied),
            "velocity_delta_rms": rms_difference(velocity_off, velocity_on),
            "next_state_delta_rms": rms_difference(next_off, next_on),
        })

    def capture(self, step, timestep, sigma, applied, velocity_off, velocity_on, *, state_off, state_on):
        if not self.wants(step):
            return
        arrays = [v[:self.sample_count].detach().float().cpu().numpy().copy()
                  for v in (velocity_off, velocity_on, state_off, state_on)]
        if any(v.shape != arrays[0].shape or v.ndim != 4 or v.shape[0] != self.sample_count for v in arrays):
            raise ValueError("Expected matching [B,C,H,W] unpacked velocity and state tensors.")
        if not all(np.isfinite(v).all() for v in arrays):
            raise ValueError("Cannot visualize non-finite velocities or states.")
        self.frames.append((int(step), float(timestep), float(sigma), bool(applied), *arrays))

    def save(self, outputs=None):
        if not self.frames:
            return []
        paths = []
        for sample in range(self.sample_count):
            name = re.sub(r"[^\w.-]+", "_", self.names[sample]).strip(".")[:100] or "sample"
            folder = self.directory / self.run_id / f"{sample:03d}-{name}"
            folder.mkdir(parents=True, exist_ok=False)
            velocity_off = np.stack([f[4][sample] for f in self.frames])
            velocity_on = np.stack([f[5][sample] for f in self.frames])
            state_off = np.stack([f[6][sample] for f in self.frames])
            state_on = np.stack([f[7][sample] for f in self.frames])
            state_delta_rms = channel_rms(state_on - state_off)
            projected_off, projected_on, basis, retained = project_fields(velocity_off, velocity_on)
            rms_off, rms_on, rms_delta = map(channel_rms, (velocity_off, velocity_on, velocity_on - velocity_off))
            mask = F.interpolate(self.mask[sample:sample + 1], size=velocity_off.shape[-2:], mode="nearest")[0, 0].numpy()
            magnitude_limit = display_limit(np.stack([rms_off, rms_on]))
            delta_limit = display_limit(rms_delta)
            arrow_limit = display_limit(np.linalg.norm(np.stack([projected_off, projected_on]), axis=-1))
            records = []
            for index, frame in enumerate(self.frames):
                regions = {}
                for label, region in {"edit_core": mask >= 0.95, "soft_boundary": (mask > 0) & (mask < 0.95),
                                      "background": mask == 0}.items():
                    # Mean of per-location channel RMS, with explicit empty regions.
                    regions[label] = {
                        "count": int(region.sum()),
                        "off_rms": float(rms_off[index][region].mean()) if region.any() else None,
                        "on_rms": float(rms_on[index][region].mean()) if region.any() else None,
                        "delta_rms": float(rms_delta[index][region].mean()) if region.any() else None,
                        "state_delta_rms": float(state_delta_rms[index][region].mean()) if region.any() else None,
                    }
                records.append({"step": frame[0], "timestep": frame[1], "sigma": frame[2],
                                "poisson_applied_on": frame[3], "regions": regions})
                self._render(folder / f"step-{frame[0]:04d}.png", self.source[sample], mask,
                             projected_off[index], projected_on[index], rms_off[index], rms_on[index],
                             rms_delta[index], magnitude_limit, delta_limit, arrow_limit, records[-1], retained)
            metadata = {
                "sample": self.names[sample], "settings": dict(self.metadata),
                "schema_version": 2,
                "comparison": "independent_poisson_off_on_trajectories",
                "meaning": "v_off(x_off, sigma) versus v_on(x_on, sigma), each used to evolve its own state.",
                "state_timing": "Saved states precede the current update; history next_state_delta_rms follows background projection.",
                "projection": "Shared uncentered channel SVD; arrows are not spatial motion; no per-frame refitting.",
                "retained_energy": retained,
                "scales": {"velocity_rms_p99": magnitude_limit, "delta_rms_p99": delta_limit,
                           "projected_length_p99": arrow_limit},
                "frames": records,
                "history": [{**row, "velocity_delta_rms": row["velocity_delta_rms"][sample],
                             "next_state_delta_rms": row["next_state_delta_rms"][sample]}
                            for row in self.history],
            }
            (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            np.savez_compressed(folder / "fields.npz", velocity_off=velocity_off, velocity_on=velocity_on,
                                projected_off=projected_off, projected_on=projected_on,
                                rms_off=rms_off, rms_on=rms_on, rms_delta=rms_delta,
                                state_off=state_off, state_on=state_on, state_delta_rms=state_delta_rms,
                                basis=basis, mask=mask, steps=[f[0] for f in self.frames],
                                sigmas=[f[2] for f in self.frames])
            if self.history:
                self._render_history(folder / "trajectory.png", metadata["history"])
            if outputs is not None:
                from PIL import Image

                for name in ("off", "on"):
                    rgb = outputs[name][sample].detach().float().cpu().numpy()
                    if rgb.shape[0] == 1:
                        rgb = np.repeat(rgb, 3, axis=0)
                    pixels = np.moveaxis(rgb, 0, -1).clip(0, 1)
                    Image.fromarray((pixels * 255).round().astype(np.uint8)).save(folder / f"final-poisson-{name}.png")
            self._write_viewer(folder, records, self.names[sample], bool(self.history), outputs is not None)
            paths.append((folder / "index.html").resolve())
        self.frames.clear()
        self.history.clear()
        return paths

    def _render(self, path, source, mask, velocity_off, velocity_on, mag_off, mag_on, delta,
                magnitude_limit, delta_limit, arrow_limit, record, retained):
        # Figure/Agg avoids changing the application's global matplotlib backend.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import Normalize
        from matplotlib.figure import Figure
        from matplotlib.cm import ScalarMappable

        height, width = mask.shape
        stride = max(1, int(np.ceil(max(height, width) / self.grid)))
        yy, xx = np.mgrid[stride // 2:height:stride, stride // 2:width:stride]
        image = np.moveaxis(source, 0, -1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        image = 0.22 * image.clip(0, 1) + 0.78
        figure = Figure(figsize=(14, 6.2), dpi=150, facecolor="#fafaf9")
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 3)
        figure.subplots_adjust(left=0.035, right=0.985, bottom=0.23, top=0.81, wspace=0.10)
        status = "ON: Poisson applied" if record["poisson_applied_on"] else "ON: outside Poisson window"
        figure.text(0.035, 0.94, "Velocity field / trajectory comparison", fontsize=19, color="#1e293b", weight="bold")
        figure.text(0.035, 0.89, f"Step {record['step']:02d}  /  sigma {record['sigma']:.4f}  /  {status}  /  independently evolved states",
                    fontsize=10, color="#64748b")
        norm = Normalize(0, magnitude_limit, clip=True)
        for ax, title in zip(axes, ("Poisson OFF", "Poisson ON", "Velocity difference / ON - OFF")):
            ax.imshow(image, extent=(-0.5, width - 0.5, height - 0.5, -0.5))
            ax.set(xlim=(-0.5, width - 0.5), ylim=(height - 0.5, -0.5))
            ax.set_title(title, loc="left", fontsize=12, color="#334155", pad=12)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#e2e8f0")
        for ax, field, magnitude in zip(axes[:2], (velocity_off, velocity_on), (mag_off, mag_on)):
            # Fixed scale and linear lengths, with the top 1% clipped to avoid clutter.
            lengths = np.linalg.norm(field, axis=-1)
            factor = np.minimum(1.0, arrow_limit / np.maximum(lengths, 1e-12))
            shown = field * factor[..., None] * (0.85 * stride / arrow_limit)
            u, v = shown[yy, xx].transpose(2, 0, 1)
            visible = lengths[yy, xx] > 1e-12
            ax.quiver(xx[visible], yy[visible], u[visible], v[visible], magnitude[yy, xx][visible],
                      cmap="plasma", norm=norm, angles="xy", scale_units="xy", scale=1,
                      pivot="mid", width=0.0028, headwidth=3.2, headlength=4.2,
                      minlength=0, minshaft=1)
        axes[2].imshow(delta, cmap="magma", vmin=0, vmax=delta_limit,
                       interpolation="nearest", extent=(-0.5, width - 0.5, height - 0.5, -0.5))
        for ax in axes:
            # Display the operational support plus the soft mask's 0.5 contour.
            if mask.min() < 0.5 < mask.max():
                ax.contour(mask, levels=[0.5], colors=["#ffffff"], linewidths=1.6)
                ax.contour(mask, levels=[0.5], colors=["#334155"], linewidths=0.65)
            if (mask > 0).any() and (mask == 0).any():
                ax.contour((mask > 0).astype(float), levels=[0.5], colors=["#94a3b8"],
                           linewidths=0.7, linestyles="dashed")
        for rect, cmap, limit, label in (
            ([0.07, 0.14, 0.54, 0.018], "plasma", magnitude_limit, "Full-channel velocity RMS"),
            ([0.72, 0.14, 0.23, 0.018], "magma", delta_limit, "Full-channel velocity difference RMS"),
        ):
            bar = figure.colorbar(ScalarMappable(norm=Normalize(0, limit), cmap=cmap),
                                 cax=figure.add_axes(rect), orientation="horizontal")
            bar.set_label(label, fontsize=9, color="#475569")
            bar.ax.tick_params(labelsize=8, colors="#64748b")
            bar.outline.set_visible(False)
        figure.text(0.035, 0.025,
                    f"Shared channel projection ({retained:.1%} energy) / arrows are not optical flow / scales fixed across steps; clipped at p99",
                    fontsize=8, color="#64748b")
        figure.savefig(path)
        figure.clear()

    @staticmethod
    def _render_history(path, history):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(14, 2.9), dpi=150, facecolor="#fafaf9")
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 2)
        figure.subplots_adjust(left=.065, right=.975, top=.82, bottom=.24, wspace=.22)
        for ax, key, title, color in zip(
            axes, ("velocity_delta_rms", "next_state_delta_rms"),
            ("Velocity difference / before integration", "Accumulated state difference / after background projection"),
            ("#7c3aed", "#0891b2"),
        ):
            ax.plot([r["step"] for r in history], [r[key] for r in history], color=color, linewidth=2)
            ax.set_title(title, loc="left", fontsize=11, color="#334155")
            ax.set_xlabel("Inference step", fontsize=9)
            ax.set_ylabel("Global RMS", fontsize=9)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
        figure.savefig(path)
        figure.clear()

    @staticmethod
    def _write_viewer(folder, records, name, has_history, has_outputs):
        frames = [{"file": f"step-{r['step']:04d}.png", "label": f"Step {r['step']} / sigma {r['sigma']:.4f}"} for r in records]
        # Frame URLs are generated numeric filenames, never user content.
        page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Velocity / Poisson</title>
<style>body{margin:0;background:#fafaf9;color:#334155;font:15px system-ui,sans-serif}
main{max-width:1500px;margin:28px auto;padding:0 24px}header{display:flex;align-items:center;gap:20px}
h1{font-size:18px;margin-right:auto}button{background:#334155;color:white;border:0;border-radius:8px;padding:9px 16px;cursor:pointer}
input{flex:1;accent-color:#7c3aed;min-width:80px}img{width:100%;display:block}p{font-size:13px;color:#64748b;line-height:1.65}
a{color:#6d28d9}label{font-variant-numeric:tabular-nums;min-width:185px}@media(max-width:650px){header{flex-wrap:wrap}}</style>
<main><h1>__NAME__</h1><header><button id="play">Play</button><input aria-label="Inference step" id="step" type="range" min="0" max="__MAX__" value="0"><label id="label"></label></header>
<img id="frame" alt="Poisson OFF and ON independently evolved latent velocities and their difference">
<p>Arrows show a fixed two-dimensional projection of latent channel velocities, not image motion.
Colors show full-channel RMS. The same projection, arrow scale and color limits are used at every captured step.
Solid contour: soft mask = 0.5. Dashed contour: editing support (mask &gt; 0).<br>
Two complete trajectories start from the same noise and conditions, and independently evolve with Poisson OFF or ON.
The difference combines accumulated state divergence and the current Poisson correction.
A step outside the Poisson window can still differ because earlier updates have already separated the states.
Velocity is measured before scheduler background projection; the state curve below is measured after it.</p>
__EXTRAS__
<p><a href="metadata.json">Settings &amp; regional statistics</a> · <a href="fields.npz">Full latent fields &amp; projection</a></p></main>
<script>const frames=__FRAMES__, slider=document.getElementById('step'), frame=document.getElementById('frame'),
label=document.getElementById('label'), play=document.getElementById('play');let timer=null;
function show(){const f=frames[+slider.value];frame.src=f.file;label.textContent=f.label}
slider.oninput=show;play.onclick=()=>{if(timer){clearInterval(timer);timer=null;play.textContent='Play'}else{
play.textContent='Pause';timer=setInterval(()=>{slider.value=(+slider.value+1)%frames.length;show()},850)}};show();</script></html>"""
        extras = ""
        if has_history:
            extras += '<img src="trajectory.png" alt="Velocity difference and accumulated state divergence over all steps">'
        if has_outputs:
            extras += ('<h1>Final generation</h1><div style="display:flex;gap:16px">'
                       '<div style="width:50%"><p>Poisson OFF</p><img src="final-poisson-off.png" alt="Final Poisson OFF output"></div>'
                       '<div style="width:50%"><p>Poisson ON</p><img src="final-poisson-on.png" alt="Final Poisson ON output"></div></div>')
        page = page.replace("__EXTRAS__", extras)
        page = page.replace("__MAX__", str(len(frames) - 1)).replace("__FRAMES__", json.dumps(frames))
        page = page.replace("__NAME__", html.escape(name))
        (folder / "index.html").write_text(page, encoding="utf-8")
