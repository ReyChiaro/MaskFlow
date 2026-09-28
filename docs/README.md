# MaskFlow project page

This directory contains the static project page intended for GitHub Pages.

## Content configuration

Edit `site-config.js` to add or update:

- paper, code, model, demo, and dataset URLs;
- author names;
- ordered experiment figures and captions;
- the eight carousel images in `galleryFigures`;
- the final BibTeX citation.

When a resource URL is empty, the page renders it as a disabled "Coming soon" item.

## Figures

Optimized web figures live in `assets/figures/`. The current figures were exported from `ICLR27_Preprint/figs/` on 2026-09-28.
Gallery images are copied from `ICLR27_Preprint/supp_figs/qualitative_comparisons/`.
Source PDFs are not modified. See `CONTENT_SOURCES.md` for the source mapping.

All displayed figures use their intrinsic aspect ratio. When replacing a figure,
regenerate the web image from its source PDF instead of resizing it to fixed width
and height values independently.

## GitHub Pages

Configure the repository to deploy from the `main` branch and `/docs` folder.
All asset paths are relative so the page works under the `/MaskFlow/` project path.
