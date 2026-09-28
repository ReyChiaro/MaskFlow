/**
 * MaskFlow project-page configuration.
 * Add public URLs and the final BibTeX here; the UI updates automatically.
 */
window.MASKFLOW_SITE_CONFIG = {
  title: "MaskFlow",
  fullTitle: "MaskFlow: Precise, Consistent and Seamless Regional Image Editing",
  authors: [
    { name: "Rui Xu", affiliation: "1", href: "https://reychiaro.github.io/" },
    { name: "Yang Yong", affiliation: "1", href: "mailto:yongyang@sensetime.com" },
    { name: "Shunzi Yang", affiliation: "2", href: "mailto:michael2007@buaa.edu.cn" },
    { name: "Ruihao Gong", affiliation: "1,2", href: "https://xhplus.github.io", corresponding: true },
    { name: "Chengtao Lv", affiliation: "1,3", href: "mailto:chengtao001@e.ntu.edu.sg" },
  ],
  resources: [
    { id: "paper", label: "Paper", href: "https://arxiv.org/abs/2608.06929"},
    { id: "code", label: "Code", href: "https://github.com/ReyChiaro/MaskFlow" },
    { id: "model", label: "Model", href: "https://huggingface.co/ReyChiaro/MaskFlow" },
    { id: "dataset", label: "Dataset", href: "https://huggingface.co/datasets/ReyChiaro/MaskEdit-10k" },
    { id: "demo", label: "Demo", href: "https://github.com/ModelTC/LightX2V" },
  ],
  citation: `@misc{xu2026maskflowpreciseconsistentseamless,
  title={MaskFlow: Precise, Consistent and Seamless Regional Image Editing},
  author={Rui Xu and Yang Yong and Shunzi Yang and Ruihao Gong and Chengtao Lv},
  year={2026},
  eprint={2608.06929},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.06929},
}`,
  galleryFigures: [
  {
    "title": "Scene \u00b7 Addition",
    "image": "./assets/figures/gallery/scene-add.jpg",
    "width": 3576,
    "height": 1180,
    "alt": "Scene \u00b7 Addition comparison: masked source and outputs of ten baselines, with MaskFlow in the final panel."
  },
  {
    "title": "Scene \u00b7 Modification",
    "image": "./assets/figures/gallery/scene-modify.jpg",
    "width": 3576,
    "height": 1270,
    "alt": "Scene \u00b7 Modification comparison: masked source and outputs of ten baselines, with MaskFlow in the final panel."
  },
  {
    "title": "Scene \u00b7 Replacement",
    "image": "./assets/figures/gallery/scene-replace.jpg",
    "width": 3576,
    "height": 1270,
    "alt": "Scene \u00b7 Replacement comparison: masked source and outputs of ten baselines, with MaskFlow in the final panel."
  },
  {
    "title": "Scene \u00b7 Removal",
    "image": "./assets/figures/gallery/scene-remove.jpg",
    "width": 3576,
    "height": 1940,
    "alt": "Scene \u00b7 Removal comparison: masked source and outputs of ten baselines, with MaskFlow in the final panel."
  },
  {
    "title": "Infographics \u00b7 Addition",
    "image": "./assets/figures/gallery/infographics_en-add.jpg",
    "width": 3576,
    "height": 1478,
    "alt": "Infographics \u00b7 Addition comparison: masked source and outputs of ten baselines, with MaskFlow in the final panel."
  },
  {
    "title": "Infographics \u00b7 Modification",
    "image": "./assets/figures/gallery/infographics_en-modify.jpg",
    "width": 3576,
    "height": 2038,
    "alt": "Infographics \u00b7 Modification comparison: masked source and outputs of ten baselines, with MaskFlow in the final panel."
  },
  {
    "title": "Infographics \u00b7 Replacement",
    "image": "./assets/figures/gallery/infographics_en-replace.jpg",
    "width": 3576,
    "height": 1478,
    "alt": "Infographics \u00b7 Replacement comparison: masked source and outputs of ten baselines, with MaskFlow in the final panel."
  },
  {
    "title": "Infographics \u00b7 Removal",
    "image": "./assets/figures/gallery/infographics_en-remove.jpg",
    "width": 3576,
    "height": 1104,
    "alt": "Infographics \u00b7 Removal comparison: masked source and outputs of ten baselines, with MaskFlow in the final panel."
  }
],
  resultFigures: [
  {
    "id": "comparison",
    "figure": "Figure 3",
    "title": "General scene editing",
    "caption": "Comparison across modification, removal, replacement, and addition. MaskFlow follows the mask while preserving neighboring objects and background content.",
    "image": "./assets/figures/comparison.jpg",
    "alt": "Comparison across modification, removal, replacement, and addition. MaskFlow follows the mask while preserving neighboring objects and background content.",
    "width": 2800,
    "height": 1048
  },
  {
    "id": "infographics",
    "figure": "Figure 4",
    "title": "Editing dense infographic layouts",
    "caption": "MaskFlow updates the selected text and visual elements while retaining surrounding illustrations and typography.",
    "image": "./assets/figures/infographics.jpg",
    "alt": "MaskFlow updates the selected text and visual elements while retaining surrounding illustrations and typography.",
    "width": 2800,
    "height": 1119
  },
  {
    "id": "ablation",
    "figure": "Figure 5",
    "title": "Regional control and boundary continuity",
    "caption": "Qualitative ablation of the fine-tuned baseline, MaskFlow without SPD, and the full framework. Red arrows highlight visible errors.",
    "image": "./assets/figures/ablation.jpg",
    "alt": "Qualitative ablation of the fine-tuned baseline, MaskFlow without SPD, and the full framework. Red arrows highlight visible errors.",
    "width": 2800,
    "height": 1099
  }
],
};
