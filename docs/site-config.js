/**
 * MaskFlow project-page configuration.
 * Add public URLs and the final BibTeX here; the UI updates automatically.
 */
window.MASKFLOW_SITE_CONFIG = {
  title: "MaskFlow",
  fullTitle: "MaskFlow: Precise, Consistent and Seamless Regional Image Editing",
  authors: [
    { name: "Rui Xu", affiliation: "1" },
    { name: "Yang Yong", affiliation: "1" },
    { name: "Shunzi Yang", affiliation: "2" },
    { name: "Ruihao Gong", affiliation: "1,2" },
    { name: "Chengtao Lv", affiliation: "1,3" },
  ],
  resources: [
    { id: "paper", label: "Paper", href: "", note: "https://arxiv.org/abs/2608.06929", primary: true },
    { id: "code", label: "Code", href: "", note: "https://github.com/ReyChiaro/MaskFlow" },
    { id: "model", label: "Model", href: "", note: "https://huggingface.co/ReyChiaro/MaskFlow" },
    { id: "demo", label: "Demo", href: "", note: "Coming soon" },
    { id: "dataset", label: "Dataset", href: "", note: "https://huggingface.co/datasets/ReyChiaro/MaskFlow" },
  ],
  citation: "@misc{xu2026maskflowpreciseconsistentseamless,
      title={MaskFlow: Precise, Consistent and Seamless Regional Image Editing}, 
      author={Rui Xu and Yang Yong and Shunzi Yang and Ruihao Gong and Chengtao Lv},
      year={2026},
      eprint={2608.06929},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.06929}, 
}",
  resultFigures: [
    {
      id: "comparison",
      tab: "Comparison",
      figure: "Figure 2",
      kicker: "Qualitative comparison",
      title: "Faithful edits across diverse scenes",
      caption:
        "Compared with commercial systems, general editors, and mask-conditioned methods, MaskFlow better localizes the requested edit, preserves unrelated content, and removes visible boundary seams.",
      image: "./assets/figures/comparison.jpg",
      alt: "Qualitative comparison of MaskFlow and nine regional image editing baselines",
    },
    {
      id: "ablation",
      tab: "Ablation",
      figure: "Figure 3",
      kicker: "Component study",
      title: "Regional control and de-seaming work together",
      caption:
        "MaskFlow constrains the editable region and preserves the background. Soft-Poisson de-seaming then improves local continuity around the boundary.",
      image: "./assets/figures/ablation.jpg",
      alt: "Ablation comparison of the base model, MaskFlow, and MaskFlow with Soft-Poisson de-seaming",
    },
    {
      id: "infographics",
      tab: "Infographics",
      figure: "Figure 4",
      kicker: "Challenging application",
      title: "Precise editing in dense visual layouts",
      caption:
        "In infographic editing, masks resolve targets that are difficult to specify with language alone while MaskFlow preserves nearby text and visual elements.",
      image: "./assets/figures/infographics.jpg",
      alt: "MaskFlow editing results on complex infographic images",
    },
  ],
};
