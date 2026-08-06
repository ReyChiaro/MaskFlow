(function () {
  "use strict";

  const config = window.MASKFLOW_SITE_CONFIG;
  if (!config) return;

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  function renderAuthors() {
    $("[data-authors]").innerHTML = config.authors
      .map((author) => `<span>${author.name}<sup>${author.affiliation}</sup></span>`)
      .join('<span class="author-separator" aria-hidden="true">·</span>');
  }

  function resourceMarkup(resource, compact = false) {
    const classes = ["resource-button", compact ? "compact" : "", resource.primary ? "primary" : ""]
      .filter(Boolean)
      .join(" ");
    const content = `<strong>${resource.label}</strong>`;
    if (!resource.href) {
      return `<span class="${classes} disabled" aria-disabled="true" title="${resource.note || "Coming soon"}">${content}<small>${resource.note || "Soon"}</small></span>`;
    }
    return `<a class="${classes}" href="${resource.href}" target="_blank" rel="noreferrer">${content}</a>`;
  }

  function renderResources() {
    $("[data-resources]").innerHTML = config.resources.slice(0, 4).map((item) => resourceMarkup(item)).join("");
    const primary = config.resources.find((item) => item.primary) || config.resources[0];
    $("[data-primary-resource]").innerHTML = resourceMarkup(primary, true);
    const dataset = config.resources.find((item) => item.id === "dataset");
    const datasetTarget = $("[data-dataset-resource]");
    if (datasetTarget && dataset) datasetTarget.innerHTML = resourceMarkup(dataset, true);
  }

  function renderResults() {
    const list = $("[data-result-list]");
    list.innerHTML = config.resultFigures
      .map(
        (item) => `
          <figure class="figure-block result-figure reveal" data-zoomable>
            <div class="figure-heading">
              <span>${item.kicker}</span>
              <h3>${item.title}</h3>
            </div>
            <div class="image-frame">
              <img src="${item.image}" alt="${item.alt}" loading="lazy" />
              <button class="zoom-button" type="button" aria-label="Enlarge ${item.figure}">View full size</button>
            </div>
            <figcaption class="figure-caption"><strong>${item.figure}.</strong> ${item.caption}</figcaption>
          </figure>`,
      )
      .join("");
  }

  function renderCitation() {
    if (!config.citation.trim()) return;
    $("[data-citation]").textContent = config.citation.trim();
    $("[data-copy-citation]").disabled = false;
  }

  function setupNavigation() {
    const header = $("[data-header]");
    const toggle = $(".menu-toggle");
    const panel = $("#nav-panel");
    const progress = $(".scroll-progress span");

    function updateScroll() {
      header.classList.toggle("scrolled", window.scrollY > 10);
      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      progress.style.transform = `scaleX(${scrollable > 0 ? window.scrollY / scrollable : 0})`;
    }

    toggle.addEventListener("click", () => {
      const open = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!open));
      panel.classList.toggle("open", !open);
      document.body.classList.toggle("menu-open", !open);
    });
    $$("a", panel).forEach((link) =>
      link.addEventListener("click", () => {
        toggle.setAttribute("aria-expanded", "false");
        panel.classList.remove("open");
        document.body.classList.remove("menu-open");
      }),
    );
    window.addEventListener("scroll", updateScroll, { passive: true });
    updateScroll();
  }

  function setupReveal() {
    const targets = $$(".reveal");
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      targets.forEach((target) => target.classList.add("visible"));
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        });
      },
      { threshold: 0.1, rootMargin: "0px 0px -35px" },
    );
    targets.forEach((target) => observer.observe(target));
  }

  function setupLightbox() {
    const dialog = $("[data-lightbox]");
    const dialogImage = $("[data-lightbox-image]");
    const dialogCaption = $("[data-lightbox-caption]");
    const close = $("[data-lightbox-close]");

    function openLightbox(figure) {
      const image = $("img", figure);
      const caption = $("figcaption", figure);
      dialogImage.src = image.currentSrc || image.src;
      dialogImage.alt = image.alt;
      dialogCaption.textContent = caption?.textContent.trim() || image.alt;
      dialog.showModal();
      document.body.classList.add("lightbox-open");
    }

    $$('[data-zoomable]').forEach((figure) => {
      $(".zoom-button", figure)?.addEventListener("click", () => openLightbox(figure));
      $("img", figure)?.addEventListener("dblclick", () => openLightbox(figure));
    });
    close.addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
    dialog.addEventListener("close", () => document.body.classList.remove("lightbox-open"));
  }

  function setupCitationCopy() {
    const button = $("[data-copy-citation]");
    const toast = $("[data-toast]");
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(config.citation.trim());
        button.textContent = "Copied";
        toast.classList.add("visible");
        window.setTimeout(() => {
          button.textContent = "Copy";
          toast.classList.remove("visible");
        }, 1600);
      } catch (_) {
        button.textContent = "Select text to copy";
      }
    });
  }

  renderAuthors();
  renderResources();
  renderResults();
  renderCitation();
  setupNavigation();
  setupReveal();
  setupLightbox();
  setupCitationCopy();
  $("[data-year]").textContent = new Date().getFullYear();
})();
