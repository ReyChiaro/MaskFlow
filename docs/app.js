(function () {
  "use strict";

  const config = window.MASKFLOW_SITE_CONFIG;
  if (!config) return;

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  function renderAuthors() {
    $("[data-authors]").innerHTML = config.authors
      .map((author) => {
        const marker = `${author.affiliation}${author.corresponding ? ",*" : ""}`;
        const href = author.href || "";
        const attributes = href.startsWith("http") ? ' target="_blank" rel="noreferrer"' : "";
        const name = href
          ? `<a class="author-link" href="${href}"${attributes}>${author.name}</a>`
          : author.name;
        return `<span>${name}<sup>${marker}</sup></span>`;
      })
      .join('<span class="author-separator" aria-hidden="true">·</span>');
  }

  function resourceMarkup(resource, compact = false) {
    const classes = ["resource-button", compact ? "compact" : "", resource.primary ? "primary" : ""]
      .filter(Boolean)
      .join(" ");
    const content = `<span class="resource-icon resource-icon-${resource.id}" aria-hidden="true"></span><strong>${resource.label}</strong>`;
    if (!resource.href) {
      return `<span class="${classes} disabled" aria-disabled="true" title="${resource.note || "Coming soon"}">${content}<small>${resource.note || "Soon"}</small></span>`;
    }
    return `<a class="${classes}" href="${resource.href}" target="_blank" rel="noreferrer">${content}</a>`;
  }

  function renderResources() {
    $("[data-resources]").innerHTML = config.resources.map((item) => resourceMarkup(item)).join("");
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
              <h3>${item.title}</h3>
            </div>
            <div class="image-frame">
              <img src="${item.image}" alt="${item.alt}" width="${item.width}" height="${item.height}" loading="lazy" />
              <button class="zoom-button" type="button" aria-label="Enlarge ${item.figure}">View full size</button>
            </div>
            <figcaption class="figure-caption"><strong>${item.figure}.</strong> ${item.caption}</figcaption>
          </figure>`,
      )
      .join("");
  }

  function setupCarousel() {
    const root = $("[data-carousel]");
    const items = config.galleryFigures;
    const viewport = $(".carousel-viewport", root);
    const track = $(".carousel-track", root);
    const status = $(".carousel-status", root);
    const dots = $(".carousel-dots", root);
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    let position = 1;
    let busy = false;
    let resetTimer;
    const slides = [items.at(-1), ...items, items[0]];
    track.innerHTML = slides.map((item, i) => `
      <figure class="carousel-slide" data-zoomable role="group" aria-roledescription="slide" aria-label="${item.title}">
        <h3>${item.title}</h3>
        <div class="carousel-image"><img src="${item.image}" width="${item.width}" height="${item.height}" alt="${item.alt}" loading="lazy" draggable="false" />
        <button class="zoom-button" type="button" aria-label="Enlarge ${item.title}">View full size</button></div>
      </figure>`).join("");
    dots.innerHTML = items.map((item, i) => `<button type="button" aria-label="Show ${item.title}" data-slide="${i}"></button>`).join("");
    function paint(animate = false) {
      track.style.transition = animate && !reduced.matches ? "transform 360ms ease" : "none";
      track.style.transform = `translateX(-${position * 100}%)`;
      const active = (position - 1 + items.length) % items.length;
      status.textContent = `${String(active + 1).padStart(2, "0")} / 08 — ${items[active].title}`;
      $$("[data-slide]", dots).forEach((dot, i) => dot.setAttribute("aria-current", String(i === active)));
      [...track.children].forEach((slide, i) => {
        slide.inert = i !== position;
        slide.setAttribute("aria-hidden", String(i !== position));
      });
    }
    function settle() {
      clearTimeout(resetTimer);
      if (position === 0) position = items.length;
      if (position === items.length + 1) position = 1;
      paint();
      busy = false;
    }
    function move(next) {
      if (busy) return;
      busy = true;
      position = next;
      paint(true);
      resetTimer = setTimeout(settle, reduced.matches ? 0 : 400);
    }
    track.addEventListener("transitionend", (event) => {
      if (event.target === track && event.propertyName === "transform") settle();
    });
    $(".carousel-prev", root).addEventListener("click", () => move(position - 1));
    $(".carousel-next", root).addEventListener("click", () => move(position + 1));
    dots.addEventListener("click", (event) => {
      const dot = event.target.closest("[data-slide]");
      if (dot) move(Number(dot.dataset.slide) + 1);
    });
    viewport.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      move(position + (event.key === "ArrowRight" ? 1 : -1));
    });
    let gesture = null;
    viewport.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || event.target.closest("button") || busy) return;
      gesture = { id: event.pointerId, x: event.clientX, y: event.clientY };
      viewport.setPointerCapture(event.pointerId);
      viewport.classList.add("dragging");
    });
    function release(event) {
      if (!gesture || gesture.id !== event.pointerId) return;
      const dx = event.clientX - gesture.x;
      const dy = event.clientY - gesture.y;
      gesture = null;
      viewport.classList.remove("dragging");
      if (viewport.hasPointerCapture(event.pointerId)) viewport.releasePointerCapture(event.pointerId);
      if (event.type === "pointerup" && Math.abs(dx) > 45 && Math.abs(dx) > Math.abs(dy)) move(position + (dx < 0 ? 1 : -1));
    }
    viewport.addEventListener("pointerup", release);
    viewport.addEventListener("pointercancel", release);
    let wheelAmount = 0;
    let wheelTimer;
    viewport.addEventListener("wheel", (event) => {
      const dx = event.shiftKey && !event.deltaX ? event.deltaY : event.deltaX;
      if (!dx || (!event.shiftKey && Math.abs(dx) <= Math.abs(event.deltaY))) return;
      event.preventDefault();
      clearTimeout(wheelTimer);
      wheelTimer = setTimeout(() => { wheelAmount = 0; }, 150);
      if (busy) return;
      wheelAmount += dx;
      if (Math.abs(wheelAmount) > 45) {
        move(position + (wheelAmount > 0 ? 1 : -1));
        wheelAmount = 0;
      }
    }, { passive: false });
    paint();
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
    const dialogImages = $("[data-lightbox-images]");
    const dialogCaption = $("[data-lightbox-caption]");
    const close = $("[data-lightbox-close]");

    function openLightbox(figure) {
      const images = $$("img", figure);
      const image = images[0];
      const caption = $("figcaption", figure);
      dialogImages.replaceChildren(...images.map((source) => {
        const enlarged = document.createElement("img");
        enlarged.src = source.currentSrc || source.src;
        enlarged.alt = source.alt;
        return enlarged;
      }));
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
  setupCarousel();
  setupLightbox();
  setupCitationCopy();
  $("[data-year]").textContent = new Date().getFullYear();
})();
