// Keep usage tooltips within the viewport and vertically clear of their point.
// This augments native Plotly hover events; selection still belongs to Streamlit.
(() => {
  window.__semlensUsageHover?.destroy();
  const bindings = new Map();
  const tip = document.createElement('div');
  tip.className = 'semlens-usage-tooltip';
  tip.setAttribute('role', 'tooltip');
  tip.hidden = true;
  document.body.appendChild(tip);
  let owner = null, hideTimer = null, scanFrame = null;

  function hide() {
    clearTimeout(hideTimer);
    tip.hidden = true;
    owner = null;
  }

  // Recreate only our text-formatting tags, never input attributes or scripts.
  function content(text) {
    const parsed = new DOMParser().parseFromString(String(text), 'text/html');
    const fragment = document.createDocumentFragment();
    function copy(node, parent, first = false) {
      if (node.nodeType === Node.TEXT_NODE) {
        parent.appendChild(document.createTextNode(node.textContent));
      } else if (node.nodeType === Node.ELEMENT_NODE) {
        if (['B', 'I', 'BR'].includes(node.tagName)) {
          // The opening "[corpus]" stays bold; any other bold text is the
          // target occurrence, which is highlighted instead.
          const header = first && node.textContent.startsWith('[');
          const tag = node.tagName === 'B' && !header ? 'mark' : node.tagName.toLowerCase();
          const element = document.createElement(tag);
          parent.appendChild(element);
          node.childNodes.forEach(child => copy(child, element));
        } else {
          parent.appendChild(document.createTextNode(node.textContent));
        }
      }
    }
    parsed.body.childNodes.forEach((node, index) => copy(node, fragment, index === 0));
    tip.replaceChildren(fragment);
  }

  function show(chart, event) {
    const point = event.points?.[0];
    const text = point?.hovertext ?? point?.text;
    if (!point?.xaxis || !point?.yaxis || !text) return;
    clearTimeout(hideTimer);
    const rect = chart.getBoundingClientRect();
    const x = rect.left + point.xaxis._offset + point.xaxis.d2p(point.x);
    const y = rect.top + point.yaxis._offset + point.yaxis.d2p(point.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    owner = chart;
    content(text);
    tip.hidden = false;
    tip.style.visibility = 'hidden';
    tip.style.maxHeight = 'none';
    tip.style.left = '12px';
    tip.style.top = '0px';
    const natural = tip.getBoundingClientRect();
    const topEdge = Math.max(12, (document.querySelector('[data-testid="stHeader"]')?.getBoundingClientRect().bottom ?? 0) + 8);
    const gap = 18;
    const above = Math.max(0, y - gap - topEdge);
    const below = Math.max(0, window.innerHeight - 12 - y - gap);
    const useAbove = above >= natural.height || (below < natural.height && above > below);
    const available = useAbove ? above : below;
    tip.style.maxHeight = `${Math.max(1, available)}px`;
    const height = tip.getBoundingClientRect().height;
    tip.style.left = `${Math.max(12, Math.min(x - natural.width / 2, window.innerWidth - natural.width - 12))}px`;
    tip.style.top = `${useAbove ? y - gap - height : y + gap}px`;
    tip.dataset.placement = useAbove ? 'above' : 'below';
    tip.style.visibility = 'visible';
  }

  function unbind(chart, handlers) {
    for (const [event, handler] of Object.entries(handlers)) chart.removeListener?.(event, handler);
    delete chart.dataset.semlensUsageHover;
    bindings.delete(chart);
    if (owner === chart) hide();
  }

  // Sentence cells show one scrolling line: start each with its target in view.
  function centreTargets() {
    document.querySelectorAll('.semlens-sentence-cell:not([data-centred])').forEach(cell => {
      cell.dataset.centred = 'true';
      const mark = cell.querySelector('mark');
      if (mark) cell.scrollLeft = Math.max(0, mark.offsetLeft - (cell.clientWidth - mark.offsetWidth) / 2);
    });
  }

  function scan() {
    scanFrame = null;
    centreTargets();
    for (const [chart, handlers] of bindings) {
      if (!chart.isConnected) unbind(chart, handlers);
    }
    document.querySelectorAll('.js-plotly-plot').forEach(chart => {
      if (bindings.has(chart) || !chart.on || !chart.layout?.meta?.semlens_usage_hover) return;
      const handlers = {
        plotly_hover: event => show(chart, event),
        plotly_unhover: () => { hideTimer = setTimeout(hide, 220); },
        plotly_relayout: hide,
      };
      for (const [event, handler] of Object.entries(handlers)) chart.on(event, handler);
      bindings.set(chart, handlers);
      chart.dataset.semlensUsageHover = 'true';
    });
  }

  const observer = new MutationObserver(() => {
    if (scanFrame === null) scanFrame = requestAnimationFrame(scan);
  });
  observer.observe(document.body, {childList: true, subtree: true});
  tip.addEventListener('pointerenter', () => clearTimeout(hideTimer));
  tip.addEventListener('pointerleave', hide);
  function scrolled(event) { if (!tip.contains(event.target)) hide(); }
  function keydown(event) { if (event.key === 'Escape') hide(); }
  document.addEventListener('scroll', scrolled, true);
  document.addEventListener('keydown', keydown);
  window.addEventListener('resize', hide);
  scan();
  window.__semlensUsageHover = {
    destroy() {
      observer.disconnect();
      if (scanFrame !== null) cancelAnimationFrame(scanFrame);
      hide();
      for (const [chart, handlers] of bindings) unbind(chart, handlers);
      tip.remove();
      document.removeEventListener('scroll', scrolled, true);
      document.removeEventListener('keydown', keydown);
      window.removeEventListener('resize', hide);
    },
  };
})();
