// Report legend toggles of corpora back to Python, so a hidden corpus stays
// hidden when the plot is redrawn for another view, axis or annotation.
// Plotly applies the toggle itself; this only observes the result.
export default function (component) {
  const { data, setTriggerValue } = component;
  window.__semlensCorpusVisibility?.destroy();
  const bindings = new Map();
  let hidden = new Set(data?.hidden ?? []);
  let scanFrame = null;

  function report(chart) {
    const seen = new Set();
    const now = [];
    for (const trace of chart._fullData ?? []) {
      const corpus = trace.meta?.semlens_corpus;
      if (typeof corpus !== 'string' || seen.has(corpus)) continue;
      seen.add(corpus);
      if (trace.visible === 'legendonly') now.push(corpus);
    }
    if (!seen.size) return;
    if (now.length === hidden.size && now.every(corpus => hidden.has(corpus))) return;
    hidden = new Set(now);
    setTriggerValue('toggled', now);
  }

  function scan() {
    scanFrame = null;
    for (const [chart, handler] of bindings) {
      if (!chart.isConnected) {
        chart.removeListener?.('plotly_restyle', handler);
        bindings.delete(chart);
      }
    }
    document.querySelectorAll('.js-plotly-plot').forEach(chart => {
      if (bindings.has(chart) || !chart.on || !chart.layout?.meta?.semlens_usage_hover) return;
      const handler = () => report(chart);
      chart.on('plotly_restyle', handler);
      bindings.set(chart, handler);
    });
  }

  const observer = new MutationObserver(() => {
    if (scanFrame === null) scanFrame = requestAnimationFrame(scan);
  });
  observer.observe(document.body, { childList: true, subtree: true });
  scan();

  function destroy() {
    observer.disconnect();
    if (scanFrame !== null) cancelAnimationFrame(scanFrame);
    for (const [chart, handler] of bindings) chart.removeListener?.('plotly_restyle', handler);
    bindings.clear();
    if (window.__semlensCorpusVisibility?.destroy === destroy) delete window.__semlensCorpusVisibility;
  }
  window.__semlensCorpusVisibility = { destroy };
  return destroy;
}
