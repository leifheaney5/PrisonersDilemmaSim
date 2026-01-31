(function () {
  const BUTTON_ID = "YCXC33LAEKQ78";
  const CONTAINER_SELECTOR = "#paypal-container-YCXC33LAEKQ78";

  function canRender() {
    return (
      typeof window !== "undefined" &&
      window.paypal &&
      window.paypal.HostedButtons &&
      document.querySelector(CONTAINER_SELECTOR)
    );
  }

  function alreadyRendered(el) {
    return el && el.dataset && el.dataset.paypalRendered === "1";
  }

  function renderOnce() {
    const el = document.querySelector(CONTAINER_SELECTOR);
    if (!el) return false;
    if (alreadyRendered(el)) return true;
    if (!window.paypal || !window.paypal.HostedButtons) return false;

    el.dataset.paypalRendered = "1";
    window.paypal
      .HostedButtons({
        hostedButtonId: BUTTON_ID,
      })
      .render(CONTAINER_SELECTOR);
    return true;
  }

  function tick() {
    if (canRender()) {
      renderOnce();
    } else {
      setTimeout(tick, 300);
    }
  }

  window.addEventListener("load", tick);

  // Dash swaps page content dynamically; observe DOM changes.
  const obs = new MutationObserver(() => {
    if (canRender()) renderOnce();
  });
  obs.observe(document.documentElement, { childList: true, subtree: true });
})();

