(function () {
  const BUTTON_ID = "YCXC33LAEKQ78";
  const CONTAINER_SELECTOR = "#paypal-container-YCXC33LAEKQ78";
  const SDK_ID = "paypal-sdk-strategy-lab";

  function loadSdk() {
    const container = document.querySelector(CONTAINER_SELECTOR);
    if (!container || document.getElementById(SDK_ID) || window.paypal) return false;
    if (container.dataset && container.dataset.paypalRendered === "error") return false;
    const clientId = container.dataset ? container.dataset.paypalClientId || "" : "";
    if (!clientId) return false;
    const script = document.createElement("script");
    script.id = SDK_ID;
    script.src =
      "https://www.paypal.com/sdk/js?client-id=" +
      encodeURIComponent(clientId) +
      "&components=hosted-buttons&enable-funding=venmo&currency=USD";
    script.async = true;
    script.addEventListener("load", renderOnce, { once: true });
    script.addEventListener(
      "error",
      () => {
        script.remove();
        if (container.dataset) container.dataset.paypalRendered = "error";
      },
      { once: true },
    );
    document.head.appendChild(script);
    return true;
  }

  function canRender() {
    return (
      typeof window !== "undefined" &&
      window.paypal &&
      window.paypal.HostedButtons &&
      document.querySelector(CONTAINER_SELECTOR)
    );
  }

  function alreadyRendered(el) {
    return (
      el &&
      el.dataset &&
      (el.dataset.paypalRendered === "1" || el.dataset.paypalRendered === "pending")
    );
  }

  function renderOnce() {
    const el = document.querySelector(CONTAINER_SELECTOR);
    if (!el) return false;
    if (alreadyRendered(el)) return true;
    if (!window.paypal || !window.paypal.HostedButtons) return false;

    if (el.dataset) el.dataset.paypalRendered = "pending";
    Promise.resolve(
      window.paypal
        .HostedButtons({
          hostedButtonId: BUTTON_ID,
        })
        .render(CONTAINER_SELECTOR),
    )
      .then(() => {
        if (el.dataset) el.dataset.paypalRendered = "1";
      })
      .catch(() => {
        if (el.dataset) el.dataset.paypalRendered = "error";
      });
    return true;
  }

  function tick() {
    if (!document.querySelector(CONTAINER_SELECTOR)) return;
    loadSdk();
    if (canRender()) {
      renderOnce();
    }
  }

  window.addEventListener("load", tick);

  // Dash swaps page content dynamically; observe DOM changes.
  const obs = new MutationObserver(() => {
    loadSdk();
    if (canRender()) renderOnce();
  });
  obs.observe(document.documentElement, { childList: true, subtree: true });
})();
