(function () {
  const BUTTON_ID = "YCXC33LAEKQ78";
  const CONTAINER_SELECTOR = "#paypal-container-YCXC33LAEKQ78";
  const SDK_ID = "paypal-sdk-strategy-lab";

  function loadSdk() {
    const container = document.querySelector(CONTAINER_SELECTOR);
    if (!container || document.getElementById(SDK_ID) || window.paypal) return;
    const clientId = container.dataset ? container.dataset.paypalClientId || "" : "";
    if (!clientId) return;
    const script = document.createElement("script");
    script.id = SDK_ID;
    script.src =
      "https://www.paypal.com/sdk/js?client-id=" +
      encodeURIComponent(clientId) +
      "&components=hosted-buttons&enable-funding=venmo&currency=USD";
    script.async = true;
    document.head.appendChild(script);
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
    loadSdk();
    if (canRender()) {
      renderOnce();
    } else {
      setTimeout(tick, 300);
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
