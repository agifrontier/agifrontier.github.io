(() => {
  const loader = document.currentScript;
  const ninjaKeys = document.querySelector("ninja-keys");
  const button = document.getElementById("search-toggle");
  const status = document.getElementById("search-status");
  let ready;
  let openRequested = false;
  let componentAttempts = 0;

  function loadScript(url, type) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = url;
      if (type) script.type = type;
      const timeout = setTimeout(() => finish(new Error(`Search resource timed out: ${url}`)), 20000);

      function finish(error) {
        clearTimeout(timeout);
        script.onload = null;
        script.onerror = null;
        if (error) {
          script.remove();
          reject(error);
        } else {
          resolve();
        }
      }

      script.onload = () => finish();
      script.onerror = () => finish(new Error(`Search resource failed: ${url}`));
      document.head.appendChild(script);
    });
  }

  async function prepareSearch() {
    if (!customElements.get("ninja-keys")) {
      const componentUrl = new URL(loader.dataset.searchComponent, document.baseURI);
      // Browsers cache failed module loads; retry with a fresh root-module URL.
      if (componentAttempts > 0) componentUrl.searchParams.set("search_retry", componentAttempts);
      componentAttempts += 1;
      await loadScript(componentUrl.href, "module");
    }
    if (!customElements.get("ninja-keys")) {
      throw new Error("Search component did not register ninja-keys");
    }
    await loadScript(loader.dataset.searchIndex);
    if (!Array.isArray(ninjaKeys.data) || ninjaKeys.data.length === 0) {
      throw new Error("Search index is empty or invalid");
    }
    await ninjaKeys.updateComplete;
  }

  window.openSearchModal = async () => {
    openRequested = true;
    button.setAttribute("aria-busy", "true");
    status.textContent = "正在加载搜索…";
    status.hidden = false;
    if (!ready) {
      ready = prepareSearch().catch((error) => {
        ready = undefined;
        throw error;
      });
    }
    try {
      await ready;
      status.hidden = true;
      if (openRequested) {
        openRequested = false;
        ninjaKeys.classList.toggle("dark", determineComputedTheme() === "dark");
        const navbar = $("#navbarNav");
        if (navbar.hasClass("show")) navbar.collapse("hide");
        ninjaKeys.open();
      }
      return true;
    } catch (error) {
      openRequested = false;
      status.textContent = "搜索加载失败，请重试；若仍失败请刷新页面。";
      status.hidden = false;
      console.error("Search initialization failed", error);
      return false;
    } finally {
      button.setAttribute("aria-busy", "false");
    }
  };

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") openRequested = false;
    if ((event.ctrlKey || event.metaKey) && !event.altKey && !event.shiftKey && event.key.toLowerCase() === "k") {
      event.preventDefault();
      event.stopImmediatePropagation();
      if (!event.repeat) window.openSearchModal();
    }
  }, true);

  if (navigator.platform.toUpperCase().includes("MAC")) {
    button.querySelector(".nav-link").firstChild.textContent = "⌘ k ";
  }
})();
