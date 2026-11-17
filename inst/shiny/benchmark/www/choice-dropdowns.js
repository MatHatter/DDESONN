// Keep Selectize's accessible keyboard navigation, without a text-entry field.
(function () {
  function choiceOnly() {
    document.querySelectorAll('.selectize-input > input').forEach(function (input) {
      if (input.readOnly) return;
      input.readOnly = true;
      input.setAttribute('inputmode', 'none');
      input.setAttribute('aria-readonly', 'true');
      input.addEventListener('beforeinput', function (event) { event.preventDefault(); });
      input.addEventListener('paste', function (event) { event.preventDefault(); });
    });
  }
  function start() {
    choiceOnly();
    var scheduled = false;
    new MutationObserver(function () {
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(function () { scheduled = false; choiceOnly(); });
    }).observe(document.body, { childList: true, subtree: true });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
