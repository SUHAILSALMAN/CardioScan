/**
 * CardioScan — front-end behaviour
 * - Initializes Bootstrap tooltips on the info icons.
 * - Runs Bootstrap-style client-side validation before submitting.
 * - Sends the form via fetch() to /predict and renders the JSON response
 *   into the result panel without a full page reload.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ---- Tooltips ----
  document
    .querySelectorAll('[data-bs-toggle="tooltip"]')
    .forEach((el) => new bootstrap.Tooltip(el));

  const form = document.getElementById("predict-form");
  const submitBtn = document.getElementById("submit-btn");
  const submitSpinner = document.getElementById("submit-spinner");
  const submitLabel = submitBtn.querySelector(".submit-label");
  const formAlert = document.getElementById("form-alert");

  const placeholder = document.getElementById("result-placeholder");
  const resultContent = document.getElementById("result-content");
  const resultIcon = document.getElementById("result-icon");
  const resultLabel = document.getElementById("result-label");
  const confidenceValue = document.getElementById("confidence-value");
  const confidenceBar = document.getElementById("confidence-bar");
  const probPresence = document.getElementById("prob-presence");

  function showFormError(message) {
    formAlert.textContent = message;
    formAlert.classList.remove("d-none");
  }

  function hideFormError() {
    formAlert.classList.add("d-none");
    formAlert.textContent = "";
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitSpinner.classList.toggle("d-none", !isLoading);
    submitLabel.classList.toggle("d-none", isLoading);
  }
function renderResult(data) {

  const isPresence = data.prediction === "Heart Disease Detected";

  placeholder.classList.add("d-none");
  resultContent.classList.remove("d-none");

  resultIcon.className =
    "cs-result-icon " +
    (isPresence ? "cs-result-icon--presence" : "cs-result-icon--absence");

  resultIcon.innerHTML = `
    <i class="bi ${
      isPresence
        ? "bi-exclamation-triangle-fill"
        : "bi-check-circle-fill"
    }"></i>`;

  resultLabel.textContent = data.prediction;

  resultLabel.className =
    "cs-result-label " +
    (isPresence
      ? "cs-result-label--presence"
      : "cs-result-label--absence");

  confidenceValue.textContent = data.confidence_pct.toFixed(2) + "%";

  confidenceBar.style.width = data.confidence_pct + "%";

  confidenceBar.className =
    "progress-bar " +
    (isPresence
      ? "progress-bar--presence"
      : "progress-bar--absence");

  probPresence.textContent =
    data.probability_presence_pct.toFixed(2) + "%";

  resultContent.scrollIntoView({
    behavior: "smooth",
    block: "nearest",
  });
}

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    hideFormError();

    // Bootstrap-style client-side validation.
    if (!form.checkValidity()) {
      event.stopPropagation();
      form.classList.add("was-validated");
      return;
    }
    form.classList.add("was-validated");

    setLoading(true);
    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: new FormData(form),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        showFormError(data.error || "Something went wrong. Please check your inputs and try again.");
        return;
      }

      renderResult(data);
    } catch (err) {
      showFormError("Could not reach the server. Please check your connection and try again.");
    } finally {
      setLoading(false);
    }
  });

  form.addEventListener("reset", () => {
    hideFormError();
    form.classList.remove("was-validated");
    resultContent.classList.add("d-none");
    placeholder.classList.remove("d-none");
  });
});
