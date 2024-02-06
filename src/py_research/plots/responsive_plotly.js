// Load this script after the plotly.js script to make the plotly figure responsive.

// Reference the script tag that loads this script by an id passed from Python.
const script = document.getElementById(scriptID);

// Get the figure id and dimensions from the script tag.
const figureID = script.getAttribute("fig-id");
const figureWidth = script.getAttribute("fig-width");
const figureHeight = script.getAttribute("fig-height");
const minWidth = script.getAttribute("min-width");
const maxWidth = script.getAttribute("max-width");
const maxHeight = script.getAttribute("max-height");

const aspect_ratio = figureWidth / figureHeight;

const figure = document.getElementById(figureID);

const resize = (containerWidth) => {
  // Calculate the target width and height of the figure.
  targetWidth = Math.min(maxWidth, Math.max(minWidth, containerWidth));
  optimalHeight = targetWidth / aspect_ratio;
  targetHeight = Math.min(maxHeight, optimalHeight);

  // Resize the figure via Plotly.relayout.
  Plotly.relayout(figure, {
    width: targetWidth,
    height: targetHeight,
  });
};

const resizeObserver = new ResizeObserver((entries) => {
  // Get the width of the container from the ResizeObserver entry.
  // The width is stored in different places depending on the browser.
  // Then call the resize function.
  for (let entry of entries) {
    if (entry.contentBoxSize) {
      if (entry.contentBoxSize[0]) {
        resize(entry.contentBoxSize[0].inlineSize);
      } else {
        resize(entry.contentBoxSize.inlineSize);
      }
    } else {
      resize(entry.contentRect.width);
    }
  }
});

const figureContainer = figure.parentElement;

// Resize the figure after the first plot, right after page load.
figure.addEventListener("plotly_afterplot", function () {
  resize(figureContainer.clientWidth);
  figure.removeEventListener("plotly_afterplot", arguments.callee);
});

// Resize the figure when the container is resized.
resizeObserver.observe(figureContainer);
