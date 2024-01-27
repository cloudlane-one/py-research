const script = document.getElementById(scriptID); // scriptID is from Python

const figureID = script.getAttribute("fig-id");
const figureWidth = script.getAttribute("fig-width");
const figureHeight = script.getAttribute("fig-height");
const minWidth = script.getAttribute("min-width");
const maxWidth = script.getAttribute("max-width");
const maxHeight = script.getAttribute("max-height");

const aspect_ratio = figureWidth / figureHeight;

const figure = document.getElementById(figureID);

const resize = (containerWidth) => {
  targetWidth = Math.min(maxWidth, Math.max(minWidth, containerWidth));
  optimalHeight = containerWidth / aspect_ratio;
  targetHeight = Math.min(maxHeight, optimalHeight);

  Plotly.relayout(figure, {
    width: targetWidth,
    height: targetHeight,
  });
};

const resizeObserver = new ResizeObserver((entries) => {
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

window.onload = () => {
  window.setTimeout(() => {
    resize(figureContainer.clientWidth);
  }, 200);
};

resizeObserver.observe(figureContainer);
