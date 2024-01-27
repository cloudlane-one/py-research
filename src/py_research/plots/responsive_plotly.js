const plotlyJsURL = document.currentScript.getAttribute("plotly-js-url");

plotlyModule = await import(plotlyJsURL);
Plotly = plotlyModule.default;

const figureID = document.currentScript.getAttribute("fig-id");
const figureWidth = document.currentScript.getAttribute("fig-width");
const figureHeight = document.currentScript.getAttribute("fig-height");
const sizing = new Map(
  [0, 1, 2, 3].map((i) => [
    document.currentScript.getAttribute(`fig-bp-width-${i}`),
    document.currentScript.getAttribute(`fig-bp-mode-${i}`),
  ])
);

const aspect_ratio = figureWidth / figureHeight;

const get_matching_mode = (width) => {
  matching = Array(sizing.keys()).filter((key) =>
    key == "inf" ? true : width <= key
  );
  return matching.length > 0 ? sizing.get(matching[matching.length - 1]) : null;
};

let scalingMode = false;

const activateScaling = () => {
  const figure = document.getElementById(figureID);

  virtualWidth = figure.clientWidth;
  virtualHeight = figure.clientHeight;

  figure.style.width = "100%";
  figure.removeAttribute("height");

  plotContainer = figure.getElementsByClassName("svg-container")[0];
  plotContainer.style.width = "100%";
  plotContainer.removeAttribute("height");

  viewBox = `0 0 ${virtualWidth} ${virtualHeight}`;

  const svgs = plotContainer.getElementsByTagName("svg");

  for (let svg of svgs) {
    svg.setAttribute("viewBox", viewBox);
    plotContainer.removeAttribute("width");
    plotContainer.removeAttribute("height");
  }

  scalingMode = true;
};

const resize = (containerWidth) => {
  const mode = get_matching_mode(containerWidth);
  if ((mode == "scale-image") & !scalingMode) {
    activateScaling();
  } else if (mode == "stretch-width") {
    Plotly.relayout(figure, { width: containerWidth });
  } else if (mode == "scale-layout") {
    Plotly.relayout(figure, {
      width: containerWidth,
      height: containerWidth / aspect_ratio,
    });
  }
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
  resize(figureContainer.clientWidth);
};

resizeObserver.observe(figureContainer);
