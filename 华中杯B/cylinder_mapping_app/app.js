const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const controls = {
  centerX: document.getElementById("centerX"),
  centerY: document.getElementById("centerY"),
  cylinderHeight: document.getElementById("cylinderHeight"),
  cylinderRadius: document.getElementById("cylinderRadius"),
  imageBottomZ: document.getElementById("imageBottomZ"),
  imageHeight: document.getElementById("imageHeight"),
  lightTheta: document.getElementById("lightTheta"),
  imageFile: document.getElementById("imageFile"),
  imagePath: document.getElementById("imagePath"),
  imagePreview: document.getElementById("imagePreview"),
  imageEmpty: document.getElementById("imageEmpty"),
};

const A4_WIDTH = 210;
const A4_HEIGHT = 297;
const VIEW_CENTER = [A4_WIDTH / 2, A4_HEIGHT / 2, 0];

const state = {
  rotation: multiplyMatrix(rotationX(-0.9), rotationZ(0.12)),
  distance: 620,
  panX: 0,
  panY: 0,
  drag: null,
  pointerMode: "rotate",
  imageUrl: null,
};

function rotationX(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ];
}

function rotationY(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c],
  ];
}

function rotationZ(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1],
  ];
}

function multiplyMatrix(a, b) {
  const result = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      result[row][col] =
        a[row][0] * b[0][col] +
        a[row][1] * b[1][col] +
        a[row][2] * b[2][col];
    }
  }
  return result;
}

function transformPoint(point) {
  const x = point[0] - VIEW_CENTER[0];
  const y = point[1] - VIEW_CENTER[1];
  const z = point[2] - VIEW_CENTER[2];
  const r = state.rotation;
  return [
    r[0][0] * x + r[0][1] * y + r[0][2] * z,
    r[1][0] * x + r[1][1] * y + r[1][2] * z,
    r[2][0] * x + r[2][1] * y + r[2][2] * z,
  ];
}

function project(point) {
  const [x, y, z] = transformPoint(point);
  const d = Math.max(80, state.distance - z);
  const focal = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.86;
  const scale = focal / d;
  return {
    x: canvas.clientWidth / 2 + state.panX + x * scale,
    y: canvas.clientHeight / 2 + state.panY - y * scale,
    z,
  };
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function polygonPath(points) {
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.closePath();
}

function drawPaper() {
  const world = [
    [0, 0, 0],
    [A4_WIDTH, 0, 0],
    [A4_WIDTH, A4_HEIGHT, 0],
    [0, A4_HEIGHT, 0],
  ];
  const paper = world.map(project);
  const normal = transformPoint([0, 0, 1]);
  const front = normal[2] >= 0;

  ctx.save();
  polygonPath(paper);
  const gradient = ctx.createLinearGradient(paper[0].x, paper[0].y, paper[2].x, paper[2].y);
  if (front) {
    gradient.addColorStop(0, "#ffffff");
    gradient.addColorStop(0.55, "#f7f9fb");
    gradient.addColorStop(1, "#e9eef4");
  } else {
    gradient.addColorStop(0, "#e5ebf2");
    gradient.addColorStop(1, "#f7f9fb");
  }
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.strokeStyle = front ? "rgba(38, 52, 66, 0.45)" : "rgba(38, 52, 66, 0.3)";
  ctx.lineWidth = 1.6;
  ctx.stroke();

  drawPaperGrain(paper, front);
  drawPaperAxes(front);
  ctx.restore();
}

function drawSegment(a, b, color, width = 1) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function getCylinderParams() {
  return {
    centerX: Number.parseFloat(controls.centerX.value) || 0,
    centerY: Number.parseFloat(controls.centerY.value) || 0,
    height: Math.max(0, Number.parseFloat(controls.cylinderHeight.value) || 0),
    radius: Math.max(0, Number.parseFloat(controls.cylinderRadius.value) || 0),
  };
}

function getImageBottomZ() {
  return Number.parseFloat(controls.imageBottomZ.value) || 0;
}

function getImageHeight() {
  return Math.max(1, Number.parseFloat(controls.imageHeight.value) || 90);
}

function getLightThetaRadians() {
  const degrees = Number.parseFloat(controls.lightTheta.value) || 0;
  const clamped = Math.max(-85, Math.min(85, degrees));
  return (clamped * Math.PI) / 180;
}

function drawArrow(startWorld, endWorld, color, label) {
  const start = project(startWorld);
  const end = project(endWorld);
  drawSegment(start, end, color, 2.4);

  const angle = Math.atan2(end.y - start.y, end.x - start.x);
  const head = 10;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(end.x, end.y);
  ctx.lineTo(end.x - head * Math.cos(angle - 0.45), end.y - head * Math.sin(angle - 0.45));
  ctx.lineTo(end.x - head * Math.cos(angle + 0.45), end.y - head * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fill();

  ctx.font = "13px Segoe UI, sans-serif";
  ctx.fillText(label, end.x + 8, end.y - 8);
}

function drawPaperAxes(front) {
  ctx.save();
  const alpha = front ? 0.9 : 0.55;
  const { height } = getCylinderParams();
  const imageTop = getImageBottomZ() + getImageHeight();
  const sceneHeight = Math.max(height, imageTop);
  const zAxisHeight = Math.max(30, sceneHeight + Math.max(12, sceneHeight * 0.08));
  drawArrow([0, 0, 0.4], [A4_WIDTH * 0.92, 0, 0.4], `rgba(214, 69, 54, ${alpha})`, "x");
  drawArrow([0, 0, 0.4], [0, A4_HEIGHT * 0.92, 0.4], `rgba(38, 132, 82, ${alpha})`, "y");
  drawArrow([0, 0, 0.4], [0, 0, zAxisHeight], `rgba(42, 91, 215, ${alpha})`, "z");

  const origin = project([0, 0, 0.6]);
  ctx.fillStyle = `rgba(32, 42, 52, ${alpha})`;
  ctx.font = "12px Segoe UI, sans-serif";
  ctx.fillText("O(0,0)", origin.x + 8, origin.y + 16);
  ctx.restore();
}

function semiCylinderPoint(params, angle, z) {
  return [
    params.centerX + params.radius * Math.cos(angle),
    params.centerY + params.radius * Math.sin(angle),
    z,
  ];
}

function getImagePlaneMetrics(params, image) {
  const imageHeight = getImageHeight();
  const imageWidth = imageHeight * (image.naturalWidth / image.naturalHeight);
  const bottomZ = getImageBottomZ();
  const topZ = bottomZ + imageHeight;
  const outsideGap = Math.max(24, params.radius * 0.6);
  const y = Math.max(params.centerY + params.radius + outsideGap, A4_HEIGHT + outsideGap);
  return {
    imageHeight,
    imageWidth,
    bottomZ,
    topZ,
    y,
    leftX: params.centerX - imageWidth / 2,
    rightX: params.centerX + imageWidth / 2,
  };
}

function drawPolyline(points, color, width = 1) {
  if (points.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.stroke();
}

function drawClosedPolyline(points, color, width = 1) {
  if (points.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.closePath();
  ctx.stroke();
}

function drawCylinderBaseCircle() {
  const params = getCylinderParams();
  if (params.radius <= 0) return;

  const points = [];
  const segments = 96;
  for (let i = 0; i < segments; i += 1) {
    const angle = (i / segments) * Math.PI * 2;
    points.push(project([
      params.centerX + params.radius * Math.cos(angle),
      params.centerY + params.radius * Math.sin(angle),
      0.9,
    ]));
  }

  ctx.save();
  drawClosedPolyline(points, "rgba(17, 98, 140, 0.82)", 1.8);
  ctx.restore();
}

function drawSemiCylinder() {
  const params = getCylinderParams();
  if (params.radius <= 0 || params.height <= 0) return;

  const segments = 40;
  const rows = 8;
  ctx.save();

  for (let row = rows - 1; row >= 0; row -= 1) {
    const z0 = (row / rows) * params.height;
    const z1 = ((row + 1) / rows) * params.height;
    for (let i = 0; i < segments; i += 1) {
      const a0 = (i / segments) * Math.PI;
      const a1 = ((i + 1) / segments) * Math.PI;
      const mid = (a0 + a1) / 2;
      const shade = 0.54 + 0.36 * Math.cos(mid - Math.PI / 2);
      const patch = [
        project(semiCylinderPoint(params, a0, z0)),
        project(semiCylinderPoint(params, a1, z0)),
        project(semiCylinderPoint(params, a1, z1)),
        project(semiCylinderPoint(params, a0, z1)),
      ];
      polygonPath(patch);
      ctx.fillStyle = `rgba(${Math.round(95 + 80 * shade)}, ${Math.round(145 + 70 * shade)}, 205, 0.42)`;
      ctx.fill();
    }
  }

  const bottomArc = [];
  const topArc = [];
  for (let i = 0; i <= segments; i += 1) {
    const angle = (i / segments) * Math.PI;
    bottomArc.push(project(semiCylinderPoint(params, angle, 0)));
    topArc.push(project(semiCylinderPoint(params, angle, params.height)));
  }

  drawPolyline(bottomArc, "rgba(30, 96, 145, 0.88)", 2);
  drawPolyline(topArc, "rgba(30, 96, 145, 0.88)", 2);
  drawSegment(project(semiCylinderPoint(params, 0, 0)), project(semiCylinderPoint(params, 0, params.height)), "rgba(30, 96, 145, 0.78)", 2);
  drawSegment(project(semiCylinderPoint(params, Math.PI, 0)), project(semiCylinderPoint(params, Math.PI, params.height)), "rgba(30, 96, 145, 0.78)", 2);
  drawSegment(project(semiCylinderPoint(params, 0, 0)), project(semiCylinderPoint(params, Math.PI, 0)), "rgba(30, 96, 145, 0.38)", 1.2);

  const center = project([params.centerX, params.centerY, 0.8]);
  ctx.fillStyle = "rgba(21, 67, 104, 0.9)";
  ctx.beginPath();
  ctx.arc(center.x, center.y, 3, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function drawProjectedImage(image, topLeft, topRight, bottomRight, bottomLeft) {
  const sourceWidth = image.naturalWidth;
  const sourceHeight = image.naturalHeight;
  if (sourceWidth <= 0 || sourceHeight <= 0) return;

  const tl = project(topLeft);
  const tr = project(topRight);
  const br = project(bottomRight);
  const bl = project(bottomLeft);

  ctx.save();
  polygonPath([tl, tr, br, bl]);
  ctx.clip();
  ctx.transform(
    (tr.x - tl.x) / sourceWidth,
    (tr.y - tl.y) / sourceWidth,
    (bl.x - tl.x) / sourceHeight,
    (bl.y - tl.y) / sourceHeight,
    tl.x,
    tl.y,
  );
  ctx.drawImage(image, 0, 0);
  ctx.restore();

  ctx.save();
  polygonPath([tl, tr, br, bl]);
  ctx.strokeStyle = "rgba(34, 47, 62, 0.72)";
  ctx.lineWidth = 1.4;
  ctx.stroke();
  ctx.restore();
}

function drawImagePatch(image, source, topLeft, topRight, bottomRight, bottomLeft) {
  if (source.width <= 0 || source.height <= 0) return;

  const tl = project(topLeft);
  const tr = project(topRight);
  const br = project(bottomRight);
  const bl = project(bottomLeft);

  ctx.save();
  polygonPath([tl, tr, br, bl]);
  ctx.clip();
  ctx.transform(
    (tr.x - tl.x) / source.width,
    (tr.y - tl.y) / source.width,
    (bl.x - tl.x) / source.height,
    (bl.y - tl.y) / source.height,
    tl.x,
    tl.y,
  );
  ctx.drawImage(image, source.x, source.y, source.width, source.height, 0, 0, source.width, source.height);
  ctx.restore();
}

function drawImagePlane() {
  const image = controls.imagePreview;
  if (!image.complete || image.naturalWidth <= 0 || image.naturalHeight <= 0) return;

  const params = getCylinderParams();
  const imagePlane = getImagePlaneMetrics(params, image);

  drawProjectedImage(
    image,
    [imagePlane.leftX, imagePlane.y, imagePlane.topZ],
    [imagePlane.rightX, imagePlane.y, imagePlane.topZ],
    [imagePlane.rightX, imagePlane.y, imagePlane.bottomZ],
    [imagePlane.leftX, imagePlane.y, imagePlane.bottomZ],
  );

  drawSegment(
    project([params.centerX, params.centerY + params.radius, Math.max(0, imagePlane.bottomZ)]),
    project([params.centerX, imagePlane.y, Math.max(0, imagePlane.bottomZ)]),
    "rgba(80, 92, 105, 0.5)",
    1.1,
  );
}

function sourcePointForCylinderPoint(point, params, imagePlane, image, tanTheta) {
  const travelY = imagePlane.y - point[1];
  const sourceZ = point[2] + travelY * tanTheta;
  const u = ((point[0] - imagePlane.leftX) / imagePlane.imageWidth) * image.naturalWidth;
  const v = ((imagePlane.topZ - sourceZ) / imagePlane.imageHeight) * image.naturalHeight;
  return { u, v };
}

function drawMappedImageOnCylinder() {
  const image = controls.imagePreview;
  if (!image.complete || image.naturalWidth <= 0 || image.naturalHeight <= 0) return;

  const params = getCylinderParams();
  if (params.radius <= 0 || params.height <= 0) return;

  const imagePlane = getImagePlaneMetrics(params, image);
  const tanTheta = Math.tan(getLightThetaRadians());
  const angleSegments = 72;
  const zSegments = 36;

  ctx.save();
  for (let ai = 0; ai < angleSegments; ai += 1) {
    const a0 = (ai / angleSegments) * Math.PI;
    const a1 = ((ai + 1) / angleSegments) * Math.PI;
    const midAngle = (a0 + a1) / 2;

    for (let zi = 0; zi < zSegments; zi += 1) {
      const z0 = (zi / zSegments) * params.height;
      const z1 = ((zi + 1) / zSegments) * params.height;
      const midZ = (z0 + z1) / 2;
      const centerWorld = semiCylinderPoint(params, midAngle, midZ);
      const sourceCenter = sourcePointForCylinderPoint(centerWorld, params, imagePlane, image, tanTheta);
      if (
        sourceCenter.u < 0 ||
        sourceCenter.u > image.naturalWidth ||
        sourceCenter.v < 0 ||
        sourceCenter.v > image.naturalHeight
      ) {
        continue;
      }

      const leftWorld = semiCylinderPoint(params, a1, midZ);
      const rightWorld = semiCylinderPoint(params, a0, midZ);
      const lowerWorld = semiCylinderPoint(params, midAngle, z0);
      const upperWorld = semiCylinderPoint(params, midAngle, z1);
      const sourceLeft = sourcePointForCylinderPoint(leftWorld, params, imagePlane, image, tanTheta);
      const sourceRight = sourcePointForCylinderPoint(rightWorld, params, imagePlane, image, tanTheta);
      const sourceLower = sourcePointForCylinderPoint(lowerWorld, params, imagePlane, image, tanTheta);
      const sourceUpper = sourcePointForCylinderPoint(upperWorld, params, imagePlane, image, tanTheta);

      const sx0 = Math.max(0, Math.min(sourceLeft.u, sourceRight.u));
      const sx1 = Math.min(image.naturalWidth, Math.max(sourceLeft.u, sourceRight.u));
      const sy0 = Math.max(0, Math.min(sourceUpper.v, sourceLower.v));
      const sy1 = Math.min(image.naturalHeight, Math.max(sourceUpper.v, sourceLower.v));
      if (sx1 - sx0 < 0.5 || sy1 - sy0 < 0.5) continue;

      drawImagePatch(
        image,
        { x: sx0, y: sy0, width: sx1 - sx0, height: sy1 - sy0 },
        semiCylinderPoint(params, a1, z1),
        semiCylinderPoint(params, a0, z1),
        semiCylinderPoint(params, a0, z0),
        semiCylinderPoint(params, a1, z0),
      );
    }
  }
  ctx.restore();
}

function lerpPoint(a, b, t) {
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
  };
}

function drawPaperGrain(paper, front) {
  ctx.save();
  polygonPath(paper);
  ctx.clip();
  ctx.strokeStyle = front ? "rgba(120, 135, 150, 0.08)" : "rgba(120, 135, 150, 0.05)";
  ctx.lineWidth = 0.7;
  for (let i = 1; i < 7; i += 1) {
    const t = i / 7;
    const a = lerpPoint(paper[0], paper[3], t);
    const b = lerpPoint(paper[1], paper[2], t);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  for (let i = 1; i < 5; i += 1) {
    const t = i / 5;
    const a = lerpPoint(paper[0], paper[1], t);
    const b = lerpPoint(paper[3], paper[2], t);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  ctx.restore();
}

function draw() {
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  drawPaper();
  drawCylinderBaseCircle();
  drawImagePlane();
  drawSemiCylinder();
  drawMappedImageOnCylinder();
  requestAnimationFrame(draw);
}

function updateImagePreview() {
  const file = controls.imageFile.files && controls.imageFile.files[0];
  if (state.imageUrl) {
    URL.revokeObjectURL(state.imageUrl);
    state.imageUrl = null;
  }

  if (!file) {
    controls.imagePath.value = "未选择图片";
    controls.imagePath.value = "\u672a\u9009\u62e9\u56fe\u7247";
    controls.imagePath.title = "";
    controls.imagePreview.removeAttribute("src");
    controls.imagePreview.parentElement.classList.remove("has-image");
    return;
  }

  const selectedPath = controls.imageFile.value || "";
  const displayPath =
    file.path && !file.path.includes("fakepath")
      ? file.path
      : selectedPath && !selectedPath.includes("fakepath")
        ? selectedPath
        : file.name;
  controls.imagePath.value = displayPath;
  controls.imagePath.title = displayPath;
  state.imageUrl = URL.createObjectURL(file);
  controls.imagePreview.src = state.imageUrl;
  controls.imagePreview.parentElement.classList.add("has-image");
}

function fitImagePreview() {
  const image = controls.imagePreview;
  const preview = image.parentElement;
  if (!image.complete || image.naturalWidth <= 0 || image.naturalHeight <= 0) return;

  const boxWidth = preview.clientWidth;
  const boxHeight = preview.clientHeight;
  if (boxWidth <= 0 || boxHeight <= 0) return;

  const scale = Math.min(boxWidth / image.naturalWidth, boxHeight / image.naturalHeight, 1);
  image.style.width = `${Math.floor(image.naturalWidth * scale)}px`;
  image.style.height = `${Math.floor(image.naturalHeight * scale)}px`;
}

function beginDrag(event) {
  canvas.setPointerCapture(event.pointerId);
  state.pointerMode = event.button === 2 || event.shiftKey ? "pan" : "rotate";
  state.drag = {
    x: event.clientX,
    y: event.clientY,
    panX: state.panX,
    panY: state.panY,
    rotation: state.rotation.map((row) => row.slice()),
  };
}

function moveDrag(event) {
  if (!state.drag) return;
  const dx = event.clientX - state.drag.x;
  const dy = event.clientY - state.drag.y;
  if (state.pointerMode === "pan") {
    state.panX = state.drag.panX + dx;
    state.panY = state.drag.panY + dy;
    return;
  }

  const rotateY = rotationY(dx * 0.01);
  const rotateX = rotationX(dy * 0.01);
  state.rotation = multiplyMatrix(rotateX, multiplyMatrix(rotateY, state.drag.rotation));
}

function endDrag(event) {
  if (state.drag) {
    canvas.releasePointerCapture(event.pointerId);
  }
  state.drag = null;
}

canvas.addEventListener("pointerdown", beginDrag);
canvas.addEventListener("pointermove", moveDrag);
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
controls.imageFile.addEventListener("change", updateImagePreview);
controls.imagePreview.addEventListener("load", fitImagePreview);

canvas.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    const factor = event.deltaY > 0 ? 1.08 : 0.92;
    state.distance = Math.max(170, Math.min(1800, state.distance * factor));
  },
  { passive: false },
);

window.addEventListener("resize", () => {
  resizeCanvas();
  fitImagePreview();
});
requestAnimationFrame(draw);
