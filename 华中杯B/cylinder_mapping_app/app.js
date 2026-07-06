const THREE = window.THREE;
if (!THREE) {
  throw new Error("Three.js failed to load. Check network access to the CDN.");
}

const canvas = document.getElementById("scene");
const controls = {
  centerX: document.getElementById("centerX"),
  centerY: document.getElementById("centerY"),
  cylinderHeight: document.getElementById("cylinderHeight"),
  cylinderRadius: document.getElementById("cylinderRadius"),
  imageY: document.getElementById("imageY"),
  imageBottomZ: document.getElementById("imageBottomZ"),
  imageHeight: document.getElementById("imageHeight"),
  lightTheta: document.getElementById("lightTheta"),
  observerY: document.getElementById("observerY"),
  observerZ: document.getElementById("observerZ"),
  imageFile: document.getElementById("imageFile"),
  imagePath: document.getElementById("imagePath"),
  imagePreview: document.getElementById("imagePreview"),
  paperPreview: document.getElementById("paperPreview"),
  paperEmpty: document.getElementById("paperEmpty"),
  exportPaper: document.getElementById("exportPaper"),
  exportStatus: document.getElementById("exportStatus"),
};

const A4_WIDTH = 210;
const A4_HEIGHT = 297;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xeaf0f6);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
const orbit = {
  target: new THREE.Vector3(0, 0, 45),
  radius: 620,
  theta: -0.82,
  phi: 0.92,
  drag: null,
  mode: "rotate",
};

const root = new THREE.Group();
scene.add(root);

const state = {
  imageUrl: null,
  imageTexture: null,
  sourceCanvas: document.createElement("canvas"),
  sourceCtx: null,
  mappedTexture: null,
  paperMapTexture: null,
  paperMapCanvas: null,
  selectedFileName: "",
};

state.sourceCtx = state.sourceCanvas.getContext("2d", { willReadFrequently: true });

function setTextureColor(texture) {
  if ("colorSpace" in texture && THREE.SRGBColorSpace) {
    texture.colorSpace = THREE.SRGBColorSpace;
  } else if (THREE.sRGBEncoding) {
    texture.encoding = THREE.sRGBEncoding;
  }
}

const materials = {
  paper: new THREE.MeshBasicMaterial({ color: 0xfafcff, side: THREE.DoubleSide }),
  paperEdge: new THREE.LineBasicMaterial({ color: 0x5f6d7a, transparent: true, opacity: 0.62 }),
  grid: new THREE.LineBasicMaterial({ color: 0xbac5cf, transparent: true, opacity: 0.22 }),
  xAxis: new THREE.LineBasicMaterial({ color: 0xd64536 }),
  yAxis: new THREE.LineBasicMaterial({ color: 0x268452 }),
  zAxis: new THREE.LineBasicMaterial({ color: 0x2a5bd7 }),
  baseCircle: new THREE.LineBasicMaterial({ color: 0x11628c }),
  cylinder: new THREE.MeshBasicMaterial({
    color: 0x8fc3dc,
    transparent: true,
    opacity: 0.24,
    side: THREE.DoubleSide,
    depthWrite: false,
  }),
  imagePlaneFrame: new THREE.LineBasicMaterial({ color: 0x22303e }),
  guide: new THREE.LineBasicMaterial({ color: 0x6f7f8c, transparent: true, opacity: 0.55 }),
  measure: new THREE.LineBasicMaterial({ color: 0x7b3f00 }),
  observer: new THREE.MeshBasicMaterial({ color: 0xe24b3b }),
  observerLine: new THREE.LineBasicMaterial({ color: 0xe24b3b, transparent: true, opacity: 0.72 }),
};

function numberValue(input, fallback = 0) {
  const value = Number.parseFloat(input.value);
  return Number.isFinite(value) ? value : fallback;
}

function getCylinderParams() {
  return {
    centerX: numberValue(controls.centerX),
    centerY: numberValue(controls.centerY),
    height: Math.max(0, numberValue(controls.cylinderHeight)),
    radius: Math.max(0, numberValue(controls.cylinderRadius)),
  };
}

function paperPoint(x, y, z = 0) {
  const params = getCylinderParams();
  return new THREE.Vector3(x - params.centerX, y - params.centerY, z);
}

function getImageBottomZ() {
  return numberValue(controls.imageBottomZ);
}

function getImageHeight() {
  return Math.max(1, numberValue(controls.imageHeight, 90));
}

function getLightThetaRadians() {
  return (Math.max(-85, Math.min(85, numberValue(controls.lightTheta))) * Math.PI) / 180;
}

function getObserverPoint(imagePlane = null) {
  void imagePlane;
  return new THREE.Vector3(
    0,
    numberValue(controls.observerY, 210),
    numberValue(controls.observerZ, 55),
  );
}

function clearRoot() {
  while (root.children.length > 0) {
    const child = root.children.pop();
    child.traverse((node) => {
      if (node.geometry) node.geometry.dispose();
      if (node.material && !Object.values(materials).includes(node.material)) {
        if (Array.isArray(node.material)) node.material.forEach((material) => material.dispose());
        else node.material.dispose();
      }
    });
  }
}

function makeLine(points, material, closed = false) {
  const finalPoints = closed ? [...points, points[0]] : points;
  const geometry = new THREE.BufferGeometry().setFromPoints(finalPoints);
  return new THREE.Line(geometry, material);
}

function makeLabel(text, position, color = "#263442") {
  const labelCanvas = document.createElement("canvas");
  labelCanvas.width = 96;
  labelCanvas.height = 40;
  const ctx = labelCanvas.getContext("2d");
  ctx.font = "20px Segoe UI, sans-serif";
  ctx.fillStyle = color;
  ctx.fillText(text, 4, 25);
  const texture = new THREE.CanvasTexture(labelCanvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.position.copy(position);
  sprite.scale.set(34, 14, 1);
  return sprite;
}

function makePlaneLabel(text, position, color = "#263442", width = 24, height = 10) {
  const labelCanvas = document.createElement("canvas");
  labelCanvas.width = 128;
  labelCanvas.height = 64;
  const ctx = labelCanvas.getContext("2d");
  ctx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);
  ctx.font = "36px Segoe UI, sans-serif";
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, labelCanvas.width / 2, labelCanvas.height / 2);

  const texture = new THREE.CanvasTexture(labelCanvas);
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, height), material);
  mesh.position.copy(position);
  return mesh;
}

function addPaper() {
  const params = getCylinderParams();
  const paper = new THREE.Mesh(
    new THREE.PlaneGeometry(A4_WIDTH, A4_HEIGHT),
    materials.paper,
  );
  paper.position.set(A4_WIDTH / 2 - params.centerX, A4_HEIGHT / 2 - params.centerY, 0);
  root.add(paper);

  const corners = [
    paperPoint(0, 0, 0.2),
    paperPoint(A4_WIDTH, 0, 0.2),
    paperPoint(A4_WIDTH, A4_HEIGHT, 0.2),
    paperPoint(0, A4_HEIGHT, 0.2),
  ];
  root.add(makeLine(corners, materials.paperEdge, true));

  const gridPoints = [];
  for (let x = 30; x < A4_WIDTH; x += 30) {
    gridPoints.push(paperPoint(x, 0, 0.35), paperPoint(x, A4_HEIGHT, 0.35));
  }
  for (let y = 30; y < A4_HEIGHT; y += 30) {
    gridPoints.push(paperPoint(0, y, 0.35), paperPoint(A4_WIDTH, y, 0.35));
  }
  const gridGeometry = new THREE.BufferGeometry().setFromPoints(gridPoints);
  root.add(new THREE.LineSegments(gridGeometry, materials.grid));
}

function addAxes() {
  const params = getCylinderParams();
  const zHeight = Math.max(params.height, getImageBottomZ() + getImageHeight(), numberValue(controls.observerZ, 55), 30) * 1.08 + 12;
  const xEnd = Math.max(A4_WIDTH - params.centerX, params.radius + 25, 30);
  const yEnd = Math.max(A4_HEIGHT - params.centerY, params.radius + 25, 30);
  root.add(makeLine([new THREE.Vector3(0, 0, 1), new THREE.Vector3(xEnd, 0, 1)], materials.xAxis));
  root.add(makeLine([new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, yEnd, 1)], materials.yAxis));
  root.add(makeLine([new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, zHeight)], materials.zAxis));
  root.add(makeLabel("x", new THREE.Vector3(xEnd + 8, 0, 5), "#d64536"));
  root.add(makeLabel("y", new THREE.Vector3(0, yEnd + 8, 5), "#268452"));
  root.add(makeLabel("z", new THREE.Vector3(0, 0, zHeight + 8), "#2a5bd7"));
  root.add(makeLabel("O", new THREE.Vector3(8, 7, 2), "#263442"));
}

function cylinderPoint(params, angle, z, radiusOffset = 0) {
  const radius = params.radius + radiusOffset;
  return new THREE.Vector3(
    radius * Math.cos(angle),
    radius * Math.sin(angle),
    z,
  );
}

function makeSemiCylinderGeometry(params, angleSegments = 72, zSegments = 24, radiusOffset = 0) {
  const positions = [];
  const uvs = [];
  const indices = [];

  for (let zi = 0; zi <= zSegments; zi += 1) {
    const z = (zi / zSegments) * params.height;
    for (let ai = 0; ai <= angleSegments; ai += 1) {
      const angle = (ai / angleSegments) * Math.PI;
      const point = cylinderPoint(params, angle, z, radiusOffset);
      positions.push(point.x, point.y, point.z);
      uvs.push(ai / angleSegments, zi / zSegments);
    }
  }

  const stride = angleSegments + 1;
  for (let zi = 0; zi < zSegments; zi += 1) {
    for (let ai = 0; ai < angleSegments; ai += 1) {
      const a = zi * stride + ai;
      const b = a + 1;
      const c = a + stride;
      const d = c + 1;
      indices.push(a, b, d, a, d, c);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function addBaseCircle() {
  const params = getCylinderParams();
  if (params.radius <= 0) return;

  const points = [];
  for (let i = 0; i < 128; i += 1) {
    const angle = (i / 128) * Math.PI * 2;
    points.push(new THREE.Vector3(
      params.radius * Math.cos(angle),
      params.radius * Math.sin(angle),
      0.8,
    ));
  }
  root.add(makeLine(points, materials.baseCircle, true));
}

function addSemiCylinder() {
  const params = getCylinderParams();
  if (params.radius <= 0 || params.height <= 0) return;

  const mesh = new THREE.Mesh(makeSemiCylinderGeometry(params), materials.cylinder);
  root.add(mesh);

  const bottom = [];
  const top = [];
  for (let i = 0; i <= 72; i += 1) {
    const angle = (i / 72) * Math.PI;
    bottom.push(cylinderPoint(params, angle, 0));
    top.push(cylinderPoint(params, angle, params.height));
  }
  root.add(makeLine(bottom, materials.baseCircle));
  root.add(makeLine(top, materials.baseCircle));
  root.add(makeLine([cylinderPoint(params, 0, 0), cylinderPoint(params, 0, params.height)], materials.baseCircle));
  root.add(makeLine([cylinderPoint(params, Math.PI, 0), cylinderPoint(params, Math.PI, params.height)], materials.baseCircle));
}

function addMeasurementAnnotations() {
  const params = getCylinderParams();
  if (params.radius <= 0) return;

  const radiusAngle = Math.PI / 4;
  const radiusStart = new THREE.Vector3(0, 0, 1.2);
  const radiusEnd = cylinderPoint(params, radiusAngle, 1.2);
  const labelPosition = radiusEnd.clone().multiplyScalar(0.5).add(new THREE.Vector3(0, 0, 0.15));
  root.add(makeLine([radiusStart, radiusEnd], materials.measure));
  root.add(makePlaneLabel("R", labelPosition, "#7b3f00", 18, 10));

  const imagePlane = getImagePlaneMetrics(params);
  if (!imagePlane || params.height <= 0) return;

  const sourcePoint = new THREE.Vector3(0, imagePlane.y, imagePlane.topZ);
  const rayAngle = Math.PI / 2;
  const zHit = sourcePoint.z - (imagePlane.y - params.radius) * Math.tan(getLightThetaRadians());
  if (zHit < 0 || zHit > params.height) return;

  const hitPoint = cylinderPoint(params, rayAngle, zHit);
  const horizontalEnd = new THREE.Vector3(hitPoint.x, hitPoint.y, sourcePoint.z);
  root.add(makeLine([sourcePoint, hitPoint], materials.measure));
  root.add(makeLine([sourcePoint, horizontalEnd], materials.measure));

  const reflected = reflectedDirectionForCylinderAngle(rayAngle, getLightThetaRadians());
  if (reflected && reflected.z < -0.001) {
    const paperHit = hitPoint.clone().add(reflected.multiplyScalar(-hitPoint.z / reflected.z));
    root.add(makeLine([hitPoint, paperHit], materials.measure));
  }

  const rayLength = sourcePoint.distanceTo(hitPoint);
  const actualTheta = Math.atan2(Math.abs(sourcePoint.z - hitPoint.z), Math.max(0.001, sourcePoint.distanceTo(horizontalEnd)));
  const arcRadius = Math.min(20, Math.max(8, rayLength * 0.22));
  const arcPoints = [];
  const segments = 24;
  for (let i = 0; i <= segments; i += 1) {
    const a = (i / segments) * actualTheta;
    arcPoints.push(sourcePoint.clone().add(new THREE.Vector3(0, -arcRadius * Math.cos(a), -arcRadius * Math.sin(a))));
  }
  root.add(makeLine(arcPoints, materials.measure));

  const labelAngle = actualTheta / 2;
  const labelPositionTheta = sourcePoint.clone().add(new THREE.Vector3(
    0,
    -(arcRadius + 10) * Math.cos(labelAngle),
    -(arcRadius + 10) * Math.sin(labelAngle),
  ));
  root.add(makeLabel("\u03b8", labelPositionTheta, "#7b3f00"));
}

function getImagePlaneMetrics(params) {
  if (!controls.imagePreview.naturalWidth || !controls.imagePreview.naturalHeight) return null;
  const imageHeight = getImageHeight();
  const imageWidth = imageHeight * (controls.imagePreview.naturalWidth / controls.imagePreview.naturalHeight);
  const bottomZ = getImageBottomZ();
  const y = numberValue(controls.imageY, 210);
  return {
    imageHeight,
    imageWidth,
    bottomZ,
    topZ: bottomZ + imageHeight,
    y,
    leftX: -imageWidth / 2,
    rightX: imageWidth / 2,
  };
}

function addImagePlane() {
  if (!state.imageTexture) return;
  const params = getCylinderParams();
  const imagePlane = getImagePlaneMetrics(params);
  if (!imagePlane) return;

  const geometry = new THREE.PlaneGeometry(imagePlane.imageWidth, imagePlane.imageHeight);
  const material = new THREE.MeshBasicMaterial({
    map: state.imageTexture,
    side: THREE.DoubleSide,
    transparent: true,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(0, imagePlane.y, imagePlane.bottomZ + imagePlane.imageHeight / 2);
  mesh.rotation.x = Math.PI / 2;
  root.add(mesh);

  const corners = [
    new THREE.Vector3(imagePlane.leftX, imagePlane.y, imagePlane.topZ),
    new THREE.Vector3(imagePlane.rightX, imagePlane.y, imagePlane.topZ),
    new THREE.Vector3(imagePlane.rightX, imagePlane.y, imagePlane.bottomZ),
    new THREE.Vector3(imagePlane.leftX, imagePlane.y, imagePlane.bottomZ),
  ];
  root.add(makeLine(corners, materials.imagePlaneFrame, true));
  root.add(makeLine([
    new THREE.Vector3(0, params.radius, Math.max(0, imagePlane.bottomZ)),
    new THREE.Vector3(0, imagePlane.y, Math.max(0, imagePlane.bottomZ)),
  ], materials.guide));
}

function getCylinderRayHits(origin, direction, params) {
  const a = direction.x * direction.x + direction.y * direction.y;
  const b = 2 * (origin.x * direction.x + origin.y * direction.y);
  const c = origin.x * origin.x + origin.y * origin.y - params.radius * params.radius;
  const candidates = [];

  if (a > 0.000001) {
    const discriminant = b * b - 4 * a * c;
    if (discriminant >= 0) {
      const root = Math.sqrt(discriminant);
      candidates.push((-b - root) / (2 * a), (-b + root) / (2 * a));
    }
  }

  return candidates
    .filter((t) => t >= 0)
    .map((t) => ({
      t,
      point: origin.clone().add(direction.clone().multiplyScalar(t)),
    }))
    .filter(({ point }) => point.y >= -0.001 && point.z >= 0 && point.z <= params.height)
    .sort((left, right) => left.t - right.t);
}

function isFirstCylinderHit(observer, point, params) {
  const direction = point.clone().sub(observer);
  const hits = getCylinderRayHits(observer, direction, params);
  if (!hits.length) return false;

  return !hits.some(({ t }) => t < 0.995);
}

function extendObserverRay(observer, corner, params) {
  const direction = corner.clone().sub(observer);
  const cylinderHit = getCylinderRayHits(observer, direction, params)
    .filter(({ t }) => t >= 1)
    .map(({ point }) => point)[0];
  if (cylinderHit) return cylinderHit;

  if (Math.abs(direction.y) > 0.000001) {
    const t = -observer.y / direction.y;
    if (t >= 1) return observer.clone().add(direction.multiplyScalar(t));
  }

  return corner;
}

function addObserverPoint() {
  const params = getCylinderParams();
  const imagePlane = getImagePlaneMetrics(params);
  if (!imagePlane) return;

  const observer = getObserverPoint(imagePlane);
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(3.5, 24, 16), materials.observer);
  sphere.position.copy(observer);
  root.add(sphere);
  root.add(makeLabel("Eye", observer.clone().add(new THREE.Vector3(4, 4, 5)), "#b93428"));

  const imageCorners = [
    new THREE.Vector3(imagePlane.leftX, imagePlane.y, imagePlane.topZ),
    new THREE.Vector3(imagePlane.rightX, imagePlane.y, imagePlane.topZ),
    new THREE.Vector3(imagePlane.rightX, imagePlane.y, imagePlane.bottomZ),
    new THREE.Vector3(imagePlane.leftX, imagePlane.y, imagePlane.bottomZ),
  ];
  imageCorners.forEach((corner) => {
    root.add(makeLine([observer, extendObserverRay(observer, corner, params)], materials.observerLine));
  });
}

function sourcePointForCylinderPoint(point, params, imagePlane) {
  const observer = getObserverPoint(imagePlane);
  if (!isFirstCylinderHit(observer, point, params)) return null;

  const direction = point.clone().sub(observer);
  if (Math.abs(direction.y) < 0.000001) return null;

  const t = (imagePlane.y - observer.y) / direction.y;
  if (t < 0 || t > 1) return null;

  const imagePoint = observer.clone().add(direction.multiplyScalar(t));
  return {
    u: ((imagePoint.x - imagePlane.leftX) / imagePlane.imageWidth) * controls.imagePreview.naturalWidth,
    v: ((imagePlane.topZ - imagePoint.z) / imagePlane.imageHeight) * controls.imagePreview.naturalHeight,
  };
}

function reflectedDirectionForCylinderAngle(angle, theta) {
  const downwardTheta = Math.abs(theta);
  if (downwardTheta < 0.001) return null;

  const incident = new THREE.Vector3(0, -Math.cos(downwardTheta), -Math.sin(downwardTheta));
  const normal = new THREE.Vector3(Math.cos(angle), Math.sin(angle), 0);
  return incident.sub(normal.multiplyScalar(2 * incident.dot(normal)));
}

function cylinderPointToA4(point, params, angle, theta) {
  const reflected = reflectedDirectionForCylinderAngle(angle, theta);
  if (!reflected || reflected.z >= -0.001) return null;

  const travel = -point.z / reflected.z;
  if (travel < 0) return null;

  const paperHit = point.clone().add(reflected.multiplyScalar(travel));
  return {
    x: params.centerX + paperHit.x,
    y: params.centerY + paperHit.y,
  };
}

function sampleSourcePixel(source, sourcePoint) {
  if (!sourcePoint) return null;
  const sx = Math.round(sourcePoint.u);
  const sy = Math.round(sourcePoint.v);
  if (sx < 0 || sx >= source.width || sy < 0 || sy >= source.height) return null;

  const index = (sy * source.width + sx) * 4;
  return [
    source.data[index],
    source.data[index + 1],
    source.data[index + 2],
    source.data[index + 3],
  ];
}

function splatPixel(imageData, width, height, x, y, color) {
  const px = Math.round(x);
  const py = Math.round(y);
  for (let oy = -1; oy <= 1; oy += 1) {
    for (let ox = -1; ox <= 1; ox += 1) {
      const tx = px + ox;
      const ty = py + oy;
      if (tx < 0 || tx >= width || ty < 0 || ty >= height) continue;

      const index = (ty * width + tx) * 4;
      imageData.data[index] = color[0];
      imageData.data[index + 1] = color[1];
      imageData.data[index + 2] = color[2];
      imageData.data[index + 3] = color[3];
    }
  }
}

function paperToCanvasPoint(x, y, width, height) {
  return {
    x: (x / A4_WIDTH) * width,
    y: (1 - y / A4_HEIGHT) * height,
  };
}

function drawPaperMapAnnotations(ctx, params, theta) {
  const { width, height } = ctx.canvas;
  const center = paperToCanvasPoint(params.centerX, params.centerY, width, height);
  const scale = width / A4_WIDTH;
  const radius = params.radius * scale;
  const thetaDeg = Math.abs((theta * 180) / Math.PI);
  const imageBottomZ = getImageBottomZ();

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(20, 55, 80, 0.92)";
  ctx.fillStyle = "rgba(20, 55, 80, 0.92)";
  ctx.lineWidth = 3;

  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(center.x - 8, center.y);
  ctx.lineTo(center.x + 8, center.y);
  ctx.moveTo(center.x, center.y - 8);
  ctx.lineTo(center.x, center.y + 8);
  ctx.stroke();

  const radiusEnd = paperToCanvasPoint(params.centerX + params.radius, params.centerY, width, height);
  ctx.beginPath();
  ctx.moveTo(center.x, center.y);
  ctx.lineTo(radiusEnd.x, radiusEnd.y);
  ctx.stroke();

  ctx.font = "22px Microsoft YaHei, Segoe UI, sans-serif";
  ctx.fillText("R", (center.x + radiusEnd.x) / 2 + 8, center.y - 10);

  const lines = [
    "A4 mapping",
    `R = ${params.radius.toFixed(2)} mm`,
    `theta = ${thetaDeg.toFixed(2)} deg`,
    `center = (${params.centerX.toFixed(2)}, ${params.centerY.toFixed(2)}) mm`,
    `height = ${params.height.toFixed(2)} mm`,
    `image z = ${imageBottomZ.toFixed(2)} mm`,
  ];
  const boxX = 18;
  const boxY = 18;
  const lineHeight = 28;
  const boxWidth = 360;
  const boxHeight = lines.length * lineHeight + 22;

  ctx.fillStyle = "rgba(255, 255, 255, 0.86)";
  ctx.strokeStyle = "rgba(20, 55, 80, 0.35)";
  ctx.lineWidth = 2;
  ctx.fillRect(boxX, boxY, boxWidth, boxHeight);
  ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);

  ctx.fillStyle = "rgba(20, 55, 80, 0.96)";
  ctx.font = "20px Microsoft YaHei, Segoe UI, sans-serif";
  lines.forEach((line, index) => {
    ctx.fillText(line, boxX + 14, boxY + 30 + index * lineHeight);
  });
  ctx.restore();
}

function clearPaperPreview(message = "\u7b49\u5f85\u751f\u6210\u7eb8\u9762\u56fe\u50cf") {
  const previewCanvas = controls.paperPreview;
  const previewCtx = previewCanvas.getContext("2d");
  previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewCanvas.parentElement.classList.remove("has-image");
  controls.paperEmpty.textContent = message;
  controls.exportPaper.disabled = true;
  state.paperMapCanvas = null;
}

function updatePaperPreview(paperCanvas) {
  const previewCanvas = controls.paperPreview;
  const previewCtx = previewCanvas.getContext("2d");
  previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewCtx.drawImage(paperCanvas, 0, 0, previewCanvas.width, previewCanvas.height);
  previewCanvas.parentElement.classList.add("has-image");
  controls.exportPaper.disabled = false;
}

function rebuildMappedTexture() {
  if (!state.sourceCanvas.width || !state.sourceCanvas.height) return null;
  const params = getCylinderParams();
  if (params.radius <= 0 || params.height <= 0) return null;
  const imagePlane = getImagePlaneMetrics(params);
  if (!imagePlane) return null;

  const width = 512;
  const height = 512;
  const mappedCanvas = document.createElement("canvas");
  mappedCanvas.width = width;
  mappedCanvas.height = height;
  const mappedCtx = mappedCanvas.getContext("2d");
  const output = mappedCtx.createImageData(width, height);
  const source = state.sourceCtx.getImageData(0, 0, state.sourceCanvas.width, state.sourceCanvas.height);

  for (let y = 0; y < height; y += 1) {
    const z = (1 - y / (height - 1)) * params.height;
    for (let x = 0; x < width; x += 1) {
      const angle = (x / (width - 1)) * Math.PI;
      const point = cylinderPoint(params, angle, z);
      const sourcePoint = sourcePointForCylinderPoint(point, params, imagePlane);
      const outIndex = (y * width + x) * 4;
      const color = sampleSourcePixel(source, sourcePoint);
      if (!color) {
        output.data[outIndex + 3] = 0;
        continue;
      }
      output.data[outIndex] = color[0];
      output.data[outIndex + 1] = color[1];
      output.data[outIndex + 2] = color[2];
      output.data[outIndex + 3] = color[3];
    }
  }

  mappedCtx.putImageData(output, 0, 0);
  const texture = new THREE.CanvasTexture(mappedCanvas);
  setTextureColor(texture);
  texture.needsUpdate = true;
  return texture;
}

function addMappedCylinder() {
  if (!state.imageTexture) return;
  const params = getCylinderParams();
  if (params.radius <= 0 || params.height <= 0) return;

  const texture = rebuildMappedTexture();
  if (!texture) return;
  if (state.mappedTexture) state.mappedTexture.dispose();
  state.mappedTexture = texture;

  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(makeSemiCylinderGeometry(params, 128, 48, 0.28), material);
  root.add(mesh);
}

function buildPaperMapCanvas() {
  if (!state.sourceCanvas.width || !state.sourceCanvas.height) return null;
  const params = getCylinderParams();
  if (params.radius <= 0 || params.height <= 0) return null;
  const imagePlane = getImagePlaneMetrics(params);
  if (!imagePlane) return null;

  const theta = Math.abs(getLightThetaRadians());
  if (theta < 0.001) return null;

  const width = 840;
  const height = 1188;
  const paperCanvas = document.createElement("canvas");
  paperCanvas.width = width;
  paperCanvas.height = height;
  const paperCtx = paperCanvas.getContext("2d");
  const output = paperCtx.createImageData(width, height);
  const source = state.sourceCtx.getImageData(0, 0, state.sourceCanvas.width, state.sourceCanvas.height);
  const phiSamples = 640;
  const paperPhiRange = Math.PI;
  const zSamples = 640;

  for (let zi = 0; zi < zSamples; zi += 1) {
    const z = (zi / (zSamples - 1)) * params.height;
    for (let pi = 0; pi < phiSamples; pi += 1) {
      const phi = (pi / (phiSamples - 1)) * paperPhiRange;
      const point = cylinderPoint(params, phi, z);
      const sourcePoint = sourcePointForCylinderPoint(point, params, imagePlane);
      const color = sampleSourcePixel(source, sourcePoint);
      if (!color) continue;

      const a4Point = cylinderPointToA4(point, params, phi, theta);
      if (!a4Point) continue;
      if (a4Point.x < 0 || a4Point.x > A4_WIDTH || a4Point.y < 0 || a4Point.y > A4_HEIGHT) continue;

      const tx = (a4Point.x / A4_WIDTH) * (width - 1);
      const ty = (1 - a4Point.y / A4_HEIGHT) * (height - 1);
      splatPixel(output, width, height, tx, ty, color);
    }
  }

  paperCtx.putImageData(output, 0, 0);
  paperCtx.save();
  paperCtx.globalCompositeOperation = "destination-over";
  paperCtx.fillStyle = "#ffffff";
  paperCtx.fillRect(0, 0, width, height);
  paperCtx.restore();
  drawPaperMapAnnotations(paperCtx, params, theta);
  return paperCanvas;
}

function rebuildPaperMapTexture() {
  const paperCanvas = buildPaperMapCanvas();
  if (!paperCanvas) {
    clearPaperPreview("\u53c2\u6570\u4e0d\u8db3\uff0c\u65e0\u6cd5\u751f\u6210\u7eb8\u9762\u56fe\u50cf");
    return null;
  }

  state.paperMapCanvas = paperCanvas;
  updatePaperPreview(paperCanvas);

  const texture = new THREE.CanvasTexture(paperCanvas);
  setTextureColor(texture);
  texture.needsUpdate = true;
  return texture;
}

function addMappedPaper() {
  if (!state.imageTexture) {
    clearPaperPreview();
    return;
  }
  const params = getCylinderParams();
  const texture = rebuildPaperMapTexture();
  if (!texture) return;

  if (state.paperMapTexture) state.paperMapTexture.dispose();
  state.paperMapTexture = texture;

  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(A4_WIDTH, A4_HEIGHT), material);
  mesh.position.set(A4_WIDTH / 2 - params.centerX, A4_HEIGHT / 2 - params.centerY, 1.05);
  root.add(mesh);
}

function rebuildScene() {
  clearRoot();
  addPaper();
  addMappedPaper();
  addAxes();
  addBaseCircle();
  addImagePlane();
  addObserverPoint();
  addSemiCylinder();
  addMappedCylinder();
  addMeasurementAnnotations();
}

function updateCamera() {
  const sinPhi = Math.sin(orbit.phi);
  camera.position.set(
    orbit.target.x + orbit.radius * sinPhi * Math.cos(orbit.theta),
    orbit.target.y + orbit.radius * sinPhi * Math.sin(orbit.theta),
    orbit.target.z + orbit.radius * Math.cos(orbit.phi),
  );
  camera.up.set(0, 0, 1);
  camera.lookAt(orbit.target);
}

function resizeRenderer() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = Math.max(1, width) / Math.max(1, height);
  camera.updateProjectionMatrix();
}

function animate() {
  resizeRenderer();
  updateCamera();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

function updateImagePreview() {
  const file = controls.imageFile.files && controls.imageFile.files[0];
  if (state.imageUrl) {
    URL.revokeObjectURL(state.imageUrl);
    state.imageUrl = null;
  }
  if (state.imageTexture) {
    state.imageTexture.dispose();
    state.imageTexture = null;
  }

  if (!file) {
    state.selectedFileName = "";
    controls.imagePath.value = "\u672a\u9009\u62e9\u56fe\u7247";
    controls.imagePath.title = "";
    controls.exportStatus.textContent = "";
    controls.imagePreview.removeAttribute("src");
    controls.imagePreview.parentElement.classList.remove("has-image");
    rebuildScene();
    return;
  }

  state.selectedFileName = file.name;
  controls.exportStatus.textContent = "";
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
  const scale = Math.min(preview.clientWidth / image.naturalWidth, preview.clientHeight / image.naturalHeight, 1);
  image.style.width = `${Math.floor(image.naturalWidth * scale)}px`;
  image.style.height = `${Math.floor(image.naturalHeight * scale)}px`;
}

function updateTexturesFromImage() {
  const image = controls.imagePreview;
  if (!image.complete || image.naturalWidth <= 0 || image.naturalHeight <= 0) return;

  state.sourceCanvas.width = image.naturalWidth;
  state.sourceCanvas.height = image.naturalHeight;
  state.sourceCtx.drawImage(image, 0, 0);

  state.imageTexture = new THREE.Texture(image);
  setTextureColor(state.imageTexture);
  state.imageTexture.needsUpdate = true;
  fitImagePreview();
  rebuildScene();
}

function getExportFileName() {
  const sourceName = state.selectedFileName || "paper_mapping";
  const dotIndex = sourceName.lastIndexOf(".");
  const baseName = dotIndex > 0 ? sourceName.slice(0, dotIndex) : sourceName;
  return `${baseName}_a4_mapping.png`;
}

function canvasToBlob(targetCanvas) {
  return new Promise((resolve) => {
    targetCanvas.toBlob((blob) => resolve(blob), "image/png");
  });
}

function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function exportPaperMap() {
  if (!state.paperMapCanvas) {
    const paperCanvas = buildPaperMapCanvas();
    if (!paperCanvas) {
      controls.exportStatus.textContent = "\u8bf7\u5148\u9009\u62e9\u56fe\u7247\u5e76\u8bbe\u7f6e\u6709\u6548\u53c2\u6570";
      return;
    }
    state.paperMapCanvas = paperCanvas;
    updatePaperPreview(paperCanvas);
  }

  const blob = await canvasToBlob(state.paperMapCanvas);
  if (!blob) {
    controls.exportStatus.textContent = "\u5bfc\u51fa\u5931\u8d25\uff1a\u65e0\u6cd5\u751f\u6210 PNG";
    return;
  }

  const fileName = getExportFileName();
  if (window.showSaveFilePicker) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: fileName,
        types: [{
          description: "PNG image",
          accept: { "image/png": [".png"] },
        }],
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      controls.exportStatus.textContent = `\u5df2\u5bfc\u51fa\uff1a${fileName}`;
      return;
    } catch (error) {
      if (error && error.name === "AbortError") {
        controls.exportStatus.textContent = "\u5df2\u53d6\u6d88\u5bfc\u51fa";
        return;
      }
    }
  }

  downloadBlob(blob, fileName);
  controls.exportStatus.textContent =
    `\u5df2\u4e0b\u8f7d\uff1a${fileName}\uff1b\u6d4f\u89c8\u5668\u4e0d\u5141\u8bb8\u7f51\u9875\u81ea\u52a8\u5199\u5165\u539f\u56fe\u540c\u76ee\u5f55`;
}

function beginDrag(event) {
  canvas.setPointerCapture(event.pointerId);
  orbit.mode = event.button === 2 || event.shiftKey ? "pan" : "rotate";
  orbit.drag = {
    x: event.clientX,
    y: event.clientY,
    theta: orbit.theta,
    phi: orbit.phi,
    target: orbit.target.clone(),
  };
}

function moveDrag(event) {
  if (!orbit.drag) return;
  const dx = event.clientX - orbit.drag.x;
  const dy = event.clientY - orbit.drag.y;

  if (orbit.mode === "pan") {
    const panScale = orbit.radius / 720;
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);
    const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
    const up = new THREE.Vector3().copy(camera.up).normalize();
    orbit.target.copy(orbit.drag.target)
      .addScaledVector(right, -dx * panScale)
      .addScaledVector(up, dy * panScale);
    return;
  }

  orbit.theta = orbit.drag.theta - dx * 0.008;
  orbit.phi = Math.max(0.08, Math.min(Math.PI - 0.08, orbit.drag.phi + dy * 0.008));
}

function endDrag(event) {
  if (orbit.drag) canvas.releasePointerCapture(event.pointerId);
  orbit.drag = null;
}

function scheduleRebuild() {
  rebuildScene();
}

Object.values(controls).forEach((control) => {
  if (control && control.tagName === "INPUT" && control.type !== "file") {
    control.addEventListener("input", scheduleRebuild);
  }
});

controls.imageFile.addEventListener("change", updateImagePreview);
controls.imagePreview.addEventListener("load", updateTexturesFromImage);
controls.exportPaper.addEventListener("click", exportPaperMap);
canvas.addEventListener("pointerdown", beginDrag);
canvas.addEventListener("pointermove", moveDrag);
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    orbit.radius = Math.max(120, Math.min(2200, orbit.radius * (event.deltaY > 0 ? 1.08 : 0.92)));
  },
  { passive: false },
);

window.addEventListener("resize", () => {
  resizeRenderer();
  fitImagePreview();
});

rebuildScene();
requestAnimationFrame(animate);
