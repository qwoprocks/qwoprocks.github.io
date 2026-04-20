(function () {
  const container = document.querySelector('.maze-container');
  if (!container) return;

  // --- State ---
  let gridSize = 50;
  let cells = [];
  let running = false;
  let stepCount = 0;
  let totalSteps = 0;
  let stepTimer = null;
  let drawing = false;
  let drawMode = true; // true = place alive, false = remove

  // --- DOM Setup ---
  const controls = document.createElement('div');
  controls.className = 'maze-controls';

  const btnSeed = document.createElement('button');
  btnSeed.textContent = 'Seed';
  btnSeed.className = 'maze-btn';

  const btnRun = document.createElement('button');
  btnRun.textContent = 'Run';
  btnRun.className = 'maze-btn maze-btn-accent';

  const btnReset = document.createElement('button');
  btnReset.textContent = 'Reset';
  btnReset.className = 'maze-btn';

  const sizeGroup = document.createElement('div');
  sizeGroup.className = 'maze-size-group';

  const sizes = [20, 50, 100];
  const sizeBtns = sizes.map((s) => {
    const btn = document.createElement('button');
    btn.textContent = s + '\u00d7' + s;
    btn.className = 'maze-btn maze-size-btn' + (s === gridSize ? ' active' : '');
    btn.addEventListener('click', () => {
      if (running) return;
      gridSize = s;
      sizeBtns.forEach((b, i) => b.classList.toggle('active', sizes[i] === s));
      resetGrid();
    });
    return btn;
  });
  sizeBtns.forEach((b) => sizeGroup.appendChild(b));

  const stepInfo = document.createElement('span');
  stepInfo.className = 'maze-step-info';
  stepInfo.textContent = '';

  controls.appendChild(sizeGroup);
  controls.appendChild(stepInfo);
  controls.appendChild(btnSeed);
  controls.appendChild(btnRun);
  controls.appendChild(btnReset);

  const canvas = document.createElement('canvas');
  canvas.className = 'maze-canvas';

  container.appendChild(controls);
  container.appendChild(canvas);

  const ctx = canvas.getContext('2d');

  // --- Style injection ---
  const style = document.createElement('style');
  style.textContent = `
    .maze-container {
      width: 100%;
      max-width: 600px;
    }
    .maze-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      align-items: center;
      margin-bottom: 0.75rem;
    }
    .maze-btn {
      font-family: var(--font-mono, monospace);
      font-size: 0.55rem;
      letter-spacing: 0.06em;
      padding: 0.3rem 0.7rem;
      border: 1px solid var(--border, #2a2a2a);
      border-radius: 999px;
      background: transparent;
      color: var(--text-muted, #5a5550);
      cursor: pointer;
      transition: border-color 0.3s, color 0.3s, background 0.3s;
      user-select: none;
      white-space: nowrap;
    }
    .maze-btn:hover {
      border-color: var(--border-light, #3a3a3a);
      color: var(--text, #c8c4c0);
    }
    .maze-btn:disabled {
      opacity: 0.4;
      cursor: default;
    }
    .maze-btn-accent {
      border-color: var(--accent, #ff4040);
      color: var(--accent, #ff4040);
    }
    .maze-btn-accent:hover {
      background: var(--accent-dim, rgba(255,64,64,0.07));
    }
    .maze-size-group {
      display: flex;
      gap: 0.25rem;
      margin-left: 0.5rem;
    }
    .maze-size-btn.active {
      border-color: var(--accent, #ff4040);
      color: var(--accent, #ff4040);
      background: var(--accent-dim, rgba(255,64,64,0.07));
    }
    .maze-step-info {
      font-family: var(--font-mono, monospace);
      font-size: 0.55rem;
      color: var(--text-muted, #5a5550);
      margin: 0 auto;
      letter-spacing: 0.04em;
    }
    .maze-canvas {
      width: 100%;
      aspect-ratio: 1 / 1;
      display: block;
      border: 1px solid var(--border, #2a2a2a);
      cursor: crosshair;
      background: var(--surface, #181818);
      image-rendering: pixelated;
    }
  `;
  document.head.appendChild(style);

  // --- Grid logic ---
  function makeGrid(size) {
    const g = [];
    for (let i = 0; i < size; i++) {
      g[i] = [];
      for (let j = 0; j < size; j++) {
        g[i][j] = false;
      }
    }
    return g;
  }

  function resetGrid() {
    stopRun();
    cells = makeGrid(gridSize);
    stepCount = 0;
    totalSteps = gridSize + gridSize - 30;
    stepInfo.textContent = '';
    draw();
    setButtonStates(false);
  }

  function outOfBounds(x, y) {
    return x < 0 || x >= cells.length || y < 0 || y >= cells[0].length;
  }

  function numAliveNeighbours(x, y) {
    let count = 0;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        if (dx === 0 && dy === 0) continue;
        const nx = x + dx;
        const ny = y + dy;
        if (!outOfBounds(nx, ny) && cells[nx][ny]) count++;
      }
    }
    return count;
  }

  function step() {
    const size = cells.length;
    const next = makeGrid(size);
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const alive = cells[i][j];
        if (alive) {
          const n = numAliveNeighbours(i, j);
          if (n === 0) {
            next[i][j] = false;
          } else if (n <= 5) {
            // Arch-breaking rule
            if (n <= 2 && !outOfBounds(i - 1, j - 1) && !outOfBounds(i + 1, j - 1)) {
              if ((cells[i - 1][j - 1] || cells[i + 1][j - 1]) && !cells[i][j - 1]) {
                next[i][j] = false;
                continue;
              }
            }
            next[i][j] = true;
            // Kill diagonals when exactly 4 neighbours
            if (n === 4) {
              if (!outOfBounds(i - 1, j - 1)) next[i - 1][j - 1] = false;
              if (!outOfBounds(i + 1, j + 1)) next[i + 1][j + 1] = false;
            }
          } else {
            next[i][j] = false;
          }
        } else {
          if (numAliveNeighbours(i, j) === 2) {
            next[i][j] = true;
          }
        }
      }
    }
    cells = next;
  }

  function seed() {
    const size = cells.length;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        cells[i][j] = Math.random() < 0.35;
      }
    }
  }

  function buildOuterWalls() {
    const rows = cells.length;
    const cols = cells[0].length;
    for (let i = 0; i < rows; i++) {
      cells[i][0] = true;
      cells[i][cols - 1] = true;
    }
    for (let j = 0; j < cols; j++) {
      cells[0][j] = true;
      cells[rows - 1][j] = true;
    }
  }

  // --- Drawing ---
  function getColors() {
    const cs = getComputedStyle(document.documentElement);
    return {
      alive: cs.getPropertyValue('--heading').trim() || '#e8e4e0',
      dead: cs.getPropertyValue('--surface').trim() || '#181818',
      grid: cs.getPropertyValue('--border').trim() || '#2a2a2a'
    };
  }

  function draw() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(rect.width * dpr);
    if (canvas.width !== w || canvas.height !== w) {
      canvas.width = w;
      canvas.height = w;
    }

    const colors = getColors();
    const size = cells.length;
    const cellW = w / size;
    const cellH = w / size;

    ctx.fillStyle = colors.dead;
    ctx.fillRect(0, 0, w, w);

    // Alive cells
    ctx.fillStyle = colors.alive;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        if (cells[i][j]) {
          ctx.fillRect(i * cellW, j * cellH, Math.ceil(cellW), Math.ceil(cellH));
        }
      }
    }

    // Grid lines (subtle)
    if (size <= 50) {
      ctx.strokeStyle = colors.grid;
      ctx.lineWidth = dpr * 0.5;
      ctx.globalAlpha = 0.4;
      for (let i = 0; i <= size; i++) {
        const pos = Math.round(i * cellW);
        ctx.beginPath();
        ctx.moveTo(pos, 0);
        ctx.lineTo(pos, w);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, pos);
        ctx.lineTo(w, pos);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }

  // --- Mouse interaction ---
  function getCellFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const cellW = rect.width / cells.length;
    const cellH = rect.height / cells[0].length;
    const ci = Math.floor(x / cellW);
    const cj = Math.floor(y / cellH);
    if (ci >= 0 && ci < cells.length && cj >= 0 && cj < cells[0].length) {
      return [ci, cj];
    }
    return null;
  }

  canvas.addEventListener('mousedown', (e) => {
    if (running) return;
    e.preventDefault();
    drawing = true;
    drawMode = e.button !== 2; // right-click = remove
    const cell = getCellFromEvent(e);
    if (cell) {
      cells[cell[0]][cell[1]] = drawMode;
      draw();
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!drawing || running) return;
    const cell = getCellFromEvent(e);
    if (cell) {
      cells[cell[0]][cell[1]] = drawMode;
      draw();
    }
  });

  canvas.addEventListener('mouseup', () => {
    drawing = false;
  });

  canvas.addEventListener('mouseleave', () => {
    drawing = false;
  });

  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
  });

  // Touch support
  canvas.addEventListener('touchstart', (e) => {
    if (running) return;
    e.preventDefault();
    drawing = true;
    drawMode = true;
    const touch = e.touches[0];
    const cell = getCellFromEvent(touch);
    if (cell) {
      cells[cell[0]][cell[1]] = drawMode;
      draw();
    }
  });

  canvas.addEventListener('touchmove', (e) => {
    if (!drawing || running) return;
    e.preventDefault();
    const touch = e.touches[0];
    const cell = getCellFromEvent(touch);
    if (cell) {
      cells[cell[0]][cell[1]] = drawMode;
      draw();
    }
  });

  canvas.addEventListener('touchend', () => {
    drawing = false;
  });

  // --- Button handlers ---
  function setButtonStates(isRunning) {
    running = isRunning;
    btnSeed.disabled = isRunning;
    btnRun.textContent = isRunning ? 'Pause' : 'Run';
    btnReset.disabled = false;
    sizeBtns.forEach((b) => (b.disabled = isRunning));
    canvas.style.cursor = isRunning ? 'default' : 'crosshair';
  }

  function stopRun() {
    if (stepTimer) {
      clearTimeout(stepTimer);
      stepTimer = null;
    }
    setButtonStates(false);
  }

  btnSeed.addEventListener('click', () => {
    if (running) return;
    resetGrid();
    seed();
    draw();
  });

  btnRun.addEventListener('click', () => {
    if (running) {
      // Pause
      stopRun();
      stepInfo.textContent = 'Paused (step ' + stepCount + ')';
      return;
    }
    // Start or resume
    if (stepCount === 0) {
      totalSteps = cells.length + cells[0].length - 30;
      if (totalSteps < 1) totalSteps = 10;
    }
    setButtonStates(true);
    runStep();
  });

  btnReset.addEventListener('click', () => {
    resetGrid();
  });

  function runStep() {
    if (stepCount >= totalSteps) {
      buildOuterWalls();
      draw();
      stepInfo.textContent = 'Done (' + totalSteps + ' steps)';
      setButtonStates(false);
      return;
    }
    step();
    stepCount++;
    stepInfo.textContent = 'Step ' + stepCount + ' / ' + totalSteps;
    draw();
    stepTimer = setTimeout(runStep, 100);
  }

  // --- Resize handling ---
  // ResizeObserver handles both window resize and visibility changes
  // (when display:none is removed, container size goes from 0 to actual)
  let resizeTimer;
  new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(draw, 100);
  }).observe(container);

  // Theme change observer
  const themeObserver = new MutationObserver(() => {
    draw();
  });
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme']
  });

  // --- Init ---
  resetGrid();
})();
