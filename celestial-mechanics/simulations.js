// ═══════════════════════════════════════════════════════════
// Celestial Mechanics Interactive Simulations
// ═══════════════════════════════════════════════════════════

// ── Utility ──
const TAU = 2 * Math.PI;
const DEG = Math.PI / 180;
function lerp(a, b, t) { return a + (b - a) * t; }

// ═══════════════════════════════════════════════════════════
// 1. Gravitational Force Simulation
// ═══════════════════════════════════════════════════════════
(function() {
  const canvas = document.getElementById('gravityCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const distSlider = document.getElementById('gravityDist');
  const massSlider = document.getElementById('gravityMass');
  const info = document.getElementById('gravityInfo');

  const G = 6.674e-11;
  const M_sun = 1.989e30, M_earth = 5.972e24, M_moon = 7.342e22;

  function draw() {
    const distFactor = parseFloat(distSlider.value);
    const massFactor = parseFloat(massSlider.value);
    ctx.clearRect(0, 0, W, H);

    // Background stars
    ctx.fillStyle = '#0a0e27';
    ctx.fillRect(0, 0, W, H);
    for (let i = 0; i < 60; i++) {
      ctx.fillStyle = `rgba(255,255,255,${0.2 + Math.random() * 0.5})`;
      ctx.beginPath();
      ctx.arc((i * 137.5) % W, (i * 97.3) % H, 0.5 + Math.random(), 0, TAU);
      ctx.fill();
    }

    const cx = W / 2, cy = H / 2;
    const sunX = cx - 140, earthX = cx + 60 * distFactor, moonX = earthX + 60;

    // Sun
    const sunR = 30;
    const grad = ctx.createRadialGradient(sunX, cy, 5, sunX, cy, sunR + 10);
    grad.addColorStop(0, '#fff7a0');
    grad.addColorStop(0.5, '#ffcc00');
    grad.addColorStop(1, 'rgba(255,150,0,0)');
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(sunX, cy, sunR + 10, 0, TAU); ctx.fill();
    ctx.fillStyle = '#ffdd44';
    ctx.beginPath(); ctx.arc(sunX, cy, sunR, 0, TAU); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = '11px Noto Sans KR';
    ctx.textAlign = 'center';
    ctx.fillText('태양', sunX, cy + sunR + 16);

    // Earth
    const earthR = 14 * massFactor;
    ctx.fillStyle = '#4488cc';
    ctx.beginPath(); ctx.arc(earthX, cy, earthR, 0, TAU); ctx.fill();
    ctx.fillStyle = '#66bb88';
    ctx.beginPath(); ctx.arc(earthX, cy - 2, earthR * 0.6, 0.2, 2.2); ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillText('지구', earthX, cy + earthR + 16);

    // Moon
    const moonR = 6;
    ctx.fillStyle = '#bbb';
    ctx.beginPath(); ctx.arc(moonX, cy, moonR, 0, TAU); ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillText('달', moonX, cy + moonR + 16);

    // Force vectors
    const r_se = 1.496e11 * distFactor;
    const r_em = 3.844e8;
    const F_se = G * M_sun * (M_earth * massFactor) / (r_se * r_se);
    const F_em = G * (M_earth * massFactor) * M_moon / (r_em * r_em);

    // Sun→Earth force arrow
    const arrowLen_se = Math.min(80, F_se / 1e18);
    drawArrow(ctx, earthX - earthR - 4, cy, earthX - earthR - 4 - arrowLen_se, cy, '#ff6644', 3);
    ctx.fillStyle = '#ff8866'; ctx.font = '10px monospace';
    ctx.fillText(`F = ${F_se.toExponential(2)} N`, (sunX + earthX) / 2, cy - 30);

    // Earth→Moon force arrow
    const arrowLen_em = Math.min(40, F_em / 1e16);
    drawArrow(ctx, moonX - moonR - 2, cy, moonX - moonR - 2 - arrowLen_em, cy, '#44ccff', 2);
    ctx.fillStyle = '#66ddff'; ctx.font = '10px monospace';
    ctx.fillText(`F = ${F_em.toExponential(2)} N`, (earthX + moonX) / 2, cy + 36);

    if (info) {
      info.textContent = `태양-지구 거리: ${(distFactor).toFixed(1)} AU | 지구 질량: ${massFactor.toFixed(1)}x | 태양-지구 인력: ${F_se.toExponential(2)} N | 지구-달 인력: ${F_em.toExponential(2)} N`;
    }
  }

  function drawArrow(ctx, x1, y1, x2, y2, color, w) {
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const len = Math.sqrt((x2-x1)**2 + (y2-y1)**2);
    if (len < 5) return;
    ctx.strokeStyle = color; ctx.lineWidth = w;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 8 * Math.cos(angle - 0.4), y2 - 8 * Math.sin(angle - 0.4));
    ctx.lineTo(x2 - 8 * Math.cos(angle + 0.4), y2 - 8 * Math.sin(angle + 0.4));
    ctx.fill();
  }

  distSlider.addEventListener('input', draw);
  massSlider.addEventListener('input', draw);
  draw();
})();

// ═══════════════════════════════════════════════════════════
// 1b. Escape Velocity Simulation
// ═══════════════════════════════════════════════════════════
(function() {
  const canvas = document.getElementById('escapeCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const velSlider = document.getElementById('escapeVelSlider');
  const velVal = document.getElementById('escapeVelVal');
  const launchBtn = document.getElementById('escapeLaunchBtn');
  const resetBtn = document.getElementById('escapeResetBtn');
  const info = document.getElementById('escapeInfo');

  // Normalized units: Earth radius = 1, GM = 1
  // v_circular = 1, v_escape = sqrt(2) ≈ 1.414
  const GM = 1;
  const Re = 1; // Earth radius in sim units
  const V_CIRC = 7.9;  // km/s (for display)
  const V_ESC = 11.2;  // km/s

  const cx = W * 0.4, cy = H * 0.5;
  const scale = 45; // pixels per sim unit
  const earthPixR = Re * scale;
  const dt = 0.005;
  const maxSteps = 12000;
  const trailMax = 4000;

  let trail = [];
  let sx, sy, svx, svy; // spacecraft state
  let running = false;
  let animId = null;
  let stepCount = 0;

  function resetSim() {
    running = false;
    if (animId) { cancelAnimationFrame(animId); animId = null; }
    trail = [];
    stepCount = 0;
    // Start above Earth surface (altitude ≈ 0.2 Re ≈ 1,270km), launch horizontally
    const launchR = Re * 1.2;
    sx = 0; sy = -launchR;
    const vNorm = parseFloat(velSlider.value);
    svx = vNorm; svy = 0;
    drawFrame();
    updateInfo(vNorm);
  }

  function updateInfo(vNorm) {
    const vKms = (vNorm * V_CIRC).toFixed(1);
    let type = '';
    if (vNorm < 0.85) type = '지표면 충돌 (아궤도)';
    else if (Math.abs(vNorm - 1.0) < 0.05) type = '원 궤도 (v = v₁)';
    else if (vNorm < 1.414) type = '타원 궤도';
    else if (Math.abs(vNorm - 1.414) < 0.05) type = '포물선 탈출 (v = v₂)';
    else type = '쌍곡선 탈출 (v > v₂)';
    info.innerHTML =
      '<strong>초기속도:</strong> ' + vKms + ' km/s (' + vNorm.toFixed(2) + ' v₁) | ' +
      '<strong>궤도 유형:</strong> ' + type + ' | ' +
      '<strong>v₁(원궤도):</strong> ' + V_CIRC + ' km/s | ' +
      '<strong>v₂(탈출):</strong> ' + V_ESC + ' km/s';
  }

  function step() {
    // Velocity Verlet integration
    const r2 = sx * sx + sy * sy;
    const r = Math.sqrt(r2);
    if (r < Re) { running = false; return; } // hit Earth surface
    const a = -GM / r2;
    const ax = a * sx / r;
    const ay = a * sy / r;

    // Half-step velocity
    const hvx = svx + ax * dt * 0.5;
    const hvy = svy + ay * dt * 0.5;
    // Full-step position
    sx += hvx * dt;
    sy += hvy * dt;
    // New acceleration
    const nr2 = sx * sx + sy * sy;
    const nr = Math.sqrt(nr2);
    const na = -GM / nr2;
    const nax = na * sx / nr;
    const nay = na * sy / nr;
    // Full-step velocity
    svx = hvx + nax * dt * 0.5;
    svy = hvy + nay * dt * 0.5;

    trail.push({ x: sx, y: sy });
    if (trail.length > trailMax) trail.shift();
    stepCount++;

    // Stop conditions
    if (nr < Re && stepCount > 5) { running = false; return; } // impact on surface
    if (nr > 12) { running = false; return; } // escaped
    if (stepCount > maxSteps) { running = false; return; }
  }

  function drawFrame() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0e27';
    ctx.fillRect(0, 0, W, H);

    // Stars
    for (let i = 0; i < 50; i++) {
      ctx.fillStyle = `rgba(255,255,255,${0.15 + Math.random() * 0.3})`;
      ctx.beginPath();
      ctx.arc((i * 137.5) % W, (i * 97.3) % H, 0.4 + Math.random() * 0.5, 0, TAU);
      ctx.fill();
    }

    // Orbit reference circles
    ctx.strokeStyle = 'rgba(100,150,255,0.1)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    for (let r = 2; r <= 6; r += 2) {
      ctx.beginPath(); ctx.arc(cx, cy, r * scale, 0, TAU); ctx.stroke();
    }
    ctx.setLineDash([]);

    // Earth
    const eGrad = ctx.createRadialGradient(cx - earthPixR * 0.2, cy - earthPixR * 0.2, earthPixR * 0.1, cx, cy, earthPixR);
    eGrad.addColorStop(0, '#6eb5ff');
    eGrad.addColorStop(0.6, '#2266cc');
    eGrad.addColorStop(1, '#0a3366');
    ctx.fillStyle = eGrad;
    ctx.beginPath(); ctx.arc(cx, cy, earthPixR, 0, TAU); ctx.fill();
    ctx.strokeStyle = '#4488cc';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, earthPixR, 0, TAU); ctx.stroke();

    // Earth label
    ctx.fillStyle = '#88bbee';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('지구', cx, cy + earthPixR + 14);

    // Trail
    if (trail.length > 1) {
      for (let i = 1; i < trail.length; i++) {
        const alpha = 0.15 + 0.85 * (i / trail.length);
        const speed = i < trail.length - 1 ?
          Math.sqrt((trail[i].x - trail[i-1].x)**2 + (trail[i].y - trail[i-1].y)**2) / dt : 0;
        // Color by speed: slow=blue, fast=red
        const t = Math.min(speed / 2, 1);
        const r = Math.floor(80 + 175 * t);
        const g = Math.floor(180 - 80 * t);
        const b = Math.floor(255 - 200 * t);
        ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx + trail[i-1].x * scale, cy + trail[i-1].y * scale);
        ctx.lineTo(cx + trail[i].x * scale, cy + trail[i].y * scale);
        ctx.stroke();
      }
    }

    // Spacecraft
    const spx = cx + sx * scale;
    const spy = cy + sy * scale;
    if (spx > -20 && spx < W + 20 && spy > -20 && spy < H + 20) {
      // Glow
      const sGrad = ctx.createRadialGradient(spx, spy, 1, spx, spy, 10);
      sGrad.addColorStop(0, 'rgba(255,200,100,0.8)');
      sGrad.addColorStop(1, 'rgba(255,200,100,0)');
      ctx.fillStyle = sGrad;
      ctx.beginPath(); ctx.arc(spx, spy, 10, 0, TAU); ctx.fill();
      // Body
      ctx.fillStyle = '#ffdd88';
      ctx.beginPath(); ctx.arc(spx, spy, 3, 0, TAU); ctx.fill();

      // Velocity vector
      const vLen = Math.sqrt(svx * svx + svy * svy);
      if (vLen > 0.01) {
        const vScale = 25;
        ctx.strokeStyle = '#ff6644';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(spx, spy);
        ctx.lineTo(spx + svx / vLen * vScale, spy + svy / vLen * vScale);
        ctx.stroke();
        // Arrow
        const va = Math.atan2(svy, svx);
        ctx.beginPath();
        ctx.moveTo(spx + svx / vLen * vScale, spy + svy / vLen * vScale);
        ctx.lineTo(spx + svx / vLen * vScale - 6 * Math.cos(va - 0.4), spy + svy / vLen * vScale - 6 * Math.sin(va - 0.4));
        ctx.moveTo(spx + svx / vLen * vScale, spy + svy / vLen * vScale);
        ctx.lineTo(spx + svx / vLen * vScale - 6 * Math.cos(va + 0.4), spy + svy / vLen * vScale - 6 * Math.sin(va + 0.4));
        ctx.stroke();
      }
    }

    // Escape velocity reference (right side info)
    const infoX = W - 140;
    ctx.fillStyle = 'rgba(150,180,220,0.7)';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('v₁ (원궤도) = 7.9 km/s', infoX, 24);
    ctx.fillText('v₂ (탈출) = 11.2 km/s', infoX, 40);
    ctx.fillText('v₂ = √2 · v₁', infoX, 56);

    // Speed color legend
    ctx.fillStyle = 'rgba(150,180,220,0.5)';
    ctx.font = '10px sans-serif';
    ctx.fillText('궤적 색상: ', infoX, H - 30);
    const legY = H - 18;
    for (let i = 0; i < 60; i++) {
      const t = i / 59;
      const r = Math.floor(80 + 175 * t);
      const g = Math.floor(180 - 80 * t);
      const b = Math.floor(255 - 200 * t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(infoX + i * 1.5, legY, 1.5, 8);
    }
    ctx.fillStyle = 'rgba(150,180,220,0.5)';
    ctx.fillText('느림', infoX - 2, legY + 7);
    ctx.fillText('빠름', infoX + 95, legY + 7);
  }

  function animate() {
    if (!running) return;
    for (let i = 0; i < 8; i++) { // multiple steps per frame
      step();
      if (!running) break;
    }
    drawFrame();
    if (running) animId = requestAnimationFrame(animate);
  }

  velSlider.addEventListener('input', function() {
    const v = parseFloat(velSlider.value);
    velVal.textContent = (v * V_CIRC).toFixed(1) + ' km/s';
    updateInfo(v);
    if (!running) resetSim();
  });

  launchBtn.addEventListener('click', function() {
    resetSim();
    running = true;
    animate();
  });

  resetBtn.addEventListener('click', resetSim);

  resetSim();
})();

// ═══════════════════════════════════════════════════════════
// 2. Orbital Motion Simulation (Sun-Earth-Moon)
// ═══════════════════════════════════════════════════════════
(function() {
  const canvas = document.getElementById('orbitalCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2;

  const playBtn = document.getElementById('orbitalPlayPause');
  const speedSlider = document.getElementById('orbitalSpeed');
  const info = document.getElementById('orbitalInfo');

  let running = true, time = 0;
  const earthOrbitR = 180, moonOrbitR = 32;
  const earthPeriod = 365.25, moonPeriod = 27.32;
  let trail = [];

  playBtn.addEventListener('click', () => {
    running = !running;
    playBtn.textContent = running ? '⏸ 일시정지' : '▶ 재생';
    if (running) animate();
  });

  function animate() {
    if (!running) return;
    const speed = parseFloat(speedSlider.value);
    time += speed;

    ctx.fillStyle = 'rgba(10,14,39,0.15)';
    ctx.fillRect(0, 0, W, H);

    // Stars (only on first frame or when cleared)
    if (time < speed * 2) {
      ctx.fillStyle = '#0a0e27'; ctx.fillRect(0, 0, W, H);
      for (let i = 0; i < 80; i++) {
        ctx.fillStyle = `rgba(255,255,255,${0.2 + Math.random() * 0.4})`;
        ctx.beginPath();
        ctx.arc((i * 137.5 + 33) % W, (i * 97.3 + 17) % H, 0.5 + Math.random() * 0.5, 0, TAU);
        ctx.fill();
      }
    }

    // Orbit paths
    ctx.strokeStyle = 'rgba(100,150,255,0.15)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, earthOrbitR, 0, TAU); ctx.stroke();

    // Earth position (elliptical approximation e=0.0167)
    const earthAngle = TAU * time / earthPeriod;
    const e = 0.0167;
    const earthR = earthOrbitR * (1 - e * e) / (1 + e * Math.cos(earthAngle));
    const earthX = cx + earthR * Math.cos(earthAngle);
    const earthY = cy - earthR * Math.sin(earthAngle);

    // Moon orbit path
    ctx.strokeStyle = 'rgba(200,200,200,0.12)'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.arc(earthX, earthY, moonOrbitR, 0, TAU); ctx.stroke();

    // Moon position (CCW as seen from North Pole)
    const moonAngle = TAU * time / moonPeriod;
    const moonX = earthX + moonOrbitR * Math.cos(moonAngle);
    const moonY = earthY - moonOrbitR * Math.sin(moonAngle);

    // Earth trail
    trail.push({ x: earthX, y: earthY });
    if (trail.length > 600) trail.shift();
    if (trail.length > 2) {
      ctx.strokeStyle = 'rgba(68,136,204,0.3)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(trail[0].x, trail[0].y);
      for (let i = 1; i < trail.length; i++) {
        if (Math.abs(trail[i].x - trail[i-1].x) < 50) ctx.lineTo(trail[i].x, trail[i].y);
        else ctx.moveTo(trail[i].x, trail[i].y);
      }
      ctx.stroke();
    }

    // Sun
    const sunGrad = ctx.createRadialGradient(cx, cy, 5, cx, cy, 28);
    sunGrad.addColorStop(0, '#fff7a0');
    sunGrad.addColorStop(0.6, '#ffcc00');
    sunGrad.addColorStop(1, 'rgba(255,150,0,0)');
    ctx.fillStyle = sunGrad;
    ctx.beginPath(); ctx.arc(cx, cy, 28, 0, TAU); ctx.fill();
    ctx.fillStyle = '#ffdd44';
    ctx.beginPath(); ctx.arc(cx, cy, 18, 0, TAU); ctx.fill();

    // Earth
    ctx.fillStyle = '#4488cc';
    ctx.beginPath(); ctx.arc(earthX, earthY, 8, 0, TAU); ctx.fill();
    ctx.fillStyle = '#66bb88';
    ctx.beginPath(); ctx.arc(earthX, earthY - 1, 5, 0.3, 2.5); ctx.fill();

    // Axial tilt indicator
    const tiltAngle = 23.44 * DEG;
    ctx.strokeStyle = '#88ccff'; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(earthX, earthY - 12);
    ctx.lineTo(earthX + 4 * Math.sin(tiltAngle), earthY - 12 - 4 * Math.cos(tiltAngle));
    ctx.stroke();

    // Moon
    ctx.fillStyle = '#ccc';
    ctx.beginPath(); ctx.arc(moonX, moonY, 4, 0, TAU); ctx.fill();

    // Labels
    ctx.fillStyle = '#fff'; ctx.font = '11px Noto Sans KR'; ctx.textAlign = 'center';
    ctx.fillText('태양', cx, cy + 30);
    ctx.fillText('지구', earthX, earthY + 16);
    ctx.fillStyle = '#aaa'; ctx.font = '9px Noto Sans KR';
    ctx.fillText('달', moonX, moonY + 10);

    // Date display
    const dayOfYear = Math.floor(time % earthPeriod);
    const months = ['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'];
    const monthDays = [31,28,31,30,31,30,31,31,30,31,30,31];
    let d = dayOfYear, m = 0;
    while (m < 11 && d >= monthDays[m]) { d -= monthDays[m]; m++; }
    ctx.fillStyle = '#7ec8e3'; ctx.font = '12px monospace'; ctx.textAlign = 'left';
    ctx.fillText(`${months[m]} ${d+1}일 (${dayOfYear}일차)`, 10, 20);

    if (info) info.textContent = `경과: ${dayOfYear}일 | 지구 공전각: ${(earthAngle * 180 / Math.PI % 360).toFixed(1)}° | 달 위상각: ${(moonAngle * 180 / Math.PI % 360).toFixed(1)}°`;

    requestAnimationFrame(animate);
  }

  // Initial stars
  ctx.fillStyle = '#0a0e27'; ctx.fillRect(0, 0, W, H);
  animate();
})();

// ═══════════════════════════════════════════════════════════
// 3. Solar Irradiance Simulation (Seoul / Jakarta / London)
// ═══════════════════════════════════════════════════════════
(function() {
  const canvas = document.getElementById('irradianceCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const daySlider = document.getElementById('daySlider');
  const info = document.getElementById('irradianceInfo');

  const S0 = 1361; // solar constant W/m²
  const tilt = 23.44 * DEG;

  // City data: latitude, color, monthly weather clearness factor (Jan-Dec)
  // Based on typical monthly sunshine hours / possible sunshine hours
  const cities = [
    { name: '서울',    nameEn: 'Seoul',   lat: 37.5, color: '#ffaa44', colorDim: 'rgba(255,170,68,0.35)',
      weather: [0.52, 0.53, 0.50, 0.48, 0.49, 0.36, 0.28, 0.32, 0.44, 0.53, 0.50, 0.52],
      note: '장마·태풍(6-9월)' },
    { name: '자카르타', nameEn: 'Jakarta', lat: -6.2, color: '#ff5566', colorDim: 'rgba(255,85,102,0.35)',
      weather: [0.38, 0.38, 0.42, 0.48, 0.52, 0.55, 0.58, 0.60, 0.56, 0.48, 0.40, 0.36],
      note: '우기(11-3월)' },
    { name: '런던',    nameEn: 'London',  lat: 51.5, color: '#44aaff', colorDim: 'rgba(68,170,255,0.35)',
      weather: [0.22, 0.28, 0.33, 0.38, 0.40, 0.42, 0.42, 0.40, 0.36, 0.28, 0.22, 0.18],
      note: '연중 흐림' },
  ];

  function solarDeclination(day) {
    return tilt * Math.sin(TAU / 365 * (day - 81));
  }

  function maxElevation(day, latDeg) {
    const latR = latDeg * DEG;
    const dec = solarDeclination(day);
    return Math.asin(Math.sin(latR) * Math.sin(dec) + Math.cos(latR) * Math.cos(dec));
  }

  function dayLength(day, latDeg) {
    const latR = latDeg * DEG;
    const dec = solarDeclination(day);
    const cosHA = -Math.tan(latR) * Math.tan(dec);
    if (cosHA < -1) return 24;
    if (cosHA > 1) return 0;
    return 2 * Math.acos(cosHA) / TAU * 24;
  }

  function clearSkyIrradiance(day, latDeg) {
    const elev = maxElevation(day, latDeg);
    if (elev <= 0) return 0;
    const AM = 1 / Math.sin(elev);
    const transmission = Math.pow(0.7, Math.pow(AM, 0.678));
    return S0 * Math.sin(elev) * transmission;
  }

  function weatherFactor(day, weatherArr) {
    // Interpolate monthly weather factor to daily
    const monthDays = [31,28,31,30,31,30,31,31,30,31,30,31];
    let d = day, m = 0;
    while (m < 11 && d >= monthDays[m]) { d -= monthDays[m]; m++; }
    const frac = d / monthDays[m];
    const next = (m + 1) % 12;
    return weatherArr[m] * (1 - frac) + weatherArr[next] * frac;
  }

  function actualIrradiance(day, city) {
    return clearSkyIrradiance(day, city.lat) * weatherFactor(day, city.weather);
  }

  function draw() {
    const currentDay = parseInt(daySlider.value);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0f1525'; ctx.fillRect(0, 0, W, H);

    // ── Left: Earth orbit diagram ──
    const lW = W * 0.32, lCx = lW / 2, lCy = H / 2;
    const orbitR = Math.min(lW, H) * 0.32;

    // Sun
    const sunGrad = ctx.createRadialGradient(lCx, lCy, 3, lCx, lCy, 16);
    sunGrad.addColorStop(0, '#fff7a0');
    sunGrad.addColorStop(0.7, '#ffcc00');
    sunGrad.addColorStop(1, 'rgba(255,150,0,0)');
    ctx.fillStyle = sunGrad;
    ctx.beginPath(); ctx.arc(lCx, lCy, 16, 0, TAU); ctx.fill();
    ctx.fillStyle = '#ffdd44';
    ctx.beginPath(); ctx.arc(lCx, lCy, 9, 0, TAU); ctx.fill();

    // Orbit
    ctx.strokeStyle = 'rgba(100,150,255,0.2)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(lCx, lCy, orbitR, 0, TAU); ctx.stroke();

    // Season labels
    ctx.fillStyle = '#556'; ctx.font = '9px Noto Sans KR'; ctx.textAlign = 'center';
    ctx.fillText('하지', lCx, lCy - orbitR - 6);
    ctx.fillText('동지', lCx, lCy + orbitR + 12);
    ctx.fillText('춘분', lCx + orbitR + 6, lCy);
    ctx.fillText('추분', lCx - orbitR - 6, lCy);

    // Earth position (CCW as seen from North Pole)
    const angle = TAU * (currentDay - 80) / 365;
    const ex = lCx + orbitR * Math.cos(angle);
    const ey = lCy - orbitR * Math.sin(angle);
    ctx.fillStyle = '#4488cc';
    ctx.beginPath(); ctx.arc(ex, ey, 6, 0, TAU); ctx.fill();

    // Axial tilt
    ctx.strokeStyle = '#88ccff'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(ex - 4 * Math.sin(tilt), ey - 4 * Math.cos(tilt));
    ctx.lineTo(ex + 10 * Math.sin(tilt), ey + 10 * Math.cos(tilt));
    ctx.stroke();

    // ── Right: Irradiance chart (3 cities) ──
    const rX = W * 0.36, rW = W * 0.61, rY = 14, rH = H - 40;
    const chartL = rX + 38, chartR = rX + rW - 6;
    const chartT = rY + 6, chartB = rY + rH - 24;
    const chartW = chartR - chartL, chartH = chartB - chartT;

    // Axes
    ctx.strokeStyle = '#556'; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(chartL, chartT); ctx.lineTo(chartL, chartB); ctx.lineTo(chartR, chartB);
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle = '#889'; ctx.font = '8px monospace'; ctx.textAlign = 'right';
    for (let v = 0; v <= 1000; v += 200) {
      const y = chartB - (v / 1000) * chartH;
      ctx.fillText(v + '', chartL - 3, y + 3);
      ctx.strokeStyle = 'rgba(100,100,120,0.15)';
      ctx.beginPath(); ctx.moveTo(chartL, y); ctx.lineTo(chartR, y); ctx.stroke();
    }
    ctx.fillStyle = '#889'; ctx.font = '9px Noto Sans KR';
    ctx.save(); ctx.translate(rX + 6, (chartT + chartB) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center'; ctx.fillText('일사량 (W/m²)', 0, 0);
    ctx.restore();

    // X-axis labels
    const monthLabels = ['1','2','3','4','5','6','7','8','9','10','11','12'];
    ctx.fillStyle = '#889'; ctx.font = '8px Noto Sans KR'; ctx.textAlign = 'center';
    for (let m = 0; m < 12; m++) {
      const x = chartL + (m + 0.5) / 12 * chartW;
      ctx.fillText(monthLabels[m] + '월', x, chartB + 12);
    }

    // Monsoon/typhoon season band for Seoul (Jun-Sep)
    const junStart = chartL + (151 / 365) * chartW;
    const sepEnd = chartL + (273 / 365) * chartW;
    ctx.fillStyle = 'rgba(100,100,255,0.06)';
    ctx.fillRect(junStart, chartT, sepEnd - junStart, chartH);
    ctx.fillStyle = 'rgba(100,150,255,0.3)'; ctx.font = '8px Noto Sans KR';
    ctx.textAlign = 'center';
    ctx.fillText('장마·태풍', (junStart + sepEnd) / 2, chartT + 10);

    // Draw curves for each city
    cities.forEach((city, ci) => {
      // Clear-sky (dashed, dimmer)
      ctx.strokeStyle = city.colorDim; ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      for (let d = 0; d < 365; d++) {
        const x = chartL + d / 365 * chartW;
        const y = chartB - clearSkyIrradiance(d, city.lat) / 1000 * chartH;
        d === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Actual (solid, bright)
      ctx.strokeStyle = city.color; ctx.lineWidth = 2;
      ctx.beginPath();
      for (let d = 0; d < 365; d++) {
        const x = chartL + d / 365 * chartW;
        const y = chartB - actualIrradiance(d, city) / 1000 * chartH;
        d === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    // Current day marker
    const cdX = chartL + currentDay / 365 * chartW;
    ctx.strokeStyle = 'rgba(255,255,255,0.4)'; ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(cdX, chartT); ctx.lineTo(cdX, chartB); ctx.stroke();
    ctx.setLineDash([]);

    // Current day dots for each city
    cities.forEach(city => {
      const irr = actualIrradiance(currentDay, city);
      const y = chartB - irr / 1000 * chartH;
      ctx.fillStyle = city.color;
      ctx.beginPath(); ctx.arc(cdX, y, 3.5, 0, TAU); ctx.fill();
    });

    // Legend (top-right of chart)
    const legX = chartR - 125, legY = chartT + 4;
    ctx.font = '9px Noto Sans KR'; ctx.textAlign = 'left';
    cities.forEach((city, i) => {
      const y = legY + i * 14;
      // Solid line
      ctx.strokeStyle = city.color; ctx.lineWidth = 2; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(legX, y + 4); ctx.lineTo(legX + 14, y + 4); ctx.stroke();
      // Dashed line
      ctx.strokeStyle = city.colorDim; ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.moveTo(legX + 16, y + 4); ctx.lineTo(legX + 28, y + 4); ctx.stroke();
      ctx.setLineDash([]);
      // Label
      ctx.fillStyle = city.color;
      ctx.fillText(`${city.name} (${city.lat > 0 ? city.lat+'°N' : Math.abs(city.lat)+'°S'})`, legX + 32, y + 7);
    });
    ctx.fillStyle = '#667'; ctx.font = '8px Noto Sans KR';
    ctx.fillText('실선=실질 / 점선=이론', legX, legY + 48);

    // Info
    const months2 = ['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'];
    const monthDays2 = [31,28,31,30,31,30,31,31,30,31,30,31];
    let dd = currentDay, mm = 0;
    while (mm < 11 && dd >= monthDays2[mm]) { dd -= monthDays2[mm]; mm++; }

    const seoulClear = clearSkyIrradiance(currentDay, 37.5);
    const seoulActual = actualIrradiance(currentDay, cities[0]);
    const jakartaActual = actualIrradiance(currentDay, cities[1]);
    const londonActual = actualIrradiance(currentDay, cities[2]);

    if (info) info.textContent = `${months2[mm]} ${dd+1}일 | 서울: ${seoulActual.toFixed(0)} W/m² (이론 ${seoulClear.toFixed(0)}) | 자카르타: ${jakartaActual.toFixed(0)} | 런던: ${londonActual.toFixed(0)}`;
  }

  daySlider.addEventListener('input', draw);
  draw();
})();

// ═══════════════════════════════════════════════════════════
// 4. Lunar Phase Simulation
// ═══════════════════════════════════════════════════════════
(function() {
  const canvas = document.getElementById('lunarCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const daySlider = document.getElementById('lunarDay');
  const info = document.getElementById('lunarInfo');

  const synodicPeriod = 29.53;
  const phaseNames = ['삭 (신월)', '초승달', '상현달', '상현망간', '망 (보름달)', '하현망간', '하현달', '그믐달'];

  function draw() {
    const day = parseFloat(daySlider.value);
    const phase = (day % synodicPeriod) / synodicPeriod; // 0-1
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0e27'; ctx.fillRect(0, 0, W, H);

    // ── Top view: Sun→Earth→Moon ──
    const topCx = W * 0.35, topCy = H * 0.45;
    const moonOrbit = 100;

    // Sun direction (far left)
    ctx.fillStyle = '#ffcc00'; ctx.font = '11px Noto Sans KR'; ctx.textAlign = 'center';
    const sunArrowX = topCx - 160;
    ctx.fillText('☀ 태양광', sunArrowX, topCy - 10);
    ctx.strokeStyle = '#ffcc44'; ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = topCy - 40 + i * 20;
      ctx.beginPath(); ctx.moveTo(sunArrowX + 30, y); ctx.lineTo(topCx - moonOrbit - 20, y); ctx.stroke();
      // arrowhead
      ctx.beginPath();
      ctx.moveTo(topCx - moonOrbit - 20, y);
      ctx.lineTo(topCx - moonOrbit - 14, y - 3);
      ctx.lineTo(topCx - moonOrbit - 14, y + 3);
      ctx.fill();
    }

    // Earth
    ctx.fillStyle = '#4488cc';
    ctx.beginPath(); ctx.arc(topCx, topCy, 10, 0, TAU); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = '10px Noto Sans KR';
    ctx.fillText('지구', topCx, topCy + 22);

    // Moon orbit
    ctx.strokeStyle = 'rgba(200,200,200,0.15)'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.arc(topCx, topCy, moonOrbit, 0, TAU); ctx.stroke();

    // Moon position (phase 0 = new moon = between Earth and Sun)
    // CCW orbit as seen from North Pole (+ sin for screen Y-down)
    const moonAngle = TAU * phase;
    const moonX = topCx - moonOrbit * Math.cos(moonAngle);
    const moonY = topCy + moonOrbit * Math.sin(moonAngle);

    ctx.fillStyle = '#ccc';
    ctx.beginPath(); ctx.arc(moonX, moonY, 8, 0, TAU); ctx.fill();
    ctx.fillStyle = '#aaa'; ctx.font = '9px Noto Sans KR';
    ctx.fillText('달', moonX, moonY + 16);

    // Sunlit side indicator on moon
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.beginPath(); ctx.arc(moonX, moonY, 8, -Math.PI/2, Math.PI/2); ctx.fill();

    // Phase position markers
    ctx.fillStyle = '#445'; ctx.font = '9px Noto Sans KR';
    ctx.fillText('삭', topCx - moonOrbit, topCy - 12);
    ctx.fillText('하현', topCx, topCy - moonOrbit - 8);
    ctx.fillText('망', topCx + moonOrbit, topCy - 12);
    ctx.fillText('상현', topCx, topCy + moonOrbit + 14);

    // ── Right: Moon appearance from Earth ──
    const moonViewX = W * 0.78, moonViewY = H * 0.45, moonViewR = 55;

    // Moon disk
    ctx.fillStyle = '#e8e4d8';
    ctx.beginPath(); ctx.arc(moonViewX, moonViewY, moonViewR, 0, TAU); ctx.fill();

    // Surface texture
    ctx.fillStyle = '#ccc8b8';
    ctx.beginPath(); ctx.arc(moonViewX - 15, moonViewY - 10, 12, 0, TAU); ctx.fill();
    ctx.beginPath(); ctx.arc(moonViewX + 10, moonViewY + 15, 8, 0, TAU); ctx.fill();
    ctx.beginPath(); ctx.arc(moonViewX + 20, moonViewY - 20, 6, 0, TAU); ctx.fill();
    ctx.beginPath(); ctx.arc(moonViewX - 5, moonViewY + 25, 10, 0, TAU); ctx.fill();

    // Shadow overlay using explicit point-by-point terminator tracing
    // (avoids Canvas arc/ellipse direction ambiguity)
    ctx.save();
    ctx.beginPath(); ctx.arc(moonViewX, moonViewY, moonViewR, 0, TAU); ctx.clip();

    // c = cos(2π·phase): +1 at new, 0 at quarters, -1 at full
    const c = Math.cos(phase * TAU);
    const R = moonViewR;
    const steps = 64;

    // Terminator is an ellipse centered on the disk.
    // Waxing: terminator x = +c·R·cos(a), dark = left limb → terminator
    // Waning: terminator x = -c·R·cos(a), dark = terminator → right limb
    ctx.fillStyle = '#0a0e27';
    ctx.beginPath();

    if (phase < 0.5) {
      // WAXING: shadow on LEFT (Northern Hemisphere: right side lit)
      // 1) Left limb from top to bottom
      for (let i = 0; i <= steps; i++) {
        const a = Math.PI/2 - i * Math.PI / steps;
        const x = moonViewX - R * Math.cos(a);
        const y = moonViewY - R * Math.sin(a);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      // 2) Terminator from bottom to top
      for (let i = 0; i <= steps; i++) {
        const a = -Math.PI/2 + i * Math.PI / steps;
        const x = moonViewX + c * R * Math.cos(a);
        const y = moonViewY - R * Math.sin(a);
        ctx.lineTo(x, y);
      }
    } else {
      // WANING: shadow on RIGHT (Northern Hemisphere: left side lit)
      // 1) Right limb from top to bottom
      for (let i = 0; i <= steps; i++) {
        const a = Math.PI/2 - i * Math.PI / steps;
        const x = moonViewX + R * Math.cos(a);
        const y = moonViewY - R * Math.sin(a);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      // 2) Terminator from bottom to top (note: -c for waning)
      for (let i = 0; i <= steps; i++) {
        const a = -Math.PI/2 + i * Math.PI / steps;
        const x = moonViewX - c * R * Math.cos(a);
        const y = moonViewY - R * Math.sin(a);
        ctx.lineTo(x, y);
      }
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // Moon label
    ctx.fillStyle = '#fff'; ctx.font = 'bold 14px Noto Sans KR'; ctx.textAlign = 'center';
    const phaseIdx = Math.floor(phase * 8) % 8;
    ctx.fillText(phaseNames[phaseIdx], moonViewX, moonViewY + moonViewR + 24);

    // Illumination percentage
    const illumPct = phase <= 0.5 ? phase * 2 * 100 : (1 - (phase - 0.5) * 2) * 100;
    ctx.fillStyle = '#aab'; ctx.font = '11px monospace';
    ctx.fillText(`${illumPct.toFixed(0)}% 조명`, moonViewX, moonViewY + moonViewR + 42);

    // Moonrise/set info
    const riseHour = Math.floor((phase * 24 + 6) % 24);
    const setHour = Math.floor((phase * 24 + 18) % 24);

    if (info) info.textContent = `${day.toFixed(1)}일차 | 위상: ${phaseNames[phaseIdx]} | 조명: ${illumPct.toFixed(0)}% | 월출: ~${riseHour}시 | 월몰: ~${setHour}시`;
  }

  daySlider.addEventListener('input', draw);
  draw();
})();

// ═══════════════════════════════════════════════════════════
// 5. Eclipse Simulation
// ═══════════════════════════════════════════════════════════
(function() {
  const canvas = document.getElementById('eclipseCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const typeSelect = document.getElementById('eclipseType');
  const inclSlider = document.getElementById('eclipseIncl');
  const info = document.getElementById('eclipseInfo');

  function draw() {
    const type = typeSelect.value;
    const inclination = parseFloat(inclSlider.value);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0e27'; ctx.fillRect(0, 0, W, H);

    const cy = H / 2;
    const moonOffset = inclination * 2; // pixels offset from ecliptic

    if (type === 'solar') {
      // Solar Eclipse: Sun -- Moon -- Earth (side view)
      const sunX = 80, moonX = W * 0.52, earthX = W - 100;
      const sunR = 40, moonR = 12, earthR = 18;

      // ── Sunlight rays (diverging from sun surface) ──
      // Rays that pass ABOVE and BELOW the moon → reach Earth
      ctx.lineWidth = 1;
      const rayCount = 14;
      for (let i = 0; i < rayCount; i++) {
        const srcY = cy + sunR * (2 * i / (rayCount - 1) - 1) * 0.95;
        // Ray extends to far right
        const dx = W - sunX - sunR;
        const slope = (srcY - cy) / (sunX + sunR);
        const endY = srcY + slope * dx * 0.5;

        // Check if ray is blocked by moon
        const tMoon = (moonX - sunX - sunR) / dx;
        const rayYatMoon = srcY + (endY - srcY) * tMoon;
        const blocked = Math.abs(rayYatMoon - (cy + moonOffset)) < moonR * 0.9;

        if (blocked) {
          // Draw ray only up to moon, dimmer
          ctx.strokeStyle = 'rgba(255,220,80,0.12)';
          ctx.beginPath();
          ctx.moveTo(sunX + sunR, srcY);
          ctx.lineTo(moonX - moonR, rayYatMoon);
          ctx.stroke();
        } else {
          // Full ray passes through to earth and beyond
          ctx.strokeStyle = 'rgba(255,220,80,0.15)';
          ctx.beginPath();
          ctx.moveTo(sunX + sunR, srcY);
          ctx.lineTo(W, endY);
          ctx.stroke();
        }
      }

      // ── Shadow cones from moon edges (tangent lines) ──
      // Penumbra: outer tangent lines (sun edge → moon opposite edge → earth)
      // Umbra: inner tangent lines (sun edge → moon same edge → converge)
      if (Math.abs(moonOffset) < 30) {
        const moonCy = cy + moonOffset;

        // Penumbra boundary lines (top-sun-edge → bottom-moon-edge → beyond)
        ctx.strokeStyle = 'rgba(100,100,200,0.3)'; ctx.lineWidth = 0.8;
        ctx.setLineDash([5, 3]);
        // Upper penumbra edge
        const pSlope1 = ((moonCy - moonR) - (cy - sunR)) / (moonX - sunX);
        const pEndY1 = (moonCy - moonR) + pSlope1 * (W - moonX);
        ctx.beginPath(); ctx.moveTo(sunX, cy - sunR);
        ctx.lineTo(W, pEndY1); ctx.stroke();
        // Lower penumbra edge
        const pSlope2 = ((moonCy + moonR) - (cy + sunR)) / (moonX - sunX);
        const pEndY2 = (moonCy + moonR) + pSlope2 * (W - moonX);
        ctx.beginPath(); ctx.moveTo(sunX, cy + sunR);
        ctx.lineTo(W, pEndY2); ctx.stroke();
        ctx.setLineDash([]);

        // Penumbra fill
        ctx.fillStyle = 'rgba(30,30,80,0.2)';
        ctx.beginPath();
        ctx.moveTo(moonX, moonCy - moonR);
        ctx.lineTo(W, pEndY1);
        ctx.lineTo(W, pEndY2);
        ctx.lineTo(moonX, moonCy + moonR);
        ctx.fill();

        // Umbra boundary lines (top-sun-edge → top-moon-edge → converge)
        ctx.strokeStyle = 'rgba(200,100,100,0.4)'; ctx.lineWidth = 0.8;
        ctx.setLineDash([3, 3]);
        const uSlope1 = ((moonCy - moonR) - (cy + sunR)) / (moonX - sunX);
        const uSlope2 = ((moonCy + moonR) - (cy - sunR)) / (moonX - sunX);
        // Convergence point
        const convX = moonX + (moonCy - moonR - (cy + sunR + uSlope1 * moonX)) / (-uSlope1);
        const convXreal = moonX - (moonCy - moonR - (moonCy + moonR)) / (uSlope1 - uSlope2);
        // Upper umbra edge
        ctx.beginPath(); ctx.moveTo(sunX, cy + sunR);
        ctx.lineTo(moonX, moonCy - moonR);
        const uEndY1 = (moonCy - moonR) + uSlope1 * (earthX + 50 - moonX);
        ctx.lineTo(earthX + 50, uEndY1); ctx.stroke();
        // Lower umbra edge
        ctx.beginPath(); ctx.moveTo(sunX, cy - sunR);
        ctx.lineTo(moonX, moonCy + moonR);
        const uEndY2 = (moonCy + moonR) + uSlope2 * (earthX + 50 - moonX);
        ctx.lineTo(earthX + 50, uEndY2); ctx.stroke();
        ctx.setLineDash([]);

        // Umbra fill (darker cone)
        ctx.fillStyle = 'rgba(0,0,20,0.45)';
        ctx.beginPath();
        ctx.moveTo(moonX, moonCy - moonR);
        ctx.lineTo(earthX + 50, uEndY1);
        ctx.lineTo(earthX + 50, uEndY2);
        ctx.lineTo(moonX, moonCy + moonR);
        ctx.fill();

        // Labels for shadow regions
        ctx.font = '9px Noto Sans KR'; ctx.textAlign = 'center';
        ctx.fillStyle = 'rgba(200,150,150,0.6)';
        ctx.fillText('본영 (Umbra)', (moonX + earthX) / 2, moonCy - 2);
        ctx.fillStyle = 'rgba(150,150,200,0.6)';
        const penLabelY = Math.min(moonCy - moonR - 12, cy - 30);
        ctx.fillText('반영 (Penumbra)', (moonX + earthX) / 2, penLabelY);
      }

      // Sun
      const sg = ctx.createRadialGradient(sunX, cy, 5, sunX, cy, sunR + 8);
      sg.addColorStop(0, '#fff7a0'); sg.addColorStop(0.7, '#ffcc00');
      sg.addColorStop(1, 'rgba(255,150,0,0)');
      ctx.fillStyle = sg;
      ctx.beginPath(); ctx.arc(sunX, cy, sunR + 8, 0, TAU); ctx.fill();
      ctx.fillStyle = '#ffdd44';
      ctx.beginPath(); ctx.arc(sunX, cy, sunR, 0, TAU); ctx.fill();

      // Moon
      ctx.fillStyle = '#666';
      ctx.beginPath(); ctx.arc(moonX, cy + moonOffset, moonR, 0, TAU); ctx.fill();
      ctx.fillStyle = '#888';
      ctx.beginPath(); ctx.arc(moonX, cy + moonOffset, moonR, -Math.PI/2 - 0.5, Math.PI/2 + 0.5, true); ctx.fill();

      // Earth
      ctx.fillStyle = '#4488cc';
      ctx.beginPath(); ctx.arc(earthX, cy, earthR, 0, TAU); ctx.fill();
      ctx.fillStyle = '#66bb88';
      ctx.beginPath(); ctx.arc(earthX, cy - 2, 11, 0.2, 2.2); ctx.fill();
      // Atmosphere glow
      ctx.strokeStyle = 'rgba(100,180,255,0.3)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(earthX, cy, earthR + 2, 0, TAU); ctx.stroke();

      // Ecliptic line
      ctx.strokeStyle = 'rgba(255,100,100,0.3)'; ctx.lineWidth = 0.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
      ctx.setLineDash([]);

      // Labels
      ctx.fillStyle = '#fff'; ctx.font = '12px Noto Sans KR'; ctx.textAlign = 'center';
      ctx.fillText('태양', sunX, cy + sunR + 18);
      ctx.fillText('달', moonX, cy + moonOffset + moonR + 18);
      ctx.fillText('지구', earthX, cy + earthR + 18);
      ctx.fillStyle = '#ff6666'; ctx.font = '10px Noto Sans KR';
      ctx.fillText('황도면', W - 40, cy - 6);

      const eclipseOccurs = Math.abs(moonOffset) < earthR;
      if (info) info.textContent = `일식 (Solar Eclipse) | 달 궤도 경사: ${inclination.toFixed(1)}° | ${eclipseOccurs ? '✓ 일식 발생!' : '✗ 일식 불발 (달이 황도면에서 벗어남)'}`;

    } else {
      // Lunar Eclipse: Sun -- Earth -- Moon (side view)
      const sunX = 80, earthX = W * 0.45, moonX = W - 90;
      const sunR = 40, earthR = 18, moonR = 10;
      const atmoR = earthR + 4; // atmosphere radius

      // ── Sunlight rays ──
      const rayCount = 18;
      ctx.lineWidth = 1;
      for (let i = 0; i < rayCount; i++) {
        const srcY = cy + sunR * (2 * i / (rayCount - 1) - 1) * 0.95;
        const dx = W - sunX - sunR;
        const slope = (srcY - cy) * 0.3 / dx;
        const endY = srcY + slope * dx;

        // Check if ray hits Earth
        const tEarth = (earthX - sunX - sunR) / dx;
        const rayYatEarth = srcY + (endY - srcY) * tEarth;
        const hitsCore = Math.abs(rayYatEarth - cy) < earthR;
        const hitsAtmo = Math.abs(rayYatEarth - cy) < atmoR;

        if (hitsCore) {
          // Ray blocked by solid Earth → draw only to Earth
          ctx.strokeStyle = 'rgba(255,220,80,0.12)';
          ctx.beginPath();
          ctx.moveTo(sunX + sunR, srcY);
          ctx.lineTo(earthX - earthR, rayYatEarth);
          ctx.stroke();
        } else if (hitsAtmo) {
          // Ray passes through atmosphere → refracted, turns reddish
          ctx.strokeStyle = 'rgba(255,220,80,0.15)';
          ctx.beginPath();
          ctx.moveTo(sunX + sunR, srcY);
          ctx.lineTo(earthX - atmoR, rayYatEarth);
          ctx.stroke();
          // Refracted red ray bends into shadow
          ctx.strokeStyle = 'rgba(220,80,40,0.3)';
          ctx.beginPath();
          ctx.moveTo(earthX + atmoR, rayYatEarth);
          const bendSign = rayYatEarth > cy ? -1 : 1;
          ctx.quadraticCurveTo(
            (earthX + moonX) / 2, rayYatEarth + bendSign * 8,
            moonX, cy + moonOffset
          );
          ctx.stroke();
        } else {
          // Ray passes freely
          ctx.strokeStyle = 'rgba(255,220,80,0.13)';
          ctx.beginPath();
          ctx.moveTo(sunX + sunR, srcY);
          ctx.lineTo(W, endY);
          ctx.stroke();
        }
      }

      // ── Shadow cones from Earth edges ──
      // Penumbra edges (sun edge → earth opposite edge → diverge)
      ctx.strokeStyle = 'rgba(100,100,200,0.35)'; ctx.lineWidth = 0.8;
      ctx.setLineDash([5, 3]);
      // Upper penumbra: from sun bottom → earth top → beyond
      const pSlope1 = ((cy - earthR) - (cy + sunR)) / (earthX - sunX);
      const pEndY1 = (cy - earthR) + pSlope1 * (W - earthX);
      ctx.beginPath(); ctx.moveTo(sunX, cy + sunR);
      ctx.lineTo(earthX, cy - earthR); ctx.lineTo(W, pEndY1); ctx.stroke();
      // Lower penumbra: from sun top → earth bottom → beyond
      const pSlope2 = ((cy + earthR) - (cy - sunR)) / (earthX - sunX);
      const pEndY2 = (cy + earthR) + pSlope2 * (W - earthX);
      ctx.beginPath(); ctx.moveTo(sunX, cy - sunR);
      ctx.lineTo(earthX, cy + earthR); ctx.lineTo(W, pEndY2); ctx.stroke();
      ctx.setLineDash([]);

      // Penumbra fill
      ctx.fillStyle = 'rgba(30,30,80,0.15)';
      ctx.beginPath();
      ctx.moveTo(earthX, cy - earthR);
      ctx.lineTo(W, pEndY1);
      ctx.lineTo(W, pEndY2);
      ctx.lineTo(earthX, cy + earthR);
      ctx.fill();

      // Umbra edges (sun top → earth top → converge beyond earth)
      ctx.strokeStyle = 'rgba(200,100,100,0.45)'; ctx.lineWidth = 0.8;
      ctx.setLineDash([3, 3]);
      const uSlope1 = ((cy - earthR) - (cy - sunR)) / (earthX - sunX);
      const uEndY1 = (cy - earthR) + uSlope1 * (W - earthX);
      ctx.beginPath(); ctx.moveTo(sunX, cy - sunR);
      ctx.lineTo(earthX, cy - earthR); ctx.lineTo(W, uEndY1); ctx.stroke();
      const uSlope2 = ((cy + earthR) - (cy + sunR)) / (earthX - sunX);
      const uEndY2 = (cy + earthR) + uSlope2 * (W - earthX);
      ctx.beginPath(); ctx.moveTo(sunX, cy + sunR);
      ctx.lineTo(earthX, cy + earthR); ctx.lineTo(W, uEndY2); ctx.stroke();
      ctx.setLineDash([]);

      // Umbra fill
      ctx.fillStyle = 'rgba(0,0,20,0.35)';
      ctx.beginPath();
      ctx.moveTo(earthX, cy - earthR);
      ctx.lineTo(W, uEndY1);
      ctx.lineTo(W, uEndY2);
      ctx.lineTo(earthX, cy + earthR);
      ctx.fill();

      // Shadow region labels
      ctx.font = '9px Noto Sans KR'; ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(200,150,150,0.6)';
      ctx.fillText('본영 (Umbra)', (earthX + moonX) / 2, cy - 3);
      ctx.fillStyle = 'rgba(150,150,200,0.6)';
      ctx.fillText('반영 (Penumbra)', (earthX + moonX) / 2, cy - earthR - 14);

      // Sun
      const sg = ctx.createRadialGradient(sunX, cy, 5, sunX, cy, sunR + 8);
      sg.addColorStop(0, '#fff7a0'); sg.addColorStop(0.7, '#ffcc00');
      sg.addColorStop(1, 'rgba(255,150,0,0)');
      ctx.fillStyle = sg;
      ctx.beginPath(); ctx.arc(sunX, cy, sunR + 8, 0, TAU); ctx.fill();
      ctx.fillStyle = '#ffdd44';
      ctx.beginPath(); ctx.arc(sunX, cy, sunR, 0, TAU); ctx.fill();

      // Earth
      ctx.fillStyle = '#4488cc';
      ctx.beginPath(); ctx.arc(earthX, cy, earthR, 0, TAU); ctx.fill();
      ctx.fillStyle = '#66bb88';
      ctx.beginPath(); ctx.arc(earthX, cy - 2, 11, 0.2, 2.2); ctx.fill();
      // Atmosphere ring (key for red refraction)
      const atmoGrad = ctx.createRadialGradient(earthX, cy, earthR, earthX, cy, atmoR + 2);
      atmoGrad.addColorStop(0, 'rgba(100,180,255,0.25)');
      atmoGrad.addColorStop(0.5, 'rgba(100,180,255,0.15)');
      atmoGrad.addColorStop(1, 'rgba(100,180,255,0)');
      ctx.fillStyle = atmoGrad;
      ctx.beginPath(); ctx.arc(earthX, cy, atmoR + 2, 0, TAU); ctx.fill();
      // Atmosphere label
      ctx.fillStyle = 'rgba(100,200,255,0.5)'; ctx.font = '8px Noto Sans KR';
      ctx.fillText('대기', earthX, cy + earthR + 12);

      // Moon
      const moonY = cy + moonOffset;
      ctx.fillStyle = '#ccc';
      ctx.beginPath(); ctx.arc(moonX, moonY, moonR, 0, TAU); ctx.fill();

      // If in shadow, darken moon and apply red tint (blood moon)
      if (Math.abs(moonOffset) < earthR * 1.5) {
        const shadowFactor = 1 - Math.abs(moonOffset) / (earthR * 1.5);
        // Base shadow
        ctx.fillStyle = `rgba(0,0,10,${shadowFactor * 0.6})`;
        ctx.beginPath(); ctx.arc(moonX, moonY, moonR, 0, TAU); ctx.fill();
        // Red tint from refracted light
        ctx.fillStyle = `rgba(180,50,20,${shadowFactor * 0.55})`;
        ctx.beginPath(); ctx.arc(moonX, moonY, moonR, 0, TAU); ctx.fill();
      }

      // Ecliptic line
      ctx.strokeStyle = 'rgba(255,100,100,0.3)'; ctx.lineWidth = 0.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
      ctx.setLineDash([]);

      // Labels
      ctx.fillStyle = '#fff'; ctx.font = '12px Noto Sans KR'; ctx.textAlign = 'center';
      ctx.fillText('태양', sunX, cy + sunR + 18);
      ctx.fillText('지구', earthX, cy + earthR + 22);
      ctx.fillText('달', moonX, moonY + moonR + 18);
      ctx.fillStyle = '#ff6666'; ctx.font = '10px Noto Sans KR';
      ctx.fillText('황도면', W - 40, cy - 6);

      // Info about red refraction
      ctx.fillStyle = 'rgba(220,100,60,0.6)'; ctx.font = '8px Noto Sans KR';
      ctx.fillText('대기 굴절 → 붉은빛', (earthX + moonX) / 2, cy + earthR + 22);

      const eclipseOccurs = Math.abs(moonOffset) < earthR * 1.5;
      if (info) info.textContent = `월식 (Lunar Eclipse) | 달 궤도 경사: ${inclination.toFixed(1)}° | ${eclipseOccurs ? '✓ 월식 발생! (대기 굴절로 달이 붉게 물듦)' : '✗ 월식 불발'}`;
    }
  }

  typeSelect.addEventListener('change', draw);
  inclSlider.addEventListener('input', draw);
  draw();
})();

// ═══════════════════════════════════════════════════════════
// 5b. Eclipse Observer Simulation (Seoul, 37.5°N)
// ═══════════════════════════════════════════════════════════
(function() {
  const canvas = document.getElementById('eclipseObsCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const eventSelect = document.getElementById('eclipseEvent');
  const timeSlider = document.getElementById('eclipseTime');
  const info = document.getElementById('eclipseObsInfo');

  // Real eclipse data visible from Seoul area (UTC+9)
  // {name, type, date, startH, maxH, endH, maxMag, sunAlt, desc}
  const eclipseEvents = [
    { id: 'solar_20350902', name: '2035-09-02 개기일식', type: 'solar', kind: 'total',
      date: '2035년 9월 2일', startH: 8.8, maxH: 9.95, endH: 11.2, maxMag: 1.02,
      sunAltStart: 35, sunAltMax: 47, sunAltEnd: 57,
      desc: '한반도에서 관측 가능한 개기일식. 개기식 지속시간 약 2분 10초. 북한 평양~원산 부근이 개기식 중심대.' },
    { id: 'solar_20300601', name: '2030-06-01 금환일식', type: 'solar', kind: 'annular',
      date: '2030년 6월 1일', startH: 15.5, maxH: 16.8, endH: 17.9, maxMag: 0.89,
      sunAltStart: 52, sunAltMax: 38, sunAltEnd: 22,
      desc: '서울에서 부분일식(식분 ~0.89)으로 관측. 금환일식 중심대는 일본 남부.' },
    { id: 'solar_20280722', name: '2028-07-22 부분일식', type: 'solar', kind: 'partial',
      date: '2028년 7월 22일', startH: 17.0, maxH: 17.8, endH: 18.5, maxMag: 0.36,
      sunAltStart: 30, sunAltMax: 20, sunAltEnd: 10,
      desc: '서울에서 부분일식(식분 ~0.36). 해질녘에 태양 우측 상단이 가려짐.' },
    { id: 'lunar_20250914', name: '2025-09-07 개기월식', type: 'lunar', kind: 'total',
      date: '2025년 9월 7일', startH: 23.3, maxH: 1.5, endH: 3.7, maxMag: 1.36,
      moonAltStart: 35, moonAltMax: 28, moonAltEnd: 15,
      desc: '한국에서 전 과정 관측 가능한 개기월식. 달이 붉게 물드는 "블러드 문".' },
    { id: 'lunar_20261231', name: '2028-12-31 개기월식', type: 'lunar', kind: 'total',
      date: '2028년 12월 31일', startH: 22.1, maxH: 0.5, endH: 2.9, maxMag: 1.22,
      moonAltStart: 60, moonAltMax: 65, moonAltEnd: 55,
      desc: '연말 밤하늘의 개기월식. 달이 높이 떠있어 최적의 관측 조건.' },
  ];

  function getEvent() {
    const idx = parseInt(eventSelect.value);
    return eclipseEvents[idx] || eclipseEvents[0];
  }

  // Populate select
  eclipseEvents.forEach((ev, i) => {
    const opt = document.createElement('option');
    opt.value = i; opt.textContent = ev.name;
    eventSelect.appendChild(opt);
  });

  function draw() {
    const ev = getEvent();
    const t = parseFloat(timeSlider.value); // 0-1 progress through eclipse
    ctx.clearRect(0, 0, W, H);

    const isSolar = ev.type === 'solar';
    const currentH = ev.startH + t * (ev.endH - ev.startH);
    const hours = Math.floor(currentH % 24);
    const mins = Math.floor((currentH % 1) * 60);

    // Sky gradient based on time and eclipse
    const isNight = !isSolar;
    if (isSolar) {
      // Daytime sky, darken during max eclipse
      const maxT = (ev.maxH - ev.startH) / (ev.endH - ev.startH);
      const distToMax = Math.abs(t - maxT);
      const eclipseDarkness = Math.max(0, 1 - distToMax * 4) * ev.maxMag;
      const skyBright = Math.max(0.1, 1 - eclipseDarkness * 0.8);
      const r = Math.floor(100 * skyBright);
      const g = Math.floor(160 * skyBright);
      const b = Math.floor(230 * skyBright);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(0, 0, W, H);
      // Stars visible during total eclipse
      if (eclipseDarkness > 0.9) {
        for (let i = 0; i < 30; i++) {
          ctx.fillStyle = `rgba(255,255,255,${eclipseDarkness - 0.8})`;
          ctx.beginPath();
          ctx.arc((i * 97 + 20) % W, (i * 61 + 10) % (H * 0.6), 1, 0, TAU);
          ctx.fill();
        }
      }
    } else {
      // Night sky
      const grad = ctx.createLinearGradient(0, 0, 0, H);
      grad.addColorStop(0, '#0a0e27'); grad.addColorStop(1, '#1a1a30');
      ctx.fillStyle = grad; ctx.fillRect(0, 0, W, H);
      for (let i = 0; i < 50; i++) {
        ctx.fillStyle = `rgba(255,255,255,${0.2 + Math.random() * 0.4})`;
        ctx.beginPath();
        ctx.arc((i * 137 + 30) % W, (i * 89 + 15) % (H * 0.65), 0.6, 0, TAU);
        ctx.fill();
      }
    }

    // Ground / horizon
    ctx.fillStyle = isSolar ? '#2a5a2a' : '#1a2a1a';
    ctx.fillRect(0, H * 0.78, W, H * 0.22);
    // Skyline silhouette
    ctx.fillStyle = isSolar ? '#1a3a1a' : '#0a1a0a';
    ctx.beginPath();
    ctx.moveTo(0, H * 0.78);
    for (let x = 0; x <= W; x += 20) {
      ctx.lineTo(x, H * 0.78 - Math.sin(x * 0.02) * 8 - Math.random() * 3);
    }
    ctx.lineTo(W, H * 0.78);
    ctx.fill();

    // Seoul landmark hint
    ctx.fillStyle = isSolar ? '#1a3a1a' : '#0a1a0a';
    ctx.fillRect(W * 0.45, H * 0.68, 6, H * 0.10); // tower
    ctx.beginPath();
    ctx.arc(W * 0.45 + 3, H * 0.67, 8, 0, TAU); ctx.fill();

    // Calculate celestial body position
    const altStart = isSolar ? ev.sunAltStart : ev.moonAltStart;
    const altMax = isSolar ? ev.sunAltMax : ev.moonAltMax;
    const altEnd = isSolar ? ev.sunAltEnd : ev.moonAltEnd;
    const alt = altStart + t * (altEnd - altStart); // simplified linear interpolation
    const bodyY = H * 0.78 - (alt / 90) * H * 0.70;
    const bodyX = W * 0.2 + t * W * 0.6;

    if (isSolar) {
      // ── Solar eclipse view from Seoul ──
      const sunR = 36;
      const maxT = (ev.maxH - ev.startH) / (ev.endH - ev.startH);
      // Moon moves across sun disk
      const moonPhase = (t - maxT) * 3; // -1.5 to 1.5 normalized
      const moonOffsetX = moonPhase * sunR * 1.2;
      const moonOffsetY = -moonPhase * sunR * 0.3; // slight diagonal

      // Sun glow
      const glowR = sunR + 15;
      const sg = ctx.createRadialGradient(bodyX, bodyY, sunR * 0.8, bodyX, bodyY, glowR);
      sg.addColorStop(0, 'rgba(255,200,50,0.3)');
      sg.addColorStop(1, 'rgba(255,200,50,0)');
      ctx.fillStyle = sg;
      ctx.beginPath(); ctx.arc(bodyX, bodyY, glowR, 0, TAU); ctx.fill();

      // Corona (visible near totality)
      const distMax = Math.abs(t - maxT);
      if (distMax < 0.15 && ev.kind === 'total') {
        const coronaAlpha = Math.max(0, 1 - distMax * 10);
        for (let a = 0; a < TAU; a += 0.15) {
          const cLen = sunR * (1.5 + 0.5 * Math.sin(a * 7));
          const cg = ctx.createLinearGradient(
            bodyX + sunR * 0.8 * Math.cos(a), bodyY + sunR * 0.8 * Math.sin(a),
            bodyX + cLen * Math.cos(a), bodyY + cLen * Math.sin(a));
          cg.addColorStop(0, `rgba(255,255,255,${coronaAlpha * 0.5})`);
          cg.addColorStop(1, 'rgba(255,255,255,0)');
          ctx.strokeStyle = cg; ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(bodyX + sunR * 0.9 * Math.cos(a), bodyY + sunR * 0.9 * Math.sin(a));
          ctx.lineTo(bodyX + cLen * Math.cos(a), bodyY + cLen * Math.sin(a));
          ctx.stroke();
        }
      }

      // Sun disk
      ctx.fillStyle = '#ffdd44';
      ctx.beginPath(); ctx.arc(bodyX, bodyY, sunR, 0, TAU); ctx.fill();

      // Moon disk overlapping
      ctx.fillStyle = '#1a1a2a';
      const moonR = sunR * (ev.kind === 'annular' ? 0.92 : 1.02);
      ctx.beginPath(); ctx.arc(bodyX + moonOffsetX, bodyY + moonOffsetY, moonR, 0, TAU); ctx.fill();

    } else {
      // ── Lunar eclipse view from Seoul ──
      const moonR = 32;
      const maxT = (ev.maxH - ev.startH) / (ev.endH - ev.startH);
      const eclipseProgress = 1 - Math.abs(t - maxT) / maxT; // 0 at start/end, 1 at max

      // Moon disk (bright)
      ctx.fillStyle = '#e8e4d8';
      ctx.beginPath(); ctx.arc(bodyX, bodyY, moonR, 0, TAU); ctx.fill();

      // Mare (surface features)
      ctx.fillStyle = '#ccc8b8';
      ctx.beginPath(); ctx.arc(bodyX - 8, bodyY - 5, 7, 0, TAU); ctx.fill();
      ctx.beginPath(); ctx.arc(bodyX + 6, bodyY + 8, 5, 0, TAU); ctx.fill();
      ctx.beginPath(); ctx.arc(bodyX + 12, bodyY - 10, 4, 0, TAU); ctx.fill();

      // Earth's shadow sweeping across
      ctx.save();
      ctx.beginPath(); ctx.arc(bodyX, bodyY, moonR, 0, TAU); ctx.clip();

      // Shadow enters from left
      const shadowCx = bodyX - moonR * 2.5 + eclipseProgress * moonR * 2.5;
      const shadowR = moonR * 2.2;

      // Penumbral shadow
      const penGrad = ctx.createRadialGradient(shadowCx, bodyY, shadowR * 0.5, shadowCx, bodyY, shadowR);
      penGrad.addColorStop(0, 'rgba(0,0,0,0.7)');
      penGrad.addColorStop(0.6, 'rgba(80,20,10,0.5)');
      penGrad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = penGrad;
      ctx.beginPath(); ctx.arc(shadowCx, bodyY, shadowR, 0, TAU); ctx.fill();

      // Blood moon tint at maximum
      if (eclipseProgress > 0.6) {
        const tint = (eclipseProgress - 0.6) / 0.4;
        ctx.fillStyle = `rgba(180,50,20,${tint * 0.5})`;
        ctx.beginPath(); ctx.arc(bodyX, bodyY, moonR, 0, TAU); ctx.fill();
      }
      ctx.restore();
    }

    // Time and info display
    ctx.fillStyle = '#fff'; ctx.font = 'bold 13px Noto Sans KR'; ctx.textAlign = 'left';
    ctx.fillText(`서울 관측 시각: ${String(hours).padStart(2,'0')}:${String(mins).padStart(2,'0')} KST`, 12, 22);
    ctx.font = '11px Noto Sans KR'; ctx.fillStyle = '#ddd';
    ctx.fillText(`고도: ${alt.toFixed(1)}°`, 12, 40);
    ctx.fillText(ev.date, 12, 56);

    // Event type badge
    ctx.fillStyle = isSolar ? '#ff8844' : '#8844ff';
    const badge = isSolar ? (ev.kind === 'total' ? '개기일식' : ev.kind === 'annular' ? '금환일식' : '부분일식')
                          : (ev.kind === 'total' ? '개기월식' : '부분월식');
    ctx.font = 'bold 12px Noto Sans KR'; ctx.textAlign = 'right';
    ctx.fillText(badge, W - 12, 22);

    // Compass direction
    ctx.fillStyle = '#aaa'; ctx.font = '10px Noto Sans KR'; ctx.textAlign = 'center';
    ctx.fillText('남', W * 0.5, H * 0.78 + 16);
    ctx.fillText('동', W * 0.08, H * 0.78 + 16);
    ctx.fillText('서', W * 0.92, H * 0.78 + 16);

    if (info) info.textContent = ev.desc;
  }

  eventSelect.addEventListener('change', () => { timeSlider.value = 0; draw(); });
  timeSlider.addEventListener('input', draw);
  draw();
})();

// ═══════════════════════════════════════════════════════════
// 6. Geocentric vs Heliocentric Comparison
// ═══════════════════════════════════════════════════════════
(function() {
  const geoCanvas = document.getElementById('geoCanvas');
  const helioCanvas = document.getElementById('helioCanvas');
  if (!geoCanvas || !helioCanvas) return;
  const gCtx = geoCanvas.getContext('2d');
  const hCtx = helioCanvas.getContext('2d');
  const gW = geoCanvas.width, gH = geoCanvas.height;
  const hW = helioCanvas.width, hH = helioCanvas.height;

  const playBtn = document.getElementById('compPlayPause');
  const speedSlider = document.getElementById('compSpeed');
  const info = document.getElementById('compInfo');

  let running = true, time = 0;
  const earthPeriod = 365.25, marsPeriod = 687;

  // Mars trail in geocentric view
  let marsGeoTrail = [];

  playBtn.addEventListener('click', () => {
    running = !running;
    playBtn.textContent = running ? '⏸ 일시정지' : '▶ 재생';
    if (running) animate();
  });

  function drawStars(ctx, w, h) {
    for (let i = 0; i < 40; i++) {
      ctx.fillStyle = `rgba(255,255,255,${0.15 + Math.random() * 0.3})`;
      ctx.beginPath();
      ctx.arc((i * 137.5 + 50) % w, (i * 97.3 + 30) % h, 0.5, 0, TAU);
      ctx.fill();
    }
  }

  function animate() {
    if (!running) return;
    const speed = parseFloat(speedSlider.value);
    time += speed;

    const earthAngle = TAU * time / earthPeriod;
    const marsAngle = TAU * time / marsPeriod;

    // ── Heliocentric (right) ──
    hCtx.fillStyle = '#0a0e27'; hCtx.fillRect(0, 0, hW, hH);
    drawStars(hCtx, hW, hH);
    const hCx = hW / 2, hCy = hH / 2;
    const earthR = 90, marsR = 135;

    // Orbits
    hCtx.strokeStyle = 'rgba(68,136,204,0.2)'; hCtx.lineWidth = 0.5;
    hCtx.beginPath(); hCtx.arc(hCx, hCy, earthR, 0, TAU); hCtx.stroke();
    hCtx.strokeStyle = 'rgba(204,100,68,0.2)';
    hCtx.beginPath(); hCtx.arc(hCx, hCy, marsR, 0, TAU); hCtx.stroke();

    // Sun
    hCtx.fillStyle = '#ffdd44';
    hCtx.beginPath(); hCtx.arc(hCx, hCy, 10, 0, TAU); hCtx.fill();

    // Earth
    const eX = hCx + earthR * Math.cos(earthAngle);
    const eY = hCy - earthR * Math.sin(earthAngle);
    hCtx.fillStyle = '#4488cc';
    hCtx.beginPath(); hCtx.arc(eX, eY, 5, 0, TAU); hCtx.fill();

    // Mars
    const mX = hCx + marsR * Math.cos(marsAngle);
    const mY = hCy - marsR * Math.sin(marsAngle);
    hCtx.fillStyle = '#cc6644';
    hCtx.beginPath(); hCtx.arc(mX, mY, 4, 0, TAU); hCtx.fill();

    // Sight line
    hCtx.strokeStyle = 'rgba(255,255,100,0.2)'; hCtx.lineWidth = 0.5;
    hCtx.setLineDash([3, 3]);
    hCtx.beginPath(); hCtx.moveTo(eX, eY); hCtx.lineTo(mX, mY); hCtx.stroke();
    hCtx.setLineDash([]);

    // Labels
    hCtx.fillStyle = '#fff'; hCtx.font = '9px Noto Sans KR'; hCtx.textAlign = 'center';
    hCtx.fillText('태양', hCx, hCy + 18);
    hCtx.fillText('지구', eX, eY + 12);
    hCtx.fillText('화성', mX, mY + 12);

    // ── Geocentric (left) ──
    gCtx.fillStyle = '#0a0e27'; gCtx.fillRect(0, 0, gW, gH);
    drawStars(gCtx, gW, gH);
    const gCx = gW / 2, gCy = gH / 2;

    // Earth at center
    gCtx.fillStyle = '#4488cc';
    gCtx.beginPath(); gCtx.arc(gCx, gCy, 8, 0, TAU); gCtx.fill();
    gCtx.fillStyle = '#fff'; gCtx.font = '9px Noto Sans KR'; gCtx.textAlign = 'center';
    gCtx.fillText('지구', gCx, gCy + 18);

    // Sun orbits Earth in geocentric model
    const sunGeoAngle = earthAngle; // Sun appears to orbit Earth
    const sunGeoR = 80;
    const sunGeoX = gCx + sunGeoR * Math.cos(sunGeoAngle);
    const sunGeoY = gCy - sunGeoR * Math.sin(sunGeoAngle);
    gCtx.fillStyle = '#ffdd44';
    gCtx.beginPath(); gCtx.arc(sunGeoX, sunGeoY, 8, 0, TAU); gCtx.fill();
    gCtx.fillStyle = '#fff'; gCtx.font = '9px Noto Sans KR';
    gCtx.fillText('태양', sunGeoX, sunGeoY + 15);
    gCtx.strokeStyle = 'rgba(255,220,68,0.15)';
    gCtx.beginPath(); gCtx.arc(gCx, gCy, sunGeoR, 0, TAU); gCtx.stroke();

    // Mars in geocentric: relative position
    const marsRelX = mX - eX; // Mars position relative to Earth
    const marsRelY = mY - eY;
    const marsRelDist = Math.sqrt(marsRelX * marsRelX + marsRelY * marsRelY);
    const marsRelAngle = Math.atan2(marsRelY, marsRelX);

    // Scale to fit canvas
    const marsGeoR = marsRelDist * 0.7;
    const marsGeoX = gCx + marsGeoR * Math.cos(marsRelAngle);
    const marsGeoY = gCy + marsGeoR * Math.sin(marsRelAngle);

    // Mars trail (shows epicycloid / retrograde)
    marsGeoTrail.push({ x: marsGeoX, y: marsGeoY });
    if (marsGeoTrail.length > 800) marsGeoTrail.shift();

    gCtx.strokeStyle = 'rgba(204,100,68,0.4)'; gCtx.lineWidth = 1;
    gCtx.beginPath();
    for (let i = 0; i < marsGeoTrail.length; i++) {
      const p = marsGeoTrail[i];
      i === 0 ? gCtx.moveTo(p.x, p.y) : gCtx.lineTo(p.x, p.y);
    }
    gCtx.stroke();

    gCtx.fillStyle = '#cc6644';
    gCtx.beginPath(); gCtx.arc(marsGeoX, marsGeoY, 4, 0, TAU); gCtx.fill();
    gCtx.fillStyle = '#fff'; gCtx.font = '9px Noto Sans KR';
    gCtx.fillText('화성', marsGeoX, marsGeoY + 12);

    // Epicycle explanation circle (visual only, approximate)
    if (marsGeoTrail.length > 100) {
      // Show that the trail forms loops
      gCtx.fillStyle = 'rgba(255,200,100,0.5)'; gCtx.font = '10px Noto Sans KR';
      gCtx.fillText('↺ 역행운동', gCx, gH - 12);
    }

    // Check retrograde
    const prevAngle = marsGeoTrail.length > 5 ?
      Math.atan2(marsGeoTrail[marsGeoTrail.length-5].y - gCy, marsGeoTrail[marsGeoTrail.length-5].x - gCx) : marsRelAngle;
    const angleDiff = marsRelAngle - prevAngle;
    const isRetrograde = angleDiff < -0.001 || (angleDiff > Math.PI);

    if (info) {
      const dayNum = Math.floor(time % (earthPeriod * 2));
      info.textContent = `${dayNum}일 | ${isRetrograde ? '⚠ 역행 중 (Retrograde)' : '순행 중 (Prograde)'} | 화성의 궤적이 고리 모양을 그리는 것은 지구 공전 때문입니다`;
    }

    requestAnimationFrame(animate);
  }

  gCtx.fillStyle = '#0a0e27'; gCtx.fillRect(0, 0, gW, gH);
  hCtx.fillStyle = '#0a0e27'; hCtx.fillRect(0, 0, hW, hH);
  animate();
})();

// ═══════════════════════════════════════════════════════════
// 8. Axial Tilt & Seasons Simulation
// ═══════════════════════════════════════════════════════════
(function() {
  const canvas = document.getElementById('axialTiltCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const latSlider = document.getElementById('tiltLatSlider');
  const daySlider = document.getElementById('tiltDaySlider');
  const latVal = document.getElementById('tiltLatVal');
  const dayVal = document.getElementById('tiltDayVal');
  const info = document.getElementById('tiltInfo');

  const OBLIQUITY = 23.44;
  const S0 = 1361;

  // ── Physics helpers ──
  function solarDeclination(dayOfYear) {
    return OBLIQUITY * Math.sin((dayOfYear - 81) * TAU / 365);
  }
  function maxElevation(lat, decl) {
    return 90 - Math.abs(lat - decl);
  }
  function dayLength(lat, decl) {
    const latR = lat * DEG, declR = decl * DEG;
    const cosHA = -Math.tan(latR) * Math.tan(declR);
    if (cosHA <= -1) return 24;
    if (cosHA >= 1) return 0;
    return 2 * Math.acos(cosHA) / (15 * DEG) ;
  }
  function airMass(alpha) {
    if (alpha <= 0) return Infinity;
    return 1 / Math.sin(alpha * DEG);
  }
  function irradiance(alpha) {
    if (alpha <= 0) return 0;
    const am = airMass(alpha);
    return S0 * Math.sin(alpha * DEG) * Math.pow(0.7, Math.pow(am, 0.678));
  }

  function dayName(d) {
    const months = ['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'];
    const daysInMonth = [31,28,31,30,31,30,31,31,30,31,30,31];
    let rem = d;
    for (let m = 0; m < 12; m++) {
      if (rem < daysInMonth[m]) return months[m] + ' ' + (rem + 1) + '일';
      rem -= daysInMonth[m];
    }
    return '12월 31일';
  }

  function draw() {
    const lat = parseFloat(latSlider.value);
    const day = parseInt(daySlider.value);
    latVal.textContent = (lat >= 0 ? lat.toFixed(1) + '°N' : (-lat).toFixed(1) + '°S');
    dayVal.textContent = dayName(day);

    const decl = solarDeclination(day);
    const elev = maxElevation(lat, decl);
    const elevClamped = Math.max(elev, 0);
    const dl = dayLength(lat, decl);
    const irr = irradiance(elevClamped);
    const am = elevClamped > 0 ? airMass(elevClamped) : Infinity;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0e27';
    ctx.fillRect(0, 0, W, H);

    // Stars
    for (let i = 0; i < 40; i++) {
      ctx.fillStyle = `rgba(255,255,255,${0.15 + Math.random() * 0.35})`;
      ctx.beginPath();
      ctx.arc((i * 137.5) % W, (i * 97.3) % H, 0.4 + Math.random() * 0.6, 0, TAU);
      ctx.fill();
    }

    // ── Left panel: Earth cross-section ──
    const LW = W * 0.4;
    const eCx = LW * 0.55, eCy = H * 0.5;
    const eR = Math.min(LW, H) * 0.32;
    const tiltRad = OBLIQUITY * DEG;
    // Subsolar latitude angle (declination) determines where sunlight hits equator
    const declRad = decl * DEG;

    // Sunlight from left (parallel rays)
    ctx.strokeStyle = 'rgba(255,220,100,0.25)';
    ctx.lineWidth = 1;
    for (let i = -6; i <= 6; i++) {
      const yy = eCy + i * eR * 0.28;
      ctx.beginPath();
      ctx.moveTo(0, yy);
      ctx.lineTo(eCx - eR - 5, yy);
      ctx.stroke();
      // Arrow
      ctx.beginPath();
      ctx.moveTo(eCx - eR - 5, yy);
      ctx.lineTo(eCx - eR - 12, yy - 3);
      ctx.moveTo(eCx - eR - 5, yy);
      ctx.lineTo(eCx - eR - 12, yy + 3);
      ctx.stroke();
    }

    // Sun glow (left edge)
    const sunGrad = ctx.createRadialGradient(-10, eCy, 5, -10, eCy, 60);
    sunGrad.addColorStop(0, 'rgba(255,220,100,0.4)');
    sunGrad.addColorStop(1, 'rgba(255,220,100,0)');
    ctx.fillStyle = sunGrad;
    ctx.beginPath(); ctx.arc(-10, eCy, 60, 0, TAU); ctx.fill();

    // Earth circle
    const earthGrad = ctx.createRadialGradient(eCx - eR * 0.2, eCy - eR * 0.2, eR * 0.1, eCx, eCy, eR);
    earthGrad.addColorStop(0, '#4488cc');
    earthGrad.addColorStop(0.7, '#2266aa');
    earthGrad.addColorStop(1, '#113355');
    ctx.fillStyle = earthGrad;
    ctx.beginPath(); ctx.arc(eCx, eCy, eR, 0, TAU); ctx.fill();
    ctx.strokeStyle = '#5599dd';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(eCx, eCy, eR, 0, TAU); ctx.stroke();

    // Night side (right half, since sunlight from left)
    ctx.save();
    ctx.beginPath(); ctx.arc(eCx, eCy, eR, 0, TAU); ctx.clip();
    ctx.fillStyle = 'rgba(0,0,20,0.45)';
    ctx.beginPath();
    ctx.moveTo(eCx, eCy - eR);
    ctx.quadraticCurveTo(eCx + eR * 0.3, eCy, eCx, eCy + eR);
    ctx.lineTo(eCx + eR, eCy + eR);
    ctx.lineTo(eCx + eR, eCy - eR);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // Axial tilt line
    ctx.save();
    ctx.translate(eCx, eCy);
    ctx.rotate(-tiltRad); // tilt axis
    ctx.strokeStyle = '#aaddff';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(0, -eR - 18);
    ctx.lineTo(0, eR + 18);
    ctx.stroke();
    ctx.setLineDash([]);
    // N/S labels
    ctx.fillStyle = '#aaddff';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('N', 0, -eR - 22);
    ctx.fillText('S', 0, eR + 30);
    ctx.restore();

    // Equator line
    ctx.save();
    ctx.translate(eCx, eCy);
    ctx.rotate(-tiltRad);
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(-eR, 0);
    ctx.lineTo(eR, 0);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Selected latitude point on the Earth
    // In the tilted coordinate system, latitude phi is measured from equator along the axis
    const latRad = lat * DEG;
    ctx.save();
    ctx.translate(eCx, eCy);
    ctx.rotate(-tiltRad);
    // Position on Earth surface (sunlit side = left, facing sun)
    // Show on the left side (facing the sun)
    const ptX = -eR * Math.cos(latRad);
    const ptY = -eR * Math.sin(latRad);
    // Transform back to get canvas coords
    ctx.fillStyle = '#ff4444';
    ctx.beginPath(); ctx.arc(ptX, ptY, 5, 0, TAU); ctx.fill();
    ctx.strokeStyle = '#ffaaaa';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(ptX, ptY, 5, 0, TAU); ctx.stroke();

    // Normal vector at this point (outward from center)
    const nLen = 35;
    const nx = -Math.cos(latRad);
    const ny = -Math.sin(latRad);
    ctx.strokeStyle = '#ff6666';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(ptX, ptY);
    ctx.lineTo(ptX + nx * nLen, ptY + ny * nLen);
    ctx.stroke();
    // Arrow head
    const aLen = 7;
    const nAngle = Math.atan2(ny, nx);
    ctx.beginPath();
    ctx.moveTo(ptX + nx * nLen, ptY + ny * nLen);
    ctx.lineTo(ptX + nx * nLen - aLen * Math.cos(nAngle - 0.4), ptY + ny * nLen - aLen * Math.sin(nAngle - 0.4));
    ctx.moveTo(ptX + nx * nLen, ptY + ny * nLen);
    ctx.lineTo(ptX + nx * nLen - aLen * Math.cos(nAngle + 0.4), ptY + ny * nLen - aLen * Math.sin(nAngle + 0.4));
    ctx.stroke();

    // Latitude label
    ctx.fillStyle = '#ffcccc';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    const labelText = lat >= 0 ? lat.toFixed(1) + '°N' : (-lat).toFixed(1) + '°S';
    ctx.fillText(labelText, ptX + nx * nLen + 5, ptY + ny * nLen + 3);
    ctx.restore();

    // Angle between normal and sunlight direction
    // Sunlight goes right (+x in canvas), in tilted frame it goes at angle tiltRad
    // Normal direction in canvas frame
    const cosT = Math.cos(tiltRad), sinT = Math.sin(tiltRad);
    const nxCanvas = nx * cosT - ny * (-sinT);
    const nyCanvas = nx * (-sinT) + ny * (-cosT);
    // Sunlight direction in canvas: (1, 0) pointing right
    // But solar declination shifts the effective sun direction
    // The angle between normal and sunlight = 90 - elevation
    // We already computed elevation, so just show it

    // Panel title
    ctx.fillStyle = '#88bbee';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('지구 단면 (측면)', eCx, 18);

    // Declination indicator - where the subsolar point is
    ctx.save();
    ctx.translate(eCx, eCy);
    ctx.rotate(-tiltRad);
    // Subsolar latitude line
    const subY = -eR * Math.sin(declRad);
    ctx.strokeStyle = 'rgba(255,220,100,0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(-eR, subY);
    ctx.lineTo(eR, subY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255,220,100,0.7)';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('δ=' + decl.toFixed(1) + '°', eR - 2, subY - 4);
    ctx.restore();

    // ── Right panel: Surface view ──
    const RX = LW + 20;
    const RW = W - RX - 10;
    const groundY = H * 0.72;
    const arcCx = RX + RW * 0.5;
    const arcR = Math.min(RW * 0.4, (groundY - 30) * 0.85);

    // Panel title
    ctx.fillStyle = '#88bbee';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('지표면에서 본 태양 (남중 시)', arcCx, 18);

    // Ground
    ctx.fillStyle = '#2a4020';
    ctx.fillRect(RX, groundY, RW, H - groundY);
    ctx.strokeStyle = '#5a8040';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(RX, groundY);
    ctx.lineTo(RX + RW, groundY);
    ctx.stroke();

    // Sky arc (semicircle)
    ctx.strokeStyle = 'rgba(100,150,200,0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(arcCx, groundY, arcR, Math.PI, 0);
    ctx.stroke();
    ctx.setLineDash([]);

    // Horizon labels
    ctx.fillStyle = 'rgba(150,180,200,0.5)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('E', RX + 15, groundY + 14);
    ctx.fillText('W', RX + RW - 15, groundY + 14);
    ctx.fillText('0°', RX + 8, groundY - 4);
    ctx.fillText('90°', arcCx, groundY - arcR - 4);

    // Elevation angle lines (reference)
    for (let a = 30; a <= 60; a += 30) {
      const aR = a * DEG;
      const rx = arcCx + arcR * Math.cos(Math.PI - aR);
      const ry = groundY - arcR * Math.sin(aR);
      ctx.strokeStyle = 'rgba(100,150,200,0.15)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(arcCx, groundY);
      ctx.lineTo(rx, ry);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(150,180,200,0.4)';
      ctx.font = '9px sans-serif';
      ctx.fillText(a + '°', rx + (a < 60 ? -12 : 8), ry + (a < 60 ? -2 : 8));
    }

    if (elevClamped > 0) {
      const elevRad = elevClamped * DEG;

      // Sun position on arc
      const sunArcX = arcCx - arcR * Math.cos(elevRad);
      const sunArcY = groundY - arcR * Math.sin(elevRad);

      // Sun glow
      const sGrad = ctx.createRadialGradient(sunArcX, sunArcY, 3, sunArcX, sunArcY, 22);
      sGrad.addColorStop(0, 'rgba(255,240,150,0.9)');
      sGrad.addColorStop(0.5, 'rgba(255,200,50,0.4)');
      sGrad.addColorStop(1, 'rgba(255,200,50,0)');
      ctx.fillStyle = sGrad;
      ctx.beginPath(); ctx.arc(sunArcX, sunArcY, 22, 0, TAU); ctx.fill();
      ctx.fillStyle = '#fff8d0';
      ctx.beginPath(); ctx.arc(sunArcX, sunArcY, 8, 0, TAU); ctx.fill();

      // Elevation angle arc (small arc from horizon to sun)
      ctx.strokeStyle = '#ffdd66';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(arcCx, groundY, 40, elevRad - Math.PI, -Math.PI, true);
      ctx.stroke();
      // Angle label (midpoint of arc, above ground to the left)
      const labelMid = elevRad / 2 - Math.PI;
      ctx.fillStyle = '#ffdd66';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('α=' + elevClamped.toFixed(1) + '°', arcCx + 55 * Math.cos(labelMid), groundY + 55 * Math.sin(labelMid));

      // Line from ground to sun
      ctx.strokeStyle = 'rgba(255,220,100,0.6)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(arcCx, groundY);
      ctx.lineTo(sunArcX, sunArcY);
      ctx.stroke();

      // ── Beam width diagram ──
      // Show how same-width beam covers more ground at lower angle
      // Rays come from the LEFT (sun side) going down-right to ground
      const beamW = 50; // beam width in pixels (perpendicular to ray)
      const sinE = Math.sin(elevRad);
      const cosE = Math.cos(elevRad);
      const groundCoverage = beamW / sinE;
      const maxCoverage = RW * 0.42;
      const dispCoverage = Math.min(groundCoverage, maxCoverage);
      const dispBeamW = dispCoverage * sinE;

      // Ray direction: from upper-left (sun) to lower-right (ground)
      const rdx = cosE;   // rightward
      const rdy = sinE;   // downward (canvas y-down)
      // Perpendicular to ray (for beam width)
      const pdx = -sinE;  // beam width direction
      const pdy = cosE;

      // Ground hit point (center of beam on ground)
      const hitX = RX + RW * 0.55;
      const hitY = groundY;
      const rayLen = 80;
      const halfW = dispBeamW / 2;

      ctx.strokeStyle = 'rgba(255,220,100,0.7)';
      ctx.lineWidth = 1.5;
      // Ray 1 (upper edge of beam)
      const r1_gx = hitX - pdx * halfW;  // ground hit point
      const r1_gy = hitY - pdy * halfW;
      const r1_sx = r1_gx - rdx * rayLen; // source end (toward sun)
      const r1_sy = r1_gy - rdy * rayLen;
      ctx.beginPath(); ctx.moveTo(r1_sx, r1_sy); ctx.lineTo(r1_gx, r1_gy); ctx.stroke();
      // Arrowhead
      ctx.beginPath();
      ctx.moveTo(r1_gx, r1_gy);
      ctx.lineTo(r1_gx - 7 * rdx - 3 * pdx, r1_gy - 7 * rdy - 3 * pdy);
      ctx.moveTo(r1_gx, r1_gy);
      ctx.lineTo(r1_gx - 7 * rdx + 3 * pdx, r1_gy - 7 * rdy + 3 * pdy);
      ctx.stroke();
      // Ray 2 (lower edge of beam)
      const r2_gx = hitX + pdx * halfW;
      const r2_gy = hitY + pdy * halfW;
      const r2_sx = r2_gx - rdx * rayLen;
      const r2_sy = r2_gy - rdy * rayLen;
      ctx.beginPath(); ctx.moveTo(r2_sx, r2_sy); ctx.lineTo(r2_gx, r2_gy); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(r2_gx, r2_gy);
      ctx.lineTo(r2_gx - 7 * rdx - 3 * pdx, r2_gy - 7 * rdy - 3 * pdy);
      ctx.moveTo(r2_gx, r2_gy);
      ctx.lineTo(r2_gx - 7 * rdx + 3 * pdx, r2_gy - 7 * rdy + 3 * pdy);
      ctx.stroke();

      // Beam width label (perpendicular to rays, at source end)
      ctx.fillStyle = 'rgba(255,220,100,0.6)';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('빔 폭 1m', (r1_sx + r2_sx) / 2 - 10, (r1_sy + r2_sy) / 2 - 8);

      // Ground coverage indicator
      const gcLeft = hitX - dispCoverage / 2;
      const gcRight = hitX + dispCoverage / 2;
      ctx.strokeStyle = '#ff8844';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(gcLeft, groundY);
      ctx.lineTo(gcRight, groundY);
      ctx.stroke();
      // Bracket ends
      ctx.beginPath();
      ctx.moveTo(gcLeft, groundY - 4); ctx.lineTo(gcLeft, groundY + 6);
      ctx.moveTo(gcRight, groundY - 4); ctx.lineTo(gcRight, groundY + 6);
      ctx.stroke();

      // Coverage label
      const coverageVal = (1 / sinE).toFixed(2);
      ctx.fillStyle = '#ff8844';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('조사 면적: ' + coverageVal + 'm', (gcLeft + gcRight) / 2, groundY + 20);
      ctx.font = '10px sans-serif';
      ctx.fillText('(= 1/sin α)', (gcLeft + gcRight) / 2, groundY + 33);

      // Fill beam area with translucent yellow
      ctx.fillStyle = 'rgba(255,220,100,0.08)';
      ctx.beginPath();
      ctx.moveTo(r1_sx, r1_sy); ctx.lineTo(r1_gx, r1_gy);
      ctx.lineTo(r2_gx, r2_gy); ctx.lineTo(r2_sx, r2_sy);
      ctx.closePath();
      ctx.fill();
    } else {
      // Sun below horizon
      ctx.fillStyle = '#ff6666';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('태양이 뜨지 않음 (극야)', arcCx, groundY - arcR * 0.5);
    }

    // ── Info display ──
    const elevStr = elevClamped > 0 ? elevClamped.toFixed(1) + '°' : '뜨지 않음';
    const amStr = am < 100 ? am.toFixed(2) : '∞';
    const irrStr = irr > 0 ? irr.toFixed(0) + ' W/m²' : '0 W/m²';
    const dlStr = dl.toFixed(1) + '시간';
    info.innerHTML =
      '<strong>태양 적위(δ):</strong> ' + decl.toFixed(2) + '° | ' +
      '<strong>최대 고도(α):</strong> ' + elevStr + ' | ' +
      '<strong>Air Mass:</strong> ' + amStr + ' | ' +
      '<strong>최대 일사량:</strong> ' + irrStr + ' | ' +
      '<strong>낮 길이:</strong> ' + dlStr;
  }

  latSlider.addEventListener('input', draw);
  daySlider.addEventListener('input', draw);
  draw();
})();

// ═══════════════════════════════════════════════════════════
// 9. Celestial Sphere — Sun's Diurnal Path (Oblique View)
// ═══════════════════════════════════════════════════════════
(function() {
  const canvas = document.getElementById('celestialSphereCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const latSlider = document.getElementById('csLatSlider');
  const daySlider = document.getElementById('csDaySlider');
  const latVal = document.getElementById('csLatVal');
  const dayVal = document.getElementById('csDayVal');
  const playBtn = document.getElementById('csPlayBtn');
  const info = document.getElementById('csInfo');

  const OBLIQ = 23.44;
  // View: oblique from SW at 30° elevation, 45° azimuthal offset
  // South → upper-left, North → lower-right
  const viewElev = 30 * DEG;
  const viewAz = -135 * DEG; // scene rotation so south=upper-left
  const cosV = Math.cos(viewElev), sinV = Math.sin(viewElev);
  const cosA = Math.cos(viewAz), sinA = Math.sin(viewAz);
  const R = Math.min(W, H) * 0.40; // sphere radius in pixels (+10% zoom)
  const cx = W * 0.5, cy = H * 0.52;

  let hourAngle = -180; // degrees, -180=midnight, 0=noon, +180=midnight
  let playing = true;
  let animId = null;
  let shadowTrail = []; // stores {x, y} of shadow tip positions

  function solarDecl(day) {
    return OBLIQ * Math.sin((day - 81) * TAU / 365);
  }
  function dayName(d) {
    const months = ['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'];
    const dim = [31,28,31,30,31,30,31,31,30,31,30,31];
    let rem = d;
    for (let m = 0; m < 12; m++) {
      if (rem < dim[m]) return months[m] + ' ' + (rem + 1) + '일';
      rem -= dim[m];
    }
    return '12월 31일';
  }

  // Project 3D point (azimuth A from north CW, altitude α) onto 2D canvas
  // 3D: x=east, y=north, z=up → rotate by viewAz around z → tilt by viewElev around x
  function project(azDeg, altDeg) {
    const az = azDeg * DEG, alt = altDeg * DEG;
    const x3 = Math.cos(alt) * Math.sin(az);   // east
    const y3 = Math.cos(alt) * Math.cos(az);   // north
    const z3 = Math.sin(alt);                    // up
    // 1) Rotate around z-axis by viewAz (azimuthal rotation)
    const xr = x3 * cosA - y3 * sinA;
    const yr = x3 * sinA + y3 * cosA;
    // 2) Rotate around x-axis by viewElev (tilt forward)
    const yr2 = yr * cosV - z3 * sinV;
    const zr = yr * sinV + z3 * cosV;
    // Parallel projection
    return {
      x: cx + xr * R,
      y: cy - zr * R,
      depth: yr2
    };
  }

  // Sun position at given hour angle
  function sunPos(lat, decl, haDeg) {
    const phi = lat * DEG, delta = decl * DEG, ha = haDeg * DEG;
    const sinAlt = Math.sin(phi) * Math.sin(delta) + Math.cos(phi) * Math.cos(delta) * Math.cos(ha);
    const alt = Math.asin(Math.max(-1, Math.min(1, sinAlt)));
    const cosAz = (Math.sin(delta) - Math.sin(phi) * sinAlt) / (Math.cos(phi) * Math.cos(alt) + 1e-10);
    let az = Math.acos(Math.max(-1, Math.min(1, cosAz)));
    if (Math.sin(ha) > 0) az = TAU - az; // afternoon = west
    return { azDeg: az / DEG, altDeg: alt / DEG };
  }

  function sunriseHA(lat, decl) {
    const phi = lat * DEG, delta = decl * DEG;
    const cosH = -Math.tan(phi) * Math.tan(delta);
    if (cosH <= -1) return 180; // never sets (midnight sun)
    if (cosH >= 1) return 0;   // never rises (polar night)
    return Math.acos(cosH) / DEG;
  }

  function draw() {
    const lat = parseFloat(latSlider.value);
    const day = parseInt(daySlider.value);
    latVal.textContent = (lat >= 0 ? lat.toFixed(1) + '°N' : (-lat).toFixed(1) + '°S');
    dayVal.textContent = dayName(day);

    const decl = solarDecl(day);
    const haMax = sunriseHA(lat, decl);

    ctx.clearRect(0, 0, W, H);

    // Sky gradient
    const skyGrad = ctx.createLinearGradient(0, 0, 0, H);
    skyGrad.addColorStop(0, '#0a1530');
    skyGrad.addColorStop(0.5, '#1a2a50');
    skyGrad.addColorStop(1, '#2a3a60');
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, W, H);

    // ── Horizon (projected as rotated ellipse) ──
    // Collect horizon points
    const horizPts = [];
    for (let a = 0; a <= 360; a += 2) {
      horizPts.push(project(a, 0));
    }
    // Ground fill below horizon
    ctx.fillStyle = '#2e4028';
    ctx.beginPath();
    ctx.moveTo(horizPts[0].x, horizPts[0].y);
    for (let i = 1; i < horizPts.length; i++) ctx.lineTo(horizPts[i].x, horizPts[i].y);
    // Close with bottom of canvas
    const lastH = horizPts[horizPts.length - 1];
    // Find the lowest point to extend ground fill
    let maxY = 0;
    for (const p of horizPts) if (p.y > maxY) maxY = p.y;
    const extY = Math.max(maxY + 50, H);
    // Trace bottom edge
    ctx.lineTo(W, extY); ctx.lineTo(0, extY);
    ctx.closePath();
    ctx.fill();

    // Horizon line
    ctx.strokeStyle = 'rgba(100,180,100,0.5)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(horizPts[0].x, horizPts[0].y);
    for (let i = 1; i < horizPts.length; i++) ctx.lineTo(horizPts[i].x, horizPts[i].y);
    ctx.closePath();
    ctx.stroke();

    // ── Cardinal directions ──
    const dirs = [
      { az: 0, label: 'N(북)', color: '#ff6666' },
      { az: 90, label: 'E(동)', color: '#88bbcc' },
      { az: 180, label: 'S(남)', color: '#88bbcc' },
      { az: 270, label: 'W(서)', color: '#88bbcc' }
    ];
    for (const d of dirs) {
      const p = project(d.az, 0);
      ctx.fillStyle = d.color;
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(d.label, p.x, p.y + 16);
      // Small tick mark
      ctx.strokeStyle = d.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y - 4);
      ctx.lineTo(p.x, p.y + 4);
      ctx.stroke();
    }

    // ── Ground cross lines (N-S, E-W) ──
    const groundLineLen = 0.7; // fraction of R on the ground plane
    // N-S line
    ctx.strokeStyle = 'rgba(220,180,140,0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    const gN = project(0, 0), gS = project(180, 0);
    // Interpolate from S to N through center (on ground)
    const gSteps = 30;
    for (let i = 0; i <= gSteps; i++) {
      const t = i / gSteps;
      const px = gS.x + (gN.x - gS.x) * t * groundLineLen + (cx - gS.x) * (1 - groundLineLen);
      const py = gS.y + (gN.y - gS.y) * t * groundLineLen + (cy - gS.y) * (1 - groundLineLen);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();
    // E-W line
    ctx.beginPath();
    const gE = project(90, 0), gW = project(270, 0);
    for (let i = 0; i <= gSteps; i++) {
      const t = i / gSteps;
      const px = gW.x + (gE.x - gW.x) * t * groundLineLen + (cx - gW.x) * (1 - groundLineLen);
      const py = gW.y + (gE.y - gW.y) * t * groundLineLen + (cy - gW.y) * (1 - groundLineLen);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    // ── Celestial meridian arc (N-zenith-S) ──
    ctx.strokeStyle = 'rgba(150,150,200,0.2)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    let first = true;
    for (let alt = -5; alt <= 90; alt += 2) {
      const p = project(0, alt); // north meridian
      if (first) { ctx.moveTo(p.x, p.y); first = false; }
      else ctx.lineTo(p.x, p.y);
    }
    for (let alt = 90; alt >= -5; alt -= 2) {
      const p = project(180, alt); // south meridian
      ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Zenith label
    const zen = project(0, 90);
    ctx.fillStyle = 'rgba(200,200,255,0.5)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('천정(Z)', zen.x, zen.y - 8);

    // ── Altitude reference arcs (30°, 60°) ──
    ctx.strokeStyle = 'rgba(150,180,220,0.12)';
    ctx.lineWidth = 0.5;
    ctx.setLineDash([2, 4]);
    for (const refAlt of [30, 60]) {
      ctx.beginPath();
      first = true;
      for (let az = 0; az <= 360; az += 3) {
        const p = project(az, refAlt);
        if (first) { ctx.moveTo(p.x, p.y); first = false; }
        else ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
      const lp = project(90, refAlt);
      ctx.fillStyle = 'rgba(150,180,220,0.3)';
      ctx.font = '9px sans-serif';
      ctx.fillText(refAlt + '°', lp.x + 10, lp.y);
    }
    ctx.setLineDash([]);

    // ── Sun's diurnal path (full arc, dashed below horizon) ──
    const pathPts = [];
    for (let ha = -180; ha <= 180; ha += 1) {
      const sp = sunPos(lat, decl, ha);
      const pp = project(sp.azDeg, sp.altDeg);
      pathPts.push({ ha, alt: sp.altDeg, px: pp.x, py: pp.y, depth: pp.depth });
    }

    // Below-horizon portion (dashed)
    ctx.strokeStyle = 'rgba(255,200,80,0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    first = true;
    for (const pt of pathPts) {
      if (pt.alt < 0) {
        if (first) { ctx.moveTo(pt.px, pt.py); first = false; }
        else ctx.lineTo(pt.px, pt.py);
      } else {
        first = true;
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Above-horizon portion (solid, glowing)
    ctx.strokeStyle = 'rgba(255,200,80,0.6)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    first = true;
    for (const pt of pathPts) {
      if (pt.alt >= 0) {
        if (first) { ctx.moveTo(pt.px, pt.py); first = false; }
        else ctx.lineTo(pt.px, pt.py);
      } else {
        first = true;
      }
    }
    ctx.stroke();

    // Sunrise/sunset markers
    if (haMax > 0 && haMax < 180) {
      const rise = sunPos(lat, decl, -haMax);
      const set = sunPos(lat, decl, haMax);
      const rp = project(rise.azDeg, Math.max(rise.altDeg, 0));
      const sp2 = project(set.azDeg, Math.max(set.altDeg, 0));
      // Rise marker
      ctx.fillStyle = '#ff8844';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('일출', rp.x, rp.y + 14);
      // Set marker
      ctx.fillStyle = '#cc6644';
      ctx.fillText('일몰', sp2.x, sp2.y + 14);
    }

    // ── Sun position (current hour angle) ──
    const sunNow = sunPos(lat, decl, hourAngle);
    const sunProj = project(sunNow.azDeg, sunNow.altDeg);
    const isAbove = sunNow.altDeg >= 0;

    if (isAbove) {
      // Sun glow
      const sGrad = ctx.createRadialGradient(sunProj.x, sunProj.y, 3, sunProj.x, sunProj.y, 25);
      sGrad.addColorStop(0, 'rgba(255,240,150,0.9)');
      sGrad.addColorStop(0.4, 'rgba(255,200,50,0.3)');
      sGrad.addColorStop(1, 'rgba(255,200,50,0)');
      ctx.fillStyle = sGrad;
      ctx.beginPath(); ctx.arc(sunProj.x, sunProj.y, 25, 0, TAU); ctx.fill();
      ctx.fillStyle = '#fff8d0';
      ctx.beginPath(); ctx.arc(sunProj.x, sunProj.y, 7, 0, TAU); ctx.fill();
    } else {
      // Below horizon: dim indicator
      ctx.fillStyle = 'rgba(255,200,100,0.3)';
      ctx.beginPath(); ctx.arc(sunProj.x, sunProj.y, 5, 0, TAU); ctx.fill();
      ctx.strokeStyle = 'rgba(255,200,100,0.2)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(sunProj.x, sunProj.y, 5, 0, TAU); ctx.stroke();
    }

    // ── Gnomon (vertical pole at center) and shadow ──
    // The gnomon is a vertical stick at the observer's position (center of horizon)
    const poleH = 0.22; // pole height in sphere-radius units
    const origin = project(0, 0); // project won't work for center; use cx,cy directly
    // Pole base = center of ground plane = (cx, cy) approximately
    // But we need the projected center. Project a point at alt=0 averaged:
    const baseX = cx, baseY = cy;
    // Pole top: a point straight up from center by poleH
    // In 3D: (0, 0, poleH) → apply viewAz rotation (no effect on z-only), then tilt
    const poleTopZ = poleH * cosV;
    const poleTopY = poleH * sinV; // this becomes depth
    const topX = baseX; // no x shift (pole is vertical, centered)
    const topY = baseY - poleTopZ * R;

    // Draw shadow first (behind pole)
    if (isAbove && sunNow.altDeg > 0.5) {
      // Shadow direction: opposite to sun's azimuth, projected on ground
      const sunAzRad = sunNow.azDeg * DEG;
      const sunAltRad = sunNow.altDeg * DEG;
      const shadowLen = poleH / Math.tan(sunAltRad); // in sphere units
      const clampedLen = Math.min(shadowLen, 0.8); // cap for very low sun

      // Shadow tip in 3D ground plane: opposite to sun direction
      // Sun at azimuth A → shadow points to azimuth A+180°
      const shadAz = sunAzRad + Math.PI;
      const shadX3 = clampedLen * Math.sin(shadAz); // east
      const shadY3 = clampedLen * Math.cos(shadAz); // north
      // Apply viewAz and viewElev rotation (z=0 for ground)
      const sxr = shadX3 * cosA - shadY3 * sinA;
      const syr = shadX3 * sinA + shadY3 * cosA;
      const syr2 = syr * cosV;
      const szr = syr * sinV;
      const shadTipX = baseX + sxr * R;
      const shadTipY = baseY - szr * R;

      // Store shadow tip in trail
      shadowTrail.push({ x: shadTipX, y: shadTipY });
      if (shadowTrail.length > 600) shadowTrail.shift();

      // Draw shadow trail (daytime segments only, broken by null markers)
      if (shadowTrail.length > 1) {
        ctx.strokeStyle = 'rgba(20,20,20,0.8)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        let penDown = false;
        for (let i = 0; i < shadowTrail.length; i++) {
          if (shadowTrail[i] === null) { penDown = false; continue; }
          if (!penDown) { ctx.moveTo(shadowTrail[i].x, shadowTrail[i].y); penDown = true; }
          else ctx.lineTo(shadowTrail[i].x, shadowTrail[i].y);
        }
        ctx.stroke();
      }

      // Shadow line (black, same thickness as pole)
      ctx.strokeStyle = '#111';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(baseX, baseY);
      ctx.lineTo(shadTipX, shadTipY);
      ctx.stroke();

      // Shadow tip dot
      ctx.fillStyle = '#111';
      ctx.beginPath();
      ctx.arc(shadTipX, shadTipY, 3, 0, TAU);
      ctx.fill();
    } else {
      // Sun below horizon: insert null break so trail segments don't connect
      if (shadowTrail.length > 0 && shadowTrail[shadowTrail.length - 1] !== null) {
        shadowTrail.push(null);
      }
    }

    // Pole body
    ctx.strokeStyle = '#cc8844';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(baseX, baseY);
    ctx.lineTo(topX, topY);
    ctx.stroke();
    // Pole top cap
    ctx.fillStyle = '#dd9955';
    ctx.beginPath();
    ctx.arc(topX, topY, 2.5, 0, TAU);
    ctx.fill();
    // Pole base
    ctx.fillStyle = '#aa7733';
    ctx.beginPath();
    ctx.arc(baseX, baseY, 3, 0, TAU);
    ctx.fill();

    // ── Meridian transit line (noon line from south horizon to zenith) ──
    const noonSun = sunPos(lat, decl, 0);
    if (noonSun.altDeg > 0) {
      const noonP = project(noonSun.azDeg, noonSun.altDeg);
      ctx.strokeStyle = 'rgba(255,220,100,0.2)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(noonP.x, cy); // horizon at south
      ctx.lineTo(noonP.x, noonP.y);
      ctx.stroke();
      ctx.setLineDash([]);
      // Max altitude label
      ctx.fillStyle = 'rgba(255,220,100,0.6)';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('남중고도 ' + noonSun.altDeg.toFixed(1) + '°', noonP.x + 10, noonP.y - 2);
    }

    // ── Time info ──
    // Convert hour angle to clock time (HA=0 → 12:00, HA=-90 → 06:00)
    const clockH = ((hourAngle + 180) / 15 + 0) % 24;
    const hh = Math.floor(clockH);
    const mm = Math.floor((clockH - hh) * 60);
    const timeStr = String(hh).padStart(2, '0') + ':' + String(mm).padStart(2, '0');
    const dlHours = (2 * haMax / 15);

    ctx.fillStyle = '#ddd';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('⏱ ' + timeStr, 14, 24);

    ctx.fillStyle = isAbove ? '#ffdd88' : '#667';
    ctx.font = '12px sans-serif';
    ctx.fillText(isAbove ? '☀ 낮' : '🌙 밤', 14, 42);

    // Alt/Az display
    ctx.fillStyle = 'rgba(200,210,230,0.7)';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('고도: ' + sunNow.altDeg.toFixed(1) + '°', W - 12, 24);
    ctx.fillText('방위: ' + sunNow.azDeg.toFixed(1) + '°', W - 12, 40);

    // Info bar
    info.innerHTML =
      '<strong>태양 적위(δ):</strong> ' + decl.toFixed(1) + '° | ' +
      '<strong>남중고도:</strong> ' + noonSun.altDeg.toFixed(1) + '° | ' +
      '<strong>낮 길이:</strong> ' + dlHours.toFixed(1) + '시간 | ' +
      '<strong>시각:</strong> ' + timeStr;
  }

  function animate() {
    if (!playing) return;
    hourAngle += 0.8; // speed
    if (hourAngle > 180) hourAngle -= 360;
    draw();
    animId = requestAnimationFrame(animate);
  }

  latSlider.addEventListener('input', function() { shadowTrail = []; draw(); });
  daySlider.addEventListener('input', function() { shadowTrail = []; draw(); });
  playBtn.addEventListener('click', function() {
    playing = !playing;
    playBtn.textContent = playing ? '⏸ 일시정지' : '▶ 재생';
    if (playing) { shadowTrail = []; animate(); }
  });

  draw();
  animate();
})();

// ═══════════════════════════════════════════════════════════
// KaTeX auto-render
// ═══════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', function() {
  if (typeof renderMathInElement === 'function') {
    renderMathInElement(document.body, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '\\(', right: '\\)', display: false}
      ],
      throwOnError: false
    });
  }
});
