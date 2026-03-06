# -*- coding: utf-8 -*-
"""Build celestial mechanics interactive textbook HTML."""
import sys, os

sys.path.insert(0, r'C:\Users\jehyu\Arbeitplatz\claude_functions')
from web_report_template import ReportTemplate, CSS, JS

OUTPUT_DIR = r'C:\Users\jehyu\Arbeitplatz\claude_output\celestial-mechanics\web_report'

# ═══════════════════════════════════════════════════════════
# Read external files
# ═══════════════════════════════════════════════════════════
with open(os.path.join(OUTPUT_DIR, 'styles.css'), 'r', encoding='utf-8') as f:
    CUSTOM_CSS = f.read()
with open(os.path.join(OUTPUT_DIR, 'simulations.js'), 'r', encoding='utf-8') as f:
    SIMULATION_JS = f.read()

# ═══════════════════════════════════════════════════════════
# KaTeX CDN
# ═══════════════════════════════════════════════════════════
KATEX_HEAD = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
"""

# ═══════════════════════════════════════════════════════════
# Scientist portraits (Wikimedia Commons, public domain)
# ═══════════════════════════════════════════════════════════
PORTRAITS = {
    'ptolemy': 'https://upload.wikimedia.org/wikipedia/commons/0/0e/Ptolemy_16century.jpg',
    'copernicus': 'https://upload.wikimedia.org/wikipedia/commons/f/f2/Nikolaus_Kopernikus.jpg',
    'kepler': 'https://upload.wikimedia.org/wikipedia/commons/d/d4/Johannes_Kepler_1610.jpg',
    'newton': 'https://upload.wikimedia.org/wikipedia/commons/3/3b/Portrait_of_Sir_Isaac_Newton%2C_1689.jpg',
    'galileo': 'https://upload.wikimedia.org/wikipedia/commons/d/d4/Justus_Sustermans_-_Portrait_of_Galileo_Galilei%2C_1636.jpg',
    'brahe': 'https://upload.wikimedia.org/wikipedia/commons/2/2b/Tycho_Brahe.JPG',
}

def scientist_card(key, name, years, desc):
    url = PORTRAITS.get(key, '')
    return f'''<div class="scientist-card">
  <img src="{url}" alt="{name}" loading="lazy" onerror="this.style.display='none'">
  <div class="scientist-info">
    <h4>{name}</h4>
    <div class="years">{years}</div>
    <p>{desc}</p>
  </div>
</div>'''

# ═══════════════════════════════════════════════════════════
# TOC
# ═══════════════════════════════════════════════════════════
toc_items = [
    ('ch1', '1. 천체역학의 역사', [
        ('sec1-1', '1.1 천동설: 프톨레마이오스'),
        ('sec1-2', '1.2 코페르니쿠스의 혁명'),
        ('sec1-3', '1.3 튀코 브라헤와 케플러'),
        ('sec1-4', '1.4 갈릴레오의 관측'),
        ('sec1-5', '1.5 뉴턴의 종합'),
    ]),
    ('ch2', '2. 만유인력의 법칙', [
        ('sec2-1', '2.1 뉴턴의 만유인력'),
        ('sec2-2', '2.2 만유인력 시뮬레이션'),
        ('sec2-3', '2.3 우주 탈출속도 시뮬레이션'),
    ]),
    ('ch3', '3. 공전과 자전', [
        ('sec3-1', '3.1 지구의 공전과 자전'),
        ('sec3-2', '3.2 달의 공전'),
        ('sec3-3', '3.3 궤도 시뮬레이션'),
    ]),
    ('ch4', '4. 연중 일사량 변화', [
        ('sec4-1', '4.1 지축 경사와 계절'),
        ('sec4-2', '4.2 일사량 시뮬레이션'),
    ]),
    ('ch5', '5. 달의 위상과 운동', [
        ('sec5-1', '5.1 삭망월과 항성월'),
        ('sec5-2', '5.2 달의 위상 시뮬레이션'),
    ]),
    ('ch6', '6. 일식과 월식', [
        ('sec6-1', '6.1 식 현상의 원리'),
        ('sec6-2', '6.2 일식/월식 시뮬레이션'),
        ('sec6-3', '6.3 서울에서 본 일식/월식'),
    ]),
    ('ch7', '7. 천동설 vs 지동설', [
        ('sec7-1', '7.1 두 모델의 비교'),
        ('sec7-2', '7.2 역행운동 시뮬레이션'),
    ]),
]

# ═══════════════════════════════════════════════════════════
# Chapters
# ═══════════════════════════════════════════════════════════

ch1 = f'''<div class="chapter" id="ch1">
<div class="chapter-banner" style="background:linear-gradient(135deg,#1a1a3e,#2a1a4e,#1a2a4e);height:200px;display:flex;align-items:center;justify-content:center;">
  <div style="text-align:center;color:#fff;">
    <div style="font-size:40px;margin-bottom:8px;">&#x2604; &#x2609; &#x263D;</div>
    <div style="font-size:14px;opacity:0.7;">From Ancient Skies to Universal Gravitation</div>
  </div>
</div>
<div class="chapter-body">
<h1 class="chapter-title" id="ch1-title">제1장. 천체역학의 역사</h1>
<p>인류는 밤하늘의 별과 행성의 움직임을 관찰하며 우주의 구조를 이해하려 했습니다. 천동설(geocentric model)에서 지동설(heliocentric model)로의 전환은 과학 혁명의 핵심이었으며, 이 과정에서 탄생한 물리 법칙들은 오늘날 우주 탐사의 기반이 되었습니다.</p>

<div class="era-header"><h3>고대 ~ 중세: 천동설의 시대</h3></div>

<h2 class="section-title" id="sec1-1">1.1 천동설: 프톨레마이오스의 우주</h2>
{scientist_card('ptolemy', '클라우디오스 프톨레마이오스 (Claudius Ptolemy)', 'c. 100 ~ c. 170 AD, 알렉산드리아',
  '고대 그리스-로마의 천문학자. 저서 《알마게스트(Almagest)》에서 지구중심 우주 모델을 체계화했습니다. 주전원(epicycle)과 이심원(eccentric), 등화점(equant)을 도입하여 행성의 역행운동을 설명했으며, 이 모델은 약 1400년간 서양 천문학의 표준이었습니다.')}

<p>프톨레마이오스의 천동설(Ptolemaic system)에서 지구는 우주의 중심에 정지해 있고, 태양, 달, 행성, 항성이 지구 주위를 공전합니다. 행성이 때때로 하늘에서 뒤로 움직이는 것처럼 보이는 <strong>역행운동(retrograde motion)</strong>을 설명하기 위해, 행성이 큰 원(주전원, deferent) 위의 작은 원(이심원, epicycle) 위에서 움직인다고 가정했습니다.</p>

<div class="info-box">
<div class="box-title">[프톨레마이오스 모델의 핵심 요소]</div>
<p><strong>주전원(Deferent)</strong>: 지구를 중심으로 한 큰 원궤도</p>
<p><strong>이심원(Epicycle)</strong>: 주전원 위의 점을 중심으로 한 작은 원궤도</p>
<p><strong>등화점(Equant)</strong>: 지구에서 떨어진 점으로, 이 점에서 보면 행성이 등속으로 움직이는 것처럼 보임</p>
</div>

<div class="era-header"><h3>16세기: 지동설 혁명</h3></div>

<h2 class="section-title" id="sec1-2">1.2 코페르니쿠스의 혁명</h2>
{scientist_card('copernicus', '니콜라우스 코페르니쿠스 (Nicolaus Copernicus)', '1473 ~ 1543, 폴란드',
  '폴란드의 천문학자이자 성직자. 1543년 출간된 《천구의 회전에 관하여(De Revolutionibus Orbium Coelestium)》에서 태양중심설을 제안했습니다. 지구가 태양 주위를 공전한다는 이 혁명적 주장은 당시 교회와 학계의 격렬한 반대에 부딪혔으나, 현대 천문학의 출발점이 되었습니다.')}

<p>코페르니쿠스는 태양을 우주의 중심에 놓고 지구를 포함한 행성들이 태양 주위를 공전한다고 주장했습니다. 이 모델은 역행운동을 자연스럽게 설명합니다 — 지구가 외행성을 "추월"할 때, 그 행성이 일시적으로 뒤로 가는 것처럼 보이는 것입니다. 하지만 코페르니쿠스는 여전히 원 궤도를 사용했기 때문에, 정밀한 예측을 위해 일부 주전원이 필요했습니다.</p>

<h2 class="section-title" id="sec1-3">1.3 튀코 브라헤와 요하네스 케플러</h2>
{scientist_card('brahe', '튀코 브라헤 (Tycho Brahe)', '1546 ~ 1601, 덴마크',
  '육안 관측 시대 최고의 천문학자. 우라니보르그 천문대에서 20년간 축적한 정밀 관측 데이터는 케플러의 법칙 발견에 결정적 역할을 했습니다. 지구는 정지하고 행성은 태양을, 태양은 지구를 돈다는 절충적 모델(Tychonic system)을 제안하기도 했습니다.')}

{scientist_card('kepler', '요하네스 케플러 (Johannes Kepler)', '1571 ~ 1630, 독일',
  '독일의 천문학자이자 수학자. 브라헤의 정밀 관측 데이터를 분석하여 행성운동의 세 법칙을 발견했습니다. 특히 행성 궤도가 원이 아닌 타원이라는 제1법칙의 발견은 천문학의 패러다임을 완전히 바꾸었습니다.')}

<div class="math-block">
<p><strong>케플러의 행성운동 법칙</strong></p>
<p><strong>제1법칙 (타원 궤도의 법칙)</strong>: 행성은 태양을 한 초점으로 하는 타원 궤도를 그린다.</p>
<p>$$ r = \\frac{{a(1-e^2)}}{{1 + e\\cos\\theta}} $$</p>
<p>여기서 \\(a\\)는 장반경, \\(e\\)는 이심률, \\(\\theta\\)는 진근점이각입니다.</p>
<p><strong>제2법칙 (면적 속도 일정의 법칙)</strong>: 행성과 태양을 잇는 선분이 같은 시간에 같은 면적을 쓸고 지나간다.</p>
<p>$$ \\frac{{dA}}{{dt}} = \\frac{{L}}{{2m}} = \\text{{const.}} $$</p>
<p><strong>제3법칙 (조화의 법칙)</strong>: 행성의 공전주기의 제곱은 궤도 장반경의 세제곱에 비례한다.</p>
<p>$$ T^2 = \\frac{{4\\pi^2}}{{GM}} a^3 $$</p>
</div>

<h2 class="section-title" id="sec1-4">1.4 갈릴레오의 관측적 증거</h2>
{scientist_card('galileo', '갈릴레오 갈릴레이 (Galileo Galilei)', '1564 ~ 1642, 이탈리아',
  '이탈리아의 물리학자이자 천문학자. 1609년 망원경을 천체 관측에 처음 사용하여 목성의 위성 4개(갈릴레이 위성), 금성의 위상 변화, 달의 산과 크레이터, 태양의 흑점 등을 발견했습니다. 이 관측들은 지동설의 강력한 관측적 증거가 되었습니다.')}

<div class="info-box">
<div class="box-title">[갈릴레오의 주요 관측 증거]</div>
<p><strong>목성의 위성</strong>: 지구가 아닌 다른 천체를 공전하는 물체의 존재 → 모든 것이 지구 중심으로 도는 것은 아님</p>
<p><strong>금성의 위상 변화</strong>: 천동설로는 설명 불가한 "보름금성" 관측 → 금성이 태양 주위를 돈다는 증거</p>
<p><strong>태양의 흑점</strong>: 태양도 자전함 → 천체의 완전성에 대한 반론</p>
<p><strong>달 표면의 요철</strong>: 천체가 완벽한 구가 아님 → 아리스토텔레스 천문학 반박</p>
</div>

<div class="era-header"><h3>17세기: 뉴턴의 대통합</h3></div>

<h2 class="section-title" id="sec1-5">1.5 뉴턴의 종합: 만유인력</h2>
{scientist_card('newton', '아이작 뉴턴 (Isaac Newton)', '1643 ~ 1727, 영국',
  '영국의 물리학자이자 수학자. 1687년 출간된 《프린키피아 마테마티카(Principia Mathematica)》에서 운동의 3법칙과 만유인력의 법칙을 발표했습니다. 사과가 떨어지는 것과 달이 지구를 도는 것이 같은 힘에 의한 현상임을 밝혀, 지상 역학과 천체 역학을 하나로 통합했습니다.')}

<p>뉴턴은 케플러의 법칙들이 하나의 근본 원리 — <strong>만유인력(Universal Gravitation)</strong> — 에서 유도될 수 있음을 증명했습니다. 모든 질량을 가진 물체는 서로 끌어당기며, 그 힘의 크기는 두 질량의 곱에 비례하고 거리의 제곱에 반비례합니다.</p>

<div class="math-block">
<p><strong>뉴턴의 만유인력 법칙</strong></p>
<p>$$ F = G\\frac{{Mm}}{{r^2}} $$</p>
<p>여기서 \\(G = 6.674 \\times 10^{{-11}} \\, \\text{{N}} \\cdot \\text{{m}}^2 / \\text{{kg}}^2\\)는 만유인력 상수, \\(M\\)과 \\(m\\)은 두 물체의 질량, \\(r\\)은 두 물체 중심 사이의 거리입니다.</p>
</div>

<div class="arrow-summary"><span class="arrow-icon">&#9838;</span> 프톨레마이오스의 복잡한 주전원 체계 → 코페르니쿠스의 태양중심설 → 케플러의 타원 궤도 → 뉴턴의 만유인력: 2000년에 걸친 천문학의 발전은 점점 더 단순하고 통합적인 설명을 향해 나아갔습니다.</div>
</div>
</div>'''

ch2 = '''<div class="chapter" id="ch2">
<div class="chapter-banner" style="background:linear-gradient(135deg,#0a1628,#1a2a4e);height:180px;display:flex;align-items:center;justify-content:center;">
  <div style="text-align:center;color:#fff;font-size:24px;">
    $$ F = G\\frac{Mm}{r^2} $$
  </div>
</div>
<div class="chapter-body">
<h1 class="chapter-title" id="ch2-title">제2장. 만유인력의 법칙</h1>

<h2 class="section-title" id="sec2-1">2.1 뉴턴의 만유인력</h2>
<p>만유인력(Universal Gravitation)은 질량을 가진 모든 물체 사이에 작용하는 인력입니다. 뉴턴은 사과가 떨어지는 현상과 달이 지구 주위를 도는 현상이 <strong>같은 힘</strong>에 의한 것임을 깨달았습니다.</p>

<div class="math-block">
<p><strong>중력장(Gravitational Field)</strong></p>
<p>질량 \\(M\\)인 물체가 만드는 중력장의 세기:</p>
<p>$$ \\vec{g} = -\\frac{GM}{r^2} \\hat{r} $$</p>
<p><strong>중력 위치에너지(Gravitational Potential Energy)</strong></p>
<p>$$ U = -\\frac{GMm}{r} $$</p>
<p>이 에너지가 음수인 이유는 무한히 먼 곳을 기준(\\(U=0\\))으로 삼기 때문입니다. 두 물체가 가까울수록 에너지가 더 낮습니다.</p>
</div>

<div class="info-box">
<div class="box-title">[태양계 주요 천체의 만유인력]</div>
<p><strong>태양-지구</strong>: \\(F = 3.54 \\times 10^{22}\\) N (거리 1 AU = 1.496 × 10¹¹ m)</p>
<p><strong>지구-달</strong>: \\(F = 1.98 \\times 10^{20}\\) N (거리 3.844 × 10⁸ m)</p>
<p><strong>태양-달</strong>: \\(F = 4.34 \\times 10^{20}\\) N — 흥미롭게도 태양이 달을 당기는 힘이 지구가 달을 당기는 힘의 약 2.2배!</p>
</div>

<p>그렇다면 왜 달이 태양 쪽으로 날아가지 않을까요? 달은 태양 주위를 도는 지구와 함께 공전하고 있기 때문입니다. 달은 지구와 함께 태양 주위를 돌면서, <em>동시에</em> 지구 주위를 돕니다.</p>

<div class="math-block">
<p><strong>원 궤도의 조건 (뉴턴의 통찰)</strong></p>
<p>만유인력이 구심력 역할을 합니다:</p>
<p>$$ \\frac{GMm}{r^2} = \\frac{mv^2}{r} \\quad \\Rightarrow \\quad v = \\sqrt{\\frac{GM}{r}} $$</p>
<p>이를 \\(v = 2\\pi r / T\\)와 결합하면 케플러의 제3법칙이 유도됩니다:</p>
<p>$$ T^2 = \\frac{4\\pi^2}{GM} r^3 $$</p>
</div>

<h2 class="section-title" id="sec2-2">2.2 만유인력 시뮬레이션</h2>
<p>아래 시뮬레이션에서 태양-지구 간 거리와 지구의 질량을 조절하며 만유인력의 크기 변화를 관찰해보세요. 거리를 2배로 늘리면 힘은 1/4로 줄어들고(역제곱 법칙), 질량을 2배로 늘리면 힘도 2배가 됩니다.</p>

<div class="sim-container">
  <canvas id="gravityCanvas" width="700" height="280"></canvas>
  <div class="sim-controls">
    <label>태양-지구 거리: <input type="range" id="gravityDist" min="0.5" max="3" step="0.1" value="1"> <span id="gravDistVal">1.0</span> AU</label>
    <label>지구 질량 배수: <input type="range" id="gravityMass" min="0.5" max="3" step="0.1" value="1"> <span id="gravMassVal">1.0</span>x</label>
  </div>
  <div class="sim-info" id="gravityInfo"></div>
</div>

<h2 class="section-title" id="sec2-3">2.3 우주 탈출속도 시뮬레이션</h2>

<p>지표면에서 발사된 비행체의 궤도는 <strong>초기속도</strong>에 의해 결정됩니다. 뉴턴의 만유인력에 의한 역학적 에너지 보존으로부터 두 가지 임계 속도를 유도할 수 있습니다:</p>

<div class="math-block">
<p><strong>제1우주속도 (원궤도 속도)</strong></p>
<p>$$ v_1 = \\sqrt{\\frac{GM}{R}} \\approx 7.9 \\, \\text{km/s} $$</p>
<p>지표면 근처에서 원궤도를 유지하는 최소 속도입니다.</p>

<p><strong>제2우주속도 (탈출속도)</strong></p>
<p>$$ v_2 = \\sqrt{\\frac{2GM}{R}} = \\sqrt{2} \\, v_1 \\approx 11.2 \\, \\text{km/s} $$</p>
<p>지구 중력권을 벗어나기 위한 최소 속도입니다. 총 역학적 에너지 \\(E = \\frac{1}{2}mv^2 - \\frac{GMm}{r} = 0\\)인 경계 조건에서 유도됩니다.</p>

<p><strong>궤도 유형과 에너지</strong></p>
<p>$$ E < 0: \\text{타원(또는 원) 궤도} \\quad , \\quad E = 0: \\text{포물선 탈출} \\quad , \\quad E > 0: \\text{쌍곡선 탈출} $$</p>
</div>

<p>아래 시뮬레이션에서 초기속도를 조절하고 <strong>발사</strong> 버튼을 눌러 비행체의 궤적을 관찰하세요. 궤적 색상은 속도를 나타냅니다 — 근지점에서 빠르고(붉은색), 원지점에서 느립니다(푸른색).</p>

<div class="sim-container">
  <canvas id="escapeCanvas" width="700" height="360"></canvas>
  <div class="sim-controls">
    <label>초기속도: <input type="range" id="escapeVelSlider" min="0.3" max="2.0" step="0.01" value="1.0"> <span id="escapeVelVal">7.9 km/s</span></label>
    <button id="escapeLaunchBtn">발사</button>
    <button id="escapeResetBtn">초기화</button>
  </div>
  <div class="sim-info" id="escapeInfo"></div>
</div>

<div class="info-box">
<div class="box-title">[시뮬레이션 관찰 포인트]</div>
<p><strong>v ≈ 5.5 km/s (0.7 v₁)</strong>: 비행체가 지구에 다시 충돌합니다 (아궤도)</p>
<p><strong>v = 7.9 km/s (1.0 v₁)</strong>: 원형 궤도 — 인공위성 궤도의 기본형</p>
<p><strong>v ≈ 9.5 km/s (1.2 v₁)</strong>: 타원 궤도 — 원지점이 발사 반대편에 형성</p>
<p><strong>v = 11.2 km/s (√2 v₁)</strong>: 포물선 탈출 — 무한히 멀어지며 속도가 0에 수렴</p>
<p><strong>v > 11.2 km/s</strong>: 쌍곡선 탈출 — 잔여 속도를 유지하며 탈출</p>
</div>

</div>
</div>'''

ch3 = '''<div class="chapter" id="ch3">
<div class="chapter-banner" style="background:linear-gradient(135deg,#0a1628,#0a2040);height:180px;display:flex;align-items:center;justify-content:center;">
  <div style="text-align:center;color:#fff;">
    <div style="font-size:36px;">&#x1F30D; &#x2192; &#x2609;</div>
    <div style="font-size:14px;opacity:0.7;margin-top:8px;">Orbital Mechanics of the Sun-Earth-Moon System</div>
  </div>
</div>
<div class="chapter-body">
<h1 class="chapter-title" id="ch3-title">제3장. 공전과 자전</h1>

<h2 class="section-title" id="sec3-1">3.1 지구의 공전과 자전</h2>

<p><strong>지구의 공전(Revolution)</strong>은 태양 주위를 약 365.25일에 한 바퀴 도는 운동입니다. 궤도는 거의 원에 가까운 타원이며, 이심률은 \\(e = 0.0167\\)로 매우 작습니다.</p>

<div class="info-box">
<div class="box-title">[지구 공전 궤도 매개변수]</div>
<p>장반경: \\(a = 1.496 \\times 10^{11}\\) m (1 AU)</p>
<p>이심률: \\(e = 0.0167\\)</p>
<p>근일점(1월 초): \\(r_{min} = a(1-e) = 1.471 \\times 10^{11}\\) m</p>
<p>원일점(7월 초): \\(r_{max} = a(1+e) = 1.521 \\times 10^{11}\\) m</p>
<p>공전 속도: 평균 29.78 km/s (근일점에서 약간 빠르고, 원일점에서 약간 느림 — 케플러 제2법칙)</p>
</div>

<p><strong>지구의 자전(Rotation)</strong>은 지축을 중심으로 한 회전입니다. 항성일(sidereal day)은 23시간 56분 4초이며, 이는 먼 별을 기준으로 한 회전 주기입니다. 우리가 사용하는 태양일(solar day) 24시간은 태양을 기준으로 한 것으로, 공전 때문에 약 4분이 추가됩니다.</p>

<div class="math-block">
<p><strong>자전과 공전의 관계</strong></p>
<p>$$ \\frac{1}{T_{\\text{solar}}} = \\frac{1}{T_{\\text{sidereal}}} - \\frac{1}{T_{\\text{orbital}}} $$</p>
<p>$$ \\frac{1}{24\\text{h}} = \\frac{1}{23.934\\text{h}} - \\frac{1}{8766\\text{h}} $$</p>
</div>

<h2 class="section-title" id="sec3-2">3.2 달의 공전</h2>
<p>달은 지구 주위를 약 27.32일(항성월, sidereal month)에 한 바퀴 돕니다. 그러나 삭(신월)에서 다음 삭까지의 주기인 삭망월(synodic month)은 약 29.53일입니다. 이 차이는 지구가 태양 주위를 공전하기 때문에, 달이 한 바퀴를 돈 후 추가로 약 2.2일을 더 돌아야 태양-지구-달의 상대적 배치가 원래대로 돌아오기 때문입니다.</p>

<div class="math-block">
<p><strong>항성월과 삭망월의 관계</strong></p>
<p>$$ \\frac{1}{T_{\\text{syn}}} = \\frac{1}{T_{\\text{sid}}} - \\frac{1}{T_{\\text{Earth}}} $$</p>
<p>$$ \\frac{1}{29.53} = \\frac{1}{27.32} - \\frac{1}{365.25} $$</p>
</div>

<p>달은 <strong>동주기 자전(synchronous rotation)</strong>을 합니다. 즉, 자전 주기와 공전 주기가 같아서 항상 같은 면을 지구에 보여줍니다. 이는 조석력(tidal force)에 의한 조석 잠김(tidal locking) 현상의 결과입니다.</p>

<h2 class="section-title" id="sec3-3">3.3 태양-지구-달 궤도 시뮬레이션</h2>
<p>아래 시뮬레이션은 태양 주위를 도는 지구와, 지구 주위를 도는 달의 궤도 운동을 보여줍니다. 속도를 조절하며 공전 주기를 관찰해보세요. 지구의 궤도는 타원(\\(e=0.0167\\))이며, 근일점에서 약간 빨라지는 것을 확인할 수 있습니다.</p>

<div class="sim-container">
  <canvas id="orbitalCanvas" width="700" height="500"></canvas>
  <div class="sim-controls">
    <button id="orbitalPlayPause">⏸ 일시정지</button>
    <label>속도: <input type="range" id="orbitalSpeed" min="0.2" max="5" step="0.2" value="1"> <span id="orbSpeedVal">1.0</span>x</label>
  </div>
  <div class="sim-info" id="orbitalInfo"></div>
</div>
</div>
</div>'''

ch4 = '''<div class="chapter" id="ch4">
<div class="chapter-banner" style="background:linear-gradient(135deg,#1a2040,#2a3060);height:180px;display:flex;align-items:center;justify-content:center;">
  <div style="text-align:center;color:#fff;">
    <div style="font-size:32px;">&#x2600; &#x1F30F;</div>
    <div style="font-size:14px;opacity:0.7;margin-top:8px;">Solar Irradiance — Seoul · Jakarta · London</div>
  </div>
</div>
<div class="chapter-body">
<h1 class="chapter-title" id="ch4-title">제4장. 연중 일사량 변화</h1>

<h2 class="section-title" id="sec4-1">4.1 지축 경사와 계절</h2>
<p>계절이 바뀌는 근본 원인은 지구의 <strong>자전축 경사(obliquity)</strong>입니다. 지구의 자전축은 궤도면(황도면)에 대해 약 23.44° 기울어져 있으며, 이 기울기의 방향은 공전 도중 거의 변하지 않습니다(세차운동 제외).</p>

<p>이로 인해:</p>
<ul style="margin:10px 0 10px 20px;">
<li>여름: 태양이 높이 뜨고 낮이 길어짐 → 단위면적당 일사량 증가</li>
<li>겨울: 태양이 낮게 뜨고 낮이 짧아짐 → 단위면적당 일사량 감소</li>
</ul>

<div class="math-block">
<p><strong>태양 적위(Solar Declination)</strong></p>
<p>$$ \\delta = 23.44° \\times \\sin\\left(\\frac{360°}{365}(d - 81)\\right) $$</p>
<p>여기서 \\(d\\)는 1월 1일부터 센 날짜 수입니다. 춘분(\\(d \\approx 80\\))에서 \\(\\delta = 0\\), 하지(\\(d \\approx 172\\))에서 \\(\\delta = +23.44°\\)입니다.</p>

<p><strong>태양 최대 고도(Maximum Solar Elevation)</strong></p>
<p>$$ \\alpha_{max} = 90° - |\\phi - \\delta| $$</p>
<p>서울(\\(\\phi = 37.5°\\text{N}\\))에서의 하지 최대 고도: \\(90° - |37.5° - 23.44°| = 75.94°\\)</p>
<p>서울에서의 동지 최대 고도: \\(90° - |37.5° + 23.44°| = 29.06°\\)</p>

<p><strong>대기 질량(Air Mass)과 일사량</strong></p>
<p>$$ AM = \\frac{1}{\\sin(\\alpha)} \\quad , \\quad I = S_0 \\cdot \\sin(\\alpha) \\cdot 0.7^{AM^{0.678}} $$</p>
<p>여기서 \\(S_0 = 1361 \\, \\text{W/m}^2\\)은 태양 상수, \\(\\alpha\\)는 태양 고도각입니다.</p>
</div>

<div class="info-box">
<div class="box-title">[서울의 계절별 일조 특성]</div>
<p><strong>하지(6월 21일경)</strong>: 낮 길이 ~14.9시간, 최대 고도 ~76°, 최대 일사량 ~930 W/m²</p>
<p><strong>춘분/추분</strong>: 낮 길이 ~12시간, 최대 고도 ~52.5°, 최대 일사량 ~750 W/m²</p>
<p><strong>동지(12월 22일경)</strong>: 낮 길이 ~9.5시간, 최대 고도 ~29°, 최대 일사량 ~430 W/m²</p>
</div>

<p>아래 시뮬레이션에서 <strong>위도</strong>와 <strong>날짜</strong>를 조절하며 태양 고도각과 일사량의 변화를 직관적으로 확인할 수 있습니다. 왼쪽 패널은 지구 단면에서 지축 경사와 선택한 위도(붉은 점)를 보여주고, 오른쪽 패널은 해당 위도의 지표면에서 본 태양 남중 고도와 빔 조사 면적을 시각화합니다.</p>

<div class="sim-container">
  <canvas id="axialTiltCanvas" width="700" height="360"></canvas>
  <div class="sim-controls">
    <label>위도: <input type="range" id="tiltLatSlider" min="-66.5" max="89.5" step="0.5" value="37.5"> <span id="tiltLatVal">37.5°N</span></label>
    <label>날짜: <input type="range" id="tiltDaySlider" min="0" max="364" step="1" value="172"> <span id="tiltDayVal">6월 22일</span></label>
  </div>
  <div class="sim-info" id="tiltInfo"></div>
</div>

<div class="info-box">
<div class="box-title">[시뮬레이션 관찰 포인트]</div>
<p><strong>서울(37.5°N) 하지</strong>: α ≈ 76°, 빔 조사 면적 ≈ 1.03m → 높은 일사량</p>
<p><strong>서울(37.5°N) 동지</strong>: α ≈ 29°, 빔 조사 면적 ≈ 2.06m → 낮은 일사량</p>
<p><strong>적도(0°) 춘분</strong>: α = 90°, 빔 조사 면적 = 1.00m → 최대 일사량</p>
<p>위도를 -66.5°(남극권) 이하로 낮추면 동지 때 극야(24시간 밤)를, 66.5°N 이상에서는 하지 때 백야(24시간 낮)를 확인할 수 있습니다.</p>
</div>

<p>아래 <strong>천구 시뮬레이션</strong>에서는 같은 원리를 관측자의 시점으로 확인할 수 있습니다. 남쪽에서 비스듬히 바라본 천구 위에 태양의 일주 궤적(diurnal path)이 표시되며, 위도와 날짜에 따라 태양이 하늘을 가로지르는 호의 높이와 길이가 변합니다.</p>

<div class="sim-container">
  <canvas id="celestialSphereCanvas" width="700" height="400"></canvas>
  <div class="sim-controls">
    <label>위도: <input type="range" id="csLatSlider" min="-66.5" max="89.5" step="0.5" value="37.5"> <span id="csLatVal">37.5°N</span></label>
    <label>날짜: <input type="range" id="csDaySlider" min="0" max="364" step="1" value="172"> <span id="csDayVal">6월 22일</span></label>
    <button id="csPlayBtn">⏸ 일시정지</button>
  </div>
  <div class="sim-info" id="csInfo"></div>
</div>

<div class="info-box">
<div class="box-title">[천구 시뮬레이션 관찰 포인트]</div>
<p><strong>서울(37.5°N) 하지</strong>: 태양이 높은 호를 그리며 동북→남→서북으로 이동, 긴 낮</p>
<p><strong>서울(37.5°N) 동지</strong>: 태양이 낮은 호를 그리며 동남→남→서남으로 이동, 짧은 낮</p>
<p><strong>적도(0°) 춘분</strong>: 태양이 정동에서 떠서 천정을 지나 정서로 짐, 12시간 낮</p>
<p><strong>고위도(66.5°N) 하지</strong>: 태양이 지평선 아래로 내려가지 않는 백야</p>
</div>

<h2 class="section-title" id="sec4-2">4.2 연중 일사량 시뮬레이션: 서울 · 자카르타 · 런던</h2>

<p>천문학적 일사량(clear-sky irradiance)은 위도와 태양 고도각만으로 계산되지만, 실제로 지표면에 도달하는 일사량은 <strong>기상 조건</strong>에 크게 좌우됩니다. 구름, 장마, 태풍, 안개 등이 태양복사를 산란·흡수하여 실질 일사량을 감소시킵니다.</p>

<div class="info-box">
<div class="box-title">[세 도시의 기상 특성과 일사량]</div>
<p><strong>서울 (37.5°N)</strong>: 뚜렷한 사계절. 6~9월 장마·태풍 시즌에 일조율이 30% 이하로 급감. 맑은 날 기준 일사량은 높으나 실질 일사량은 장마철에 크게 감소합니다.</p>
<p><strong>자카르타 (6.2°S)</strong>: 적도 부근으로 연중 태양 고도가 높아 천문학적 일사량 변화가 작음. 그러나 11~3월 우기(monsoon)에 강수량이 집중되어 일조율이 떨어지고, 건기(6~9월)에 오히려 일사량이 최대가 됩니다.</p>
<p><strong>런던 (51.5°N)</strong>: 고위도에 해양성 기후로 연중 흐린 날이 많음. 겨울 일조율 ~20%, 여름에도 ~42%에 불과합니다. 천문학적 일사량 자체도 낮고, 기상으로 인한 감소까지 더해져 세 도시 중 실질 일사량이 가장 낮습니다.</p>
</div>

<div class="math-block">
<p><strong>실질 일사량 모델</strong></p>
<p>$$ I_{\\text{actual}}(d) = I_{\\text{clear}}(d) \\times f_{\\text{weather}}(d) $$</p>
<p>여기서 \\(f_{\\text{weather}}(d)\\)는 월별 일조율(sunshine fraction) 데이터를 일별로 보간한 기상 감쇠 계수입니다. 맑은 날의 이론적 일사량 \\(I_{\\text{clear}}\\)에 이 계수를 곱하면 장기 평균 실질 일사량을 추정할 수 있습니다.</p>
</div>

<p>아래 시뮬레이션에서 <span style="color:#ffaa44;font-weight:bold;">주황색은 서울</span>, <span style="color:#ff5566;font-weight:bold;">붉은색은 자카르타</span>, <span style="color:#44aaff;font-weight:bold;">파란색은 런던</span>입니다. <strong>점선</strong>은 맑은 날 이론 일사량(clear-sky), <strong>실선</strong>은 기상 보정 실질 일사량입니다. 파란 띠 영역은 서울의 장마·태풍 시즌(6~9월)을 표시합니다.</p>

<div class="sim-container">
  <canvas id="irradianceCanvas" width="700" height="380"></canvas>
  <div class="sim-controls">
    <label>날짜 (일차): <input type="range" id="daySlider" min="0" max="364" step="1" value="172"> <span id="dayVal">172</span></label>
  </div>
  <div class="sim-info" id="irradianceInfo"></div>
</div>

<div class="arrow-summary"><span class="arrow-icon">&#9838;</span> 위도가 낮을수록 연간 일사량 변동폭이 작고, 기상 조건(장마, 우기, 흐림)이 실질 일사량에 미치는 영향이 위도 효과 못지않게 크다는 것을 확인할 수 있습니다.</div>
</div>
</div>'''

ch5 = '''<div class="chapter" id="ch5">
<div class="chapter-banner" style="background:linear-gradient(135deg,#0a0e20,#1a1a3a);height:180px;display:flex;align-items:center;justify-content:center;">
  <div style="text-align:center;color:#fff;">
    <div style="font-size:36px;">&#x1F311; &#x1F312; &#x1F313; &#x1F314; &#x1F315; &#x1F316; &#x1F317; &#x1F318;</div>
    <div style="font-size:14px;opacity:0.7;margin-top:8px;">Lunar Phases and Diurnal Motion</div>
  </div>
</div>
<div class="chapter-body">
<h1 class="chapter-title" id="ch5-title">제5장. 달의 위상과 운동</h1>

<h2 class="section-title" id="sec5-1">5.1 삭망월과 항성월</h2>
<p><strong>달의 위상(lunar phase)</strong>은 태양-지구-달의 상대적 위치에 따라 결정됩니다. 태양 빛을 받는 달의 밝은 면 중 지구에서 보이는 부분의 비율이 달라지면서 위상이 변합니다.</p>

<div class="table-wrap"><table>
<thead><tr><th>위상</th><th>태양-달 이각</th><th>월출</th><th>남중</th><th>월몰</th><th>한밤 가시성</th></tr></thead>
<tbody>
<tr><td>삭 (신월, New Moon)</td><td>0°</td><td>~06시</td><td>~12시</td><td>~18시</td><td>불가</td></tr>
<tr class="alt-row"><td>초승달 (Waxing Crescent)</td><td>~45°</td><td>~09시</td><td>~15시</td><td>~21시</td><td>초저녁</td></tr>
<tr><td>상현달 (First Quarter)</td><td>90°</td><td>~12시</td><td>~18시</td><td>~00시</td><td>전반야</td></tr>
<tr class="alt-row"><td>상현망간 (Waxing Gibbous)</td><td>~135°</td><td>~15시</td><td>~21시</td><td>~03시</td><td>대부분</td></tr>
<tr><td>망 (보름달, Full Moon)</td><td>180°</td><td>~18시</td><td>~00시</td><td>~06시</td><td>밤새</td></tr>
<tr class="alt-row"><td>하현망간 (Waning Gibbous)</td><td>~225°</td><td>~21시</td><td>~03시</td><td>~09시</td><td>후반야</td></tr>
<tr><td>하현달 (Last Quarter)</td><td>270°</td><td>~00시</td><td>~06시</td><td>~12시</td><td>후반야</td></tr>
<tr class="alt-row"><td>그믐달 (Waning Crescent)</td><td>~315°</td><td>~03시</td><td>~09시</td><td>~15시</td><td>새벽</td></tr>
</tbody>
</table></div>

<p><strong>달의 일주운동</strong>: 달은 매일 약 50분씩 늦게 뜹니다. 이는 달이 하루에 약 12.2° 동쪽으로 이동하기 때문입니다(360° ÷ 29.53일 ≈ 12.2°/일). 지구 자전으로 인해 이 12.2°를 추가로 돌려면 약 50분이 필요합니다.</p>

<div class="math-block">
<p><strong>달의 일일 지연시간</strong></p>
<p>$$ \\Delta t = \\frac{360° / T_{syn}}{360° / 24\\text{h}} = \\frac{24\\text{h}}{T_{syn}} \\approx \\frac{24}{29.53} \\approx 0.81 \\text{h} \\approx 49 \\text{min} $$</p>
</div>

<h2 class="section-title" id="sec5-2">5.2 달의 위상 시뮬레이션</h2>
<p>왼쪽은 위에서 본 태양-지구-달 배치(태양빛은 왼쪽에서 옵니다), 오른쪽은 지구에서 바라본 달의 모습입니다. 날짜 슬라이더를 조절하여 29.53일의 삭망월 동안 달의 위상 변화를 관찰해보세요.</p>

<div class="sim-container">
  <canvas id="lunarCanvas" width="700" height="340"></canvas>
  <div class="sim-controls">
    <label>삭망월 일차: <input type="range" id="lunarDay" min="0" max="29.5" step="0.1" value="0"> <span id="lunarDayVal">0.0</span>일</label>
  </div>
  <div class="sim-info" id="lunarInfo"></div>
</div>
</div>
</div>'''

ch6 = '''<div class="chapter" id="ch6">
<div class="chapter-banner" style="background:linear-gradient(135deg,#0a0a1a,#1a0a1a);height:180px;display:flex;align-items:center;justify-content:center;">
  <div style="text-align:center;color:#fff;">
    <div style="font-size:40px;">&#x1F311;&#x2600;&#xFE0F;</div>
    <div style="font-size:14px;opacity:0.7;margin-top:8px;">Solar and Lunar Eclipses</div>
  </div>
</div>
<div class="chapter-body">
<h1 class="chapter-title" id="ch6-title">제6장. 일식과 월식</h1>

<h2 class="section-title" id="sec6-1">6.1 식 현상의 원리</h2>
<p><strong>일식(Solar Eclipse)</strong>은 달이 태양과 지구 사이에 와서 태양을 가리는 현상이며, <strong>월식(Lunar Eclipse)</strong>은 지구가 태양과 달 사이에 와서 달이 지구의 그림자에 들어가는 현상입니다.</p>

<p>그런데 왜 매달 일식과 월식이 일어나지 않을까요? 달의 궤도면이 황도면(지구 공전 궤도면)에 대해 약 <strong>5.145°</strong> 기울어져 있기 때문입니다. 달이 황도면을 지나는 점을 <strong>교점(node)</strong>이라 하며, 식은 삭이나 망이 교점 근처에서 일어날 때만 발생합니다.</p>

<div class="info-box">
<div class="box-title">[식의 종류]</div>
<p><strong>개기일식(Total Solar Eclipse)</strong>: 달의 본영(umbra)이 지구 표면에 도달. 관측 폭 ~100-200km. 평균 18개월마다 발생하나 같은 장소에서는 약 375년에 1회.</p>
<p><strong>부분일식(Partial Solar Eclipse)</strong>: 반영(penumbra)만 도달. 비교적 자주 관측 가능.</p>
<p><strong>금환일식(Annular Solar Eclipse)</strong>: 달이 원일점 근처에 있어 시직경이 태양보다 작을 때 발생. 태양의 가장자리가 고리처럼 보임.</p>
<p><strong>개기월식(Total Lunar Eclipse)</strong>: 달 전체가 지구의 본영에 진입. 달이 붉게 물드는 "블러드 문" 현상. 지구 대기를 통과한 붉은빛이 달에 도달하기 때문.</p>
</div>

<div class="math-block">
<p><strong>식 조건과 사로스 주기(Saros Cycle)</strong></p>
<p>식이 반복되는 주기(사로스 주기):</p>
<p>$$ T_{Saros} = 223 \\times T_{syn} = 242 \\times T_{dra} = 239 \\times T_{ano} \\approx 6585.3 \\text{일} \\approx 18\\text{년} 11\\text{일} $$</p>
<p>여기서 \\(T_{syn}\\)은 삭망월, \\(T_{dra}\\)은 교점월(27.21일), \\(T_{ano}\\)은 근점월(27.55일)입니다.</p>
<p>이 세 주기가 거의 정확히 정수배로 일치하기 때문에, 약 18년 11일마다 비슷한 조건의 식이 반복됩니다.</p>
</div>

<h2 class="section-title" id="sec6-2">6.2 일식/월식 시뮬레이션</h2>
<p>아래 시뮬레이션은 일식과 월식의 기하학적 조건을 보여줍니다. 달의 궤도 경사각 슬라이더를 조절하여, 경사각이 0°일 때(매달 식 발생)와 5°일 때(대부분 식 불발)의 차이를 비교해보세요.</p>

<div class="sim-container">
  <canvas id="eclipseCanvas" width="700" height="340"></canvas>
  <div class="sim-controls">
    <label>유형: <select id="eclipseType">
      <option value="solar">일식 (Solar Eclipse)</option>
      <option value="lunar">월식 (Lunar Eclipse)</option>
    </select></label>
    <label>달 궤도 경사: <input type="range" id="eclipseIncl" min="0" max="8" step="0.1" value="5.1"> <span id="eclInclVal">5.1</span>°</label>
  </div>
  <div class="sim-info" id="eclipseInfo"></div>
</div>
<h2 class="section-title" id="sec6-3">6.3 서울에서 본 일식/월식</h2>
<p>아래 시뮬레이션은 대한민국 서울(37.5°N, 127°E)에서 실제로 관측 가능한 일식/월식 이벤트를 보여줍니다. 이벤트를 선택하고 시간 슬라이더를 조절하여, 서울의 하늘에서 식이 어떻게 진행되는지 확인해보세요.</p>

<div class="info-box">
<div class="box-title">[서울에서 관측 가능한 주요 일식/월식]</div>
<p><strong>2028년 7월 22일</strong>: 부분일식 (식분 ~0.36) — 해질녘 관측</p>
<p><strong>2030년 6월 1일</strong>: 금환일식 (서울에서는 부분일식, 식분 ~0.89) — 오후 관측</p>
<p><strong>2035년 9월 2일</strong>: 개기일식 — 한반도에서 관측 가능! 개기식대가 북한을 지나며, 서울에서는 식분 ~0.98의 극적인 부분일식</p>
<p><strong>2025년 9월 7일</strong>: 개기월식 — 전 과정 관측 가능, "블러드 문"</p>
<p><strong>2028년 12월 31일</strong>: 개기월식 — 연말 밤하늘의 장관</p>
</div>

<div class="sim-container">
  <canvas id="eclipseObsCanvas" width="700" height="380"></canvas>
  <div class="sim-controls">
    <label>이벤트: <select id="eclipseEvent"></select></label>
    <label>진행: <input type="range" id="eclipseTime" min="0" max="1" step="0.005" value="0.5"> <span id="eclTimeVal">0.5</span></label>
  </div>
  <div class="sim-info" id="eclipseObsInfo"></div>
</div>

<div class="arrow-summary"><span class="arrow-icon">&#9838;</span> 2035년 9월 2일의 개기일식은 1887년 이후 약 148년 만에 한반도에서 관측되는 개기일식입니다. 서울에서는 거의 개기에 가까운 극적인 부분일식을 볼 수 있습니다.</div>
</div>
</div>'''

ch7 = '''<div class="chapter" id="ch7">
<div class="chapter-banner" style="background:linear-gradient(135deg,#1a1a3e,#2a2a4e);height:180px;display:flex;align-items:center;justify-content:center;">
  <div style="text-align:center;color:#fff;">
    <div style="font-size:16px;opacity:0.8;">&#x1F30D;☀️ vs ☀️&#x1F30D;</div>
    <div style="font-size:14px;opacity:0.7;margin-top:8px;">Geocentric vs Heliocentric: The Great Debate</div>
  </div>
</div>
<div class="chapter-body">
<h1 class="chapter-title" id="ch7-title">제7장. 천동설 vs 지동설</h1>

<h2 class="section-title" id="sec7-1">7.1 두 모델의 비교</h2>

<div class="table-wrap"><table>
<thead><tr><th>비교 항목</th><th>천동설 (Geocentric)</th><th>지동설 (Heliocentric)</th></tr></thead>
<tbody>
<tr><td>우주 중심</td><td>지구</td><td>태양</td></tr>
<tr class="alt-row"><td>역행운동 설명</td><td>주전원(epicycle)으로 설명</td><td>지구가 외행성을 추월할 때 자연 발생</td></tr>
<tr><td>금성의 위상</td><td>항상 초승달~반달 형태만 가능</td><td>모든 위상 가능 (갈릴레오가 확인)</td></tr>
<tr class="alt-row"><td>행성 밝기 변화</td><td>설명 어려움</td><td>거리 변화로 자연스럽게 설명</td></tr>
<tr><td>연주시차</td><td>없음 (관측되지 않음 → 당시에는 천동설 유리)</td><td>있어야 함 (1838년 베셀이 최초 관측)</td></tr>
<tr class="alt-row"><td>물리적 기반</td><td>아리스토텔레스의 자연철학</td><td>뉴턴의 만유인력 법칙</td></tr>
<tr><td>예측 정밀도</td><td>주전원 추가로 개선 가능</td><td>케플러 법칙으로 정밀하게 예측</td></tr>
<tr class="alt-row"><td>단순성</td><td>복잡 (80+ 개의 원 필요)</td><td>단순 (타원 궤도 6개로 충분)</td></tr>
</tbody>
</table></div>

<p><strong>역행운동(retrograde motion)</strong>은 천동설과 지동설을 구분하는 핵심 현상입니다. 화성과 같은 외행성은 주기적으로 하늘에서 서쪽으로 후퇴하는 것처럼 보입니다.</p>

<ul style="margin:10px 0 10px 20px;">
<li><strong>천동설 설명</strong>: 화성이 주전원(deferent) 위의 주전원(epicycle) 위에서 움직이면서 일시적으로 뒤로 가는 것처럼 보임. 주전원의 크기와 속도를 정밀하게 조정해야 관측과 일치.</li>
<li><strong>지동설 설명</strong>: 지구가 안쪽 궤도에서 더 빨리 공전하므로, 화성을 "추월"할 때 상대적으로 화성이 뒤로 가는 것처럼 보임. 추가 가정 없이 자연스럽게 설명됨.</li>
</ul>

<div class="arrow-summary"><span class="arrow-icon">&#9838;</span> 오컴의 면도날(Occam's Razor): 같은 현상을 설명할 때, 더 적은 가정을 사용하는 이론이 우선됩니다. 지동설은 주전원 없이 역행운동을 설명하며, 뉴턴의 만유인력이라는 물리적 기반을 갖추고 있습니다.</div>

<h2 class="section-title" id="sec7-2">7.2 역행운동 시뮬레이션: 천동설 vs 지동설</h2>
<p>왼쪽(천동설)에서는 지구를 중심에 놓고 화성의 궤적을 추적합니다. 화성의 궤적이 복잡한 고리(역행 루프)를 그리는 것을 볼 수 있습니다.<br>
오른쪽(지동설)에서는 태양을 중심에 놓고, 지구와 화성이 각자의 궤도를 단순하게 공전합니다. 노란 점선은 지구에서 화성을 바라보는 시선 방향입니다.</p>

<div class="sim-container">
  <div class="sim-dual">
    <div>
      <div class="sim-label">천동설 (Geocentric)</div>
      <canvas id="geoCanvas" width="320" height="320"></canvas>
    </div>
    <div>
      <div class="sim-label">지동설 (Heliocentric)</div>
      <canvas id="helioCanvas" width="320" height="320"></canvas>
    </div>
  </div>
  <div class="sim-controls">
    <button id="compPlayPause">⏸ 일시정지</button>
    <label>속도: <input type="range" id="compSpeed" min="0.5" max="5" step="0.5" value="1.5"> <span id="compSpeedVal">1.5</span>x</label>
  </div>
  <div class="sim-info" id="compInfo"></div>
</div>

<div class="challenge-box">
<div class="box-title">생각해봅시다</div>
<ul>
<li>왼쪽 그림에서 화성의 궤적이 고리 모양을 그리는 이유는 무엇일까요?</li>
<li>프톨레마이오스는 이 복잡한 궤적을 설명하기 위해 몇 개의 원이 필요했을까요?</li>
<li>오른쪽 그림에서 지구-화성 시선(노란 점선)이 방향을 바꾸는 순간이 역행의 시작과 끝입니다. 확인해보세요.</li>
<li>천동설이 "틀렸다"기보다는 "불필요하게 복잡했다"고 보는 것이 더 정확합니다. 왜 그럴까요?</li>
</ul>
</div>
</div>
</div>'''

refs = '''<div id="references">
<h2>참고문헌</h2>
<div class="ref-item">[1] Newton, I. (1687). <em>Philosophiæ Naturalis Principia Mathematica</em>.</div>
<div class="ref-item">[2] Copernicus, N. (1543). <em>De Revolutionibus Orbium Coelestium</em>.</div>
<div class="ref-item">[3] Kepler, J. (1609). <em>Astronomia Nova</em>.</div>
<div class="ref-item">[4] Kepler, J. (1619). <em>Harmonices Mundi</em>.</div>
<div class="ref-item">[5] Ptolemy, C. (c. 150 AD). <em>Almagest (Mathēmatikē Syntaxis)</em>.</div>
<div class="ref-item">[6] Galilei, G. (1632). <em>Dialogo sopra i due massimi sistemi del mondo</em>.</div>
<div class="ref-item">[7] Meeus, J. (1991). <em>Astronomical Algorithms</em>. Willmann-Bell.</div>
<div class="ref-item">[8] Carroll, B. W. & Ostlie, D. A. (2017). <em>An Introduction to Modern Astrophysics</em> (2nd ed.). Cambridge University Press.</div>
<div class="ref-item">[9] 한국천문연구원. <em>천문우주지식정보</em>. <a href="https://astro.kasi.re.kr" target="_blank">https://astro.kasi.re.kr</a></div>
<div class="ref-item">[10] 기상청 기상자료개방포털. <em>일사량 관측자료</em>. <a href="https://data.kma.go.kr" target="_blank">https://data.kma.go.kr</a></div>
</div>'''

# ═══════════════════════════════════════════════════════════
# Build HTML
# ═══════════════════════════════════════════════════════════
tmpl = ReportTemplate(
    fig_dir=os.path.join(OUTPUT_DIR, '..', 'fig_images'),
    title='천체역학 인터랙티브 교재',
    subtitle='태양, 지구, 달의 춤 — 만유인력에서 일식까지',
    sidebar_title='천체역학<br>인터랙티브 교재',
    sidebar_subtitle='대학 기초물리 · Interactive Textbook',
    og_url='https://jehyunlee.github.io/celestial-mechanics/',
    og_description='태양-지구-달 시스템의 천체역학을 인터랙티브 시뮬레이션으로 배우는 교재',
    og_image_url='https://jehyunlee.github.io/celestial-mechanics/og-image.jpg',
)

chapters = [ch1, ch2, ch3, ch4, ch5, ch6, ch7]

html = tmpl.build(
    toc_items=toc_items,
    chapters_html=chapters,
    refs_html=refs,
)

# ── Inject OG image dimensions (required for KakaoTalk) ──
html = html.replace(
    '<meta property="og:type" content="website" />',
    '<meta property="og:type" content="website" />\n'
    '  <meta property="og:image:width" content="1200" />\n'
    '  <meta property="og:image:height" content="630" />'
)

# ── Inject KaTeX CDN ──
html = html.replace('</head>', KATEX_HEAD + '</head>')

# ── Inject custom CSS ──
html = html.replace('</style>', '\n/* ── Custom Simulation Styles ── */\n' + CUSTOM_CSS + '\n</style>')

# ── Inject slider value display JS + simulation JS ──
SLIDER_JS = '''
// Slider value displays
document.querySelectorAll('input[type="range"]').forEach(slider => {
  const valSpan = document.getElementById(slider.id.replace(/([A-Z])/g, (m) => m).replace('gravity','grav').replace('orbital','orb').replace('eclipse','ecl').replace('lunar','lunar').replace('day','day').replace('comp','comp') + 'Val');
  slider.addEventListener('input', () => {
    // Find the nearest span after the slider
    const label = slider.closest('label');
    if (label) {
      const span = label.querySelector('span');
      if (span) span.textContent = slider.value;
    }
  });
  // Initial value
  const label = slider.closest('label');
  if (label) {
    const span = label.querySelector('span');
    if (span) span.textContent = slider.value;
  }
});
'''

html = html.replace('</body>',
    '<script>\n' + SLIDER_JS + '\n' + SIMULATION_JS + '\n</script>\n</body>')

# ── Write output ──
output_path = os.path.join(OUTPUT_DIR, 'index.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'✓ Generated: {output_path}')
print(f'  Size: {len(html):,} bytes ({len(html)//1024:,} KB)')
