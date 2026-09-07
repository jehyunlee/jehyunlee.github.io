import * as THREE from './vendor/three.module.min.js';
import {MazeSession,DIRECTIONS,relativeDirection,hasLineOfSight,LOGO_COLOR,UNVISITED_COLOR,logoPosition,overviewCameraPose,mobileCameraLayout,generateRandomStages} from './maze-core.js?v=20260907-1';

import {BALL_RADIUS,createBallTexture,drawAnniversaryLogo,resetBallOrientation,rollBall} from './ball-player.js?v=20260907-1';

const $=id=>document.getElementById(id);
const STAGE_COLORS=['#e2f58a','#93ddf5','#e3b1ff','#ffcd75'];
const COPY={
  en:{
    documentTitle:"KIST Maze · Journey into Research",meta:'Explore a new KIST-shaped 3D maze with the KIST 60 ball every time you play.',journeyCaption:'KIST · YOUR JOURNEY',
    brandSubtitle:"Journey into Research",stage0:'PATENT',stage1:'PAPER',stage2:'TRANSFER',stage3:'FIELD USE',helpEyebrow:'HOW TO PLAY',pauseEyebrow:'TAKE A BREATH',newAdventure:'NEW ADVENTURE',
    exploring:'EXPLORING',zoneTime:'ZONE TIME',totalTime:'TOTAL TIME',steps:'STEPS',visited:'Visited',unexplored:'Unexplored',unexploredBright:'Unexplored · 50% brighter',routeRule:'15 choices, one exit',
    fullMap:'FULL MAP',currentMarker:'White marker: current position',playerName:'KIST 60 Ball',move:'MOVE',takeBreak:'TAKE A BREAK',limitedView:'Only your surroundings are visible',
    introTitle:'At the end of the research tunnel,<br> <span>a hidden treasure.</span>',introCopy:'Explore between towering walls with the KIST 60 ball<br>and complete K, I, S, and T in order.',loadingMaze:'Building a new maze…',start:'START EXPLORING',introHint:'Use the arrow keys or the translucent controls around the KIST 60 ball',wanderOkay:'A little wandering is part of discovery.',
    helpTitle:'Find your own way.',helpIntro:'Complete the four logo-shaped mazes in K → I → S → T order. Every new game creates new walls, entrances, exits, and a unique route with 15 correct choices.',helpForward:'Move one tile forward',helpTurn:'Turn and move one tile',helpBack:'Turn around and move one tile',helpControls:'Hold an arrow key to keep rolling in the same direction. The camera follows behind you. On mobile, use the translucent arrows around the KIST 60 ball; zone information and time are shown at the bottom.',helpTiles:'Visited tiles use KIST red; unexplored tiles are 50% brighter. The white number shows how many times you have stepped on that tile. The entrance starts at 1.',helpOverview:'Select Overview to see the full map and your position for one second. Cyan marks entrances; each reward color marks an exit. Overview time is not counted.',helpFinish:'At an exit, the camera zooms out and fireworks fill the screen. After five seconds, it automatically zooms into the next entrance. Camera moves, celebrations, and pauses do not count toward your time.',gotIt:'Got it',
    pauseTitle:'Take a short break.',pauseCopy:'The maze will wait for you.',resume:'Keep exploring',restart:'Start a new maze',restartTitle:'Start with a new maze?',restartCopy:'Your records will be cleared and all four mazes will be regenerated.',confirmRestart:'Generate new maze',goBack:'Go back',finalTime:'FINAL TIME',loadErrorTitle:'The maze could not open.',reload:'Reload',overview:'Overview',
    stages:[
      {name:'Patent Maze',objective:'Find the exit and file your patent.',reward:'PATENT FILED!',line:'Your first discovery meets the world.'},
      {name:'Paper Maze',objective:'Complete your paper along a new path.',reward:'PAPER PUBLISHED!',line:'Your discovery becomes a published paper.'},
      {name:'Technology Maze',objective:'Find the path that takes technology into the world.',reward:'TECH TRANSFER!',line:'A laboratory idea reaches the world.'},
      {name:'Field Application Maze',objective:'Bring your research into the field at the final exit.',reward:'FIELD APPLICATION!',line:'K → I → S → T — the hidden treasure reaches the field!'}],
    directions:['north','east','south','west'],facing:value=>`Facing ${value}`,stageStatus:(letter,state)=>`${letter} ${state==='complete'?'complete':state==='current'?'in progress':'not reached'}`,zoneComplete:letter=>`${letter} ZONE COMPLETE`,allComplete:'KIST · ALL ZONES COMPLETE',
    entrance:letter=>`Entrance ${letter}`,exit:letter=>`Exit ${letter}`,nextMarker:letter=>`Next · ${letter}`,currentPosition:'Current position',
    entered:letter=>`You reached the ${letter} entrance. Find the next exit.`,peekDone:'Position checked. Keep exploring.',blocked:'That way is blocked. Try another direction.',newMaze:'A new KIST maze has been generated.',
    stageStats:(letter,time,steps)=>`${letter} clear · ${time} · ${steps.toLocaleString()} steps`,allStats:steps=>`${steps.toLocaleString()} steps · all 4 zones complete`,next:letter=>`Enter ${letter}`,nextCountdown:(letter,count)=>`Enter ${letter} · ${count}`,replay:'PLAY A NEW MAZE',
    errorFetch:'The maze data could not be loaded. Please reload.',errorGraphics:'The 3D scene could not start. Open this page in a modern browser with WebGL support.',contextLost:'The 3D connection was interrupted. Select Reload.'
  },
  ko:{
    documentTitle:'KIST 미로 · 연구를 향한 여정',meta:'KIST 60주년 공와 매번 새롭게 생성되는 KIST 모양 3D 미로를 탐험하세요.',journeyCaption:'KIST · 나의 탐험 기록',
    brandSubtitle:'연구를 향한 여정',stage0:'특허출원',stage1:'논문출판',stage2:'기술이전',stage3:'현장적용',helpEyebrow:'게임 방법',pauseEyebrow:'잠시 쉬기',newAdventure:'새로운 탐험',
    exploring:'탐험 중',zoneTime:'현재 구역',totalTime:'전체 시간',steps:'걸음 수',visited:'지나온 길',unexplored:'미탐험',unexploredBright:'미탐험 · 50% 밝게',routeRule:'15개의 갈림길, 하나의 출구',
    fullMap:'전체 지도',currentMarker:'흰색 표식이 현재 위치입니다',playerName:'KIST 60주년 공',move:'이동',takeBreak:'잠시 쉬기',limitedView:'주변만 보이는 미로',
    introTitle:'연구의 터널 끝,<br> <span>숨겨진 보물.</span>',introCopy:'KIST 60주년 공와 높은 벽 사이를 탐험하고<br>K, I, S, T를 차례대로 통과해 보세요.',loadingMaze:'새 미로를 만들고 있어요…',start:'탐험 시작하기',introHint:'방향키 또는 공 주변의 반투명 키로 이동',wanderOkay:'조금 헤매도 괜찮아요.',
    helpTitle:'길은, 직접 찾아야죠.',helpIntro:'로고 모양의 네 미로를 K → I → S → T 순서로 통과하세요. 새 게임마다 벽, 입구, 출구와 15개의 올바른 선택으로 이루어진 정답 경로가 새로 생성됩니다.',helpForward:'보고 있는 방향으로 한 칸 이동',helpTurn:'해당 방향으로 돌아 한 칸 이동',helpBack:'뒤돌아 한 칸 이동',helpControls:'방향키를 누르고 있으면 같은 방향으로 계속 구릅니다. 시점은 공 뒤를 따라갑니다. 모바일은 공 주변의 반투명 방향키를 사용하고, 구역 정보와 시간은 하단에서 확인하세요.',helpTiles:'지나온 길은 KIST 로고의 빨강, 미탐험 길은 50% 밝은 빨강입니다. 흰 숫자는 발판을 밟은 횟수이며 시작 발판은 1회로 계산합니다.',helpOverview:'전체보기를 누르면 전체 지도와 현재 위치를 1초 동안 확인합니다. 청록빛은 입구, 각 보상색은 출구입니다. 전체보기 시간은 기록에 포함되지 않습니다.',helpFinish:'출구에서는 전체 로고로 줌아웃하며 폭죽이 터집니다. 5초 뒤 다음 입구로 자동 줌인합니다. 카메라 이동, 축하 화면과 일시 정지는 기록에 포함되지 않습니다.',gotIt:'알겠어요',
    pauseTitle:'잠깐, 쉬어가기.',pauseCopy:'미로는 기다려 줄 거예요.',resume:'계속 탐험하기',restart:'새 미로로 다시 시작',restartTitle:'새 미로로 시작할까요?',restartCopy:'현재 기록을 지우고 네 글자의 미로를 모두 새로 생성합니다.',confirmRestart:'새 미로 만들기',goBack:'돌아가기',finalTime:'최종 통과 시간',loadErrorTitle:'미로를 열 수 없어요.',reload:'다시 열기',overview:'전체보기',
    stages:[
      {name:'특허의 미로',objective:'출구를 찾아 특허를 출원하세요.',reward:'특허출원!',line:'첫 번째 발견을 세상에.'},
      {name:'논문의 미로',objective:'새로운 길에서 논문을 완성하세요.',reward:'논문출판!',line:'당신의 발견이 한 편의 논문으로.'},
      {name:'기술의 미로',objective:'기술이 세상으로 나갈 길을 찾으세요.',reward:'기술이전!',line:'연구실의 아이디어가 세상으로.'},
      {name:'현장적용의 미로',objective:'마지막 출구에서 연구를 현장에 적용하세요.',reward:'현장적용!',line:'K → I → S → T, 연구의 보물을 현장에 연결했어요!'}],
    directions:['북쪽','동쪽','남쪽','서쪽'],facing:value=>`${value}을 보는 중`,stageStatus:(letter,state)=>`${letter} ${state==='complete'?'완료':state==='current'?'탐험 중':'아직 도착하지 않음'}`,zoneComplete:letter=>`${letter} 구역 통과`,allComplete:'KIST · 전 구역 통과',
    entrance:letter=>`입구 ${letter}`,exit:letter=>`출구 ${letter}`,nextMarker:letter=>`다음 · ${letter}`,currentPosition:'현재 위치',
    entered:letter=>`${letter} 입구에 도착했어요. 다음 출구를 찾아보세요.`,peekDone:'현재 위치를 확인했어요. 탐험을 계속하세요.',blocked:'막힌 길이에요. 다른 방향을 찾아보세요.',newMaze:'새로운 KIST 미로가 생성됐어요.',
    stageStats:(letter,time,steps)=>`${letter} 통과 ${time} · ${steps.toLocaleString()}걸음`,allStats:steps=>`${steps.toLocaleString()}걸음 · 4개 구역 완주`,next:letter=>`${letter} 입구로 들어가기`,nextCountdown:(letter,count)=>`${letter} 입구로 이동 · ${count}`,replay:'새 미로 탐험하기',
    errorFetch:'미로 데이터를 불러오지 못했어요. 다시 열어 주세요.',errorGraphics:'3D 화면을 시작하지 못했어요. WebGL을 지원하는 최신 브라우저에서 다시 열어 주세요.',contextLost:'3D 화면 연결이 잠시 끊겼어요. 다시 열기를 눌러 주세요.'
  }
};
let language='en';
const copy=()=>COPY[language];
const stageInfo=index=>({...copy().stages[index],color:STAGE_COLORS[index]});
const CELL=3.2,WALL_HEIGHT=3.3,VISION_RADIUS=5.3;
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)').matches;
const mobileUI=matchMedia('(max-width: 700px), (hover: none), (pointer: coarse)');
const touchControls=$('touch-controls'),projectedPlayer=new THREE.Vector3();
let mobileLayout=null;
let renderer,scene,camera,mazeGroup,avatar,ball,portal,portalRing,playerRing;
let floorMesh,wallMesh,capMesh,visitNumbers,wallData=[],visibleCells=new Set();
let overviewGroup,overviewMarker,overviewCurrentMarker,overviewGateways=[],cameraTween=null,playFog,fireworksStarted=false;
let celebrationHold=0,peekHold=0,shownCountdown=-1;
let logoBounds={minX:Infinity,maxX:-Infinity,minZ:Infinity,maxZ:-Infinity};
const cameraLook=new THREE.Vector3();
const visitedColor=new THREE.Color(LOGO_COLOR),unvisitedColor=new THREE.Color(UNVISITED_COLOR);
let session,mazeTemplates,mode='intro',modeBeforeDialog='intro',animation=null,held=null,queued=null;
let playerPosition=new THREE.Vector3(),followPosition=new THREE.Vector3(),yaw=0,targetYaw=0;
let frameTime=0,nextInputAt=0,lastCollisionAt=-5,toastTimeout,loaded=false;
let audioContext,soundEnabled=false,fireworkTime=0,fireworkNext=0,particles=[],rockets=[];
const fireworks=$('fireworks'),fx=fireworks.getContext('2d');
const object=new THREE.Object3D(),explorerUniform={value:new THREE.Vector3()};
const temporary=new THREE.Vector3(),lastBallPosition=new THREE.Vector3();
const shadeMaterials=[];

function randomSeed() {
  if(globalThis.crypto?.getRandomValues){const value=new Uint32Array(1);crypto.getRandomValues(value);return value[0];}
  return (Date.now()^Math.floor(performance.now()*1000))>>>0;
}
function createRandomSession() { return new MazeSession(generateRandomStages(mazeTemplates,randomSeed(),15)); }
function setLabel(element,label) { element.setAttribute('aria-label',label);element.title=label; }
function applyLanguage(nextLanguage) {
  language=nextLanguage==='ko'?'ko':'en';const c=copy();
  document.documentElement.lang=language;document.title=c.documentTitle;
  document.querySelector('meta[name="description"]').content=c.meta;
  document.querySelectorAll('[data-i18n]').forEach(element=>{const value=c[element.dataset.i18n];if(typeof value==='string')element.textContent=value;});
  $('intro-title').innerHTML=c.introTitle;$('intro-copy').innerHTML=c.introCopy;
  document.querySelectorAll('[data-lang]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.lang===language)));
  setLabel($('overview'),c.overview);setLabel($('help'),language==='en'?'How to play':'게임 방법');setLabel($('pause'),language==='en'?'Pause':'일시 정지');
  setLabel($('sound'),soundEnabled?(language==='en'?'Sound off':'소리 끄기'):(language==='en'?'Sound on':'소리 켜기'));
  document.querySelector('.brand img').alt=language==='en'?'KIST logo':'KIST 로고';
  document.querySelector('.player-label canvas').setAttribute('aria-label',c.playerName);
  document.querySelector('.journey').setAttribute('aria-label',language==='en'?'Zone progress':'구역 진행 상황');
  document.querySelector('.zone-card').setAttribute('aria-label',language==='en'?'Current zone and exploration record':'현재 구역과 탐험 기록');
  document.querySelector('.compass').setAttribute('aria-label',language==='en'?'View direction':'시선 방향');
  $('touch-controls').setAttribute('aria-label',language==='en'?'Virtual direction controls around the KIST 60 ball':'공 주변 가상 방향키');
  document.querySelectorAll('.dpad button').forEach((button,index)=>button.setAttribute('aria-label',language==='en'?['Move forward','Move left','Move right','Move backward'][index]:['앞으로 이동','왼쪽으로 이동','오른쪽으로 이동','뒤로 이동'][index]));
  document.querySelectorAll('.dialog-close').forEach(button=>button.setAttribute('aria-label',language==='en'?'Close':'닫기'));
  if(renderer)renderer.domElement.setAttribute('aria-label',language==='en'?'the KIST 60 ball exploring between the tall walls of a KIST-shaped maze.':'KIST 모양 미로의 높은 벽 사이를 탐험하는 KIST 60주년 공.');
  if(session){updateHUD();if(session.finished)updateCelebrationCopy();}
  if(loaded) {
    $('start').firstElementChild.textContent=c.start;
    if(overviewGroup) {
      const wasVisible=overviewGroup.visible,showCurrent=overviewCurrentMarker?.visible;
      const entrance=mode==='intro'?0:['zoomOut','celebration'].includes(mode)?session.stageIndex+1:null;
      buildOverview(entrance);overviewGroup.visible=wasVisible;overviewCurrentMarker.visible=showCurrent;
    }
  }
}

function fogMaterial(options={}) {
  const material=new THREE.MeshStandardMaterial(options);
  material.onBeforeCompile=shader=>{
    shader.uniforms.uExplorer=explorerUniform;
    shader.vertexShader='varying vec3 vMazeWorld;\n'+shader.vertexShader;
    shader.vertexShader=shader.vertexShader.replace('#include <project_vertex>',`#include <project_vertex>
      vec4 mazePosition=vec4(transformed,1.0);
      #ifdef USE_INSTANCING
        mazePosition=instanceMatrix*mazePosition;
      #endif
      vMazeWorld=(modelMatrix*mazePosition).xyz;
    `);
    shader.fragmentShader='uniform vec3 uExplorer; varying vec3 vMazeWorld;\n'+shader.fragmentShader;
    shader.fragmentShader=shader.fragmentShader.replace('#include <fog_fragment>',`#include <fog_fragment>
      float mazeDistance=length(vMazeWorld.xz-uExplorer.xz);
      float darkness=smoothstep(6.0,17.0,mazeDistance);
      gl_FragColor.rgb=mix(gl_FragColor.rgb,vec3(0.0627451,0.0980392,0.1176471),darkness);
    `);
  };
  shadeMaterials.push(material);return material;
}

function makeEnvironment() {
  const canvas=document.createElement('canvas');canvas.width=512;canvas.height=256;
  const context=canvas.getContext('2d');
  const gradient=context.createLinearGradient(0,0,0,256);
  gradient.addColorStop(0,'#d4e7ec');gradient.addColorStop(.47,'#45515a');gradient.addColorStop(1,'#171d23');
  context.fillStyle=gradient;context.fillRect(0,0,512,256);
  context.fillStyle='#ffffff';context.fillRect(25,30,70,65);context.fillRect(305,40,27,95);
  context.fillStyle='#bedde9';context.fillRect(160,26,95,17);context.fillRect(440,80,45,65);
  const texture=new THREE.CanvasTexture(canvas);texture.mapping=THREE.EquirectangularReflectionMapping;texture.colorSpace=THREE.SRGBColorSpace;
  const pmrem=new THREE.PMREMGenerator(renderer),environment=pmrem.fromEquirectangular(texture);
  scene.environment=environment.texture;texture.dispose();pmrem.dispose();
}

function createAvatar(logoImage) {
  const character=new THREE.Group();
  const material=new THREE.MeshPhysicalMaterial({color:0xffffff,map:createBallTexture(logoImage,Math.min(8,renderer.capabilities.getMaxAnisotropy())),roughness:.18,metalness:0,clearcoat:1,clearcoatRoughness:.09,envMapIntensity:.85});
  ball=new THREE.Mesh(new THREE.SphereGeometry(BALL_RADIUS,64,48),material);
  ball.position.y=BALL_RADIUS;ball.castShadow=true;ball.receiveShadow=true;character.add(ball);
  const badge=document.querySelector('.player-label canvas'),badgeContext=badge.getContext('2d');
  badgeContext.fillStyle='#ffffff';badgeContext.fillRect(0,0,badge.width,badge.height);
  drawAnniversaryLogo(badgeContext,logoImage,4,2,72,72*176/170);
  const shadowCanvas=document.createElement('canvas');shadowCanvas.width=128;shadowCanvas.height=128;
  const c=shadowCanvas.getContext('2d'),g=c.createRadialGradient(64,64,12,64,64,64);
  g.addColorStop(0,'rgba(0,0,0,.55)');g.addColorStop(1,'rgba(0,0,0,0)');c.fillStyle=g;c.fillRect(0,0,128,128);
  const shadow=new THREE.Mesh(new THREE.PlaneGeometry(2.4,2.4),new THREE.MeshBasicMaterial({map:new THREE.CanvasTexture(shadowCanvas),transparent:true,depthWrite:false}));shadow.rotation.x=-Math.PI/2;shadow.position.y=.013;character.add(shadow);
  return character;
}

function setMatrix(mesh,index,x,y,z,sx=1,sy=1,sz=1,angle=0) {
  object.position.set(x,y,z);object.rotation.set(0,angle,0);object.scale.set(sx,sy,sz);object.updateMatrix();mesh.setMatrixAt(index,object.matrix);
}
function cellPosition(id) {
  return new THREE.Vector3(...logoPosition(session.stage,id,CELL));
}
function makeTextSprite(text,size,color='#e2f58a') {
  const canvas=document.createElement('canvas');canvas.width=256;canvas.height=256;
  const c=canvas.getContext('2d');c.fillStyle=color;c.font=`${size}px Georgia, serif`;c.textAlign='center';c.textBaseline='middle';
  c.shadowColor=color;c.shadowBlur=9;c.fillText(text,128,128);
  const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
  const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,transparent:true,depthTest:true}));sprite.scale.set(1.8,1.8,1);return sprite;
}
function makeVisitNumbers(capacity) {
  // One texture and one draw call for all white floor digits, including revisits.
  const canvas=document.createElement('canvas');canvas.width=1280;canvas.height=128;
  const context=canvas.getContext('2d');
  context.font='bold 90px ui-monospace, monospace';context.textAlign='center';context.textBaseline='middle';
  context.lineJoin='round';context.lineWidth=7;context.strokeStyle='#620a08';context.fillStyle='#ffffff';
  for(let digit=0;digit<10;digit++) {
    context.strokeText(String(digit),digit*128+64,68);context.fillText(String(digit),digit*128+64,68);
  }
  const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
  const geometry=new THREE.PlaneGeometry(1,1);geometry.rotateX(-Math.PI/2);
  geometry.setAttribute('visitDigit',new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1,capacity)),1).setUsage(THREE.DynamicDrawUsage));
  const material=new THREE.MeshBasicMaterial({map:texture,transparent:true,alphaTest:.03,depthWrite:false,toneMapped:false,polygonOffset:true,polygonOffsetFactor:-1,polygonOffsetUnits:-1});
  material.onBeforeCompile=shader=>{
    shader.vertexShader='attribute float visitDigit;\n'+shader.vertexShader;
    shader.vertexShader=shader.vertexShader.replace('#include <uv_vertex>',`#include <uv_vertex>
      #ifdef USE_MAP
        vMapUv.x=(vMapUv.x+visitDigit)/10.0;
      #endif
    `);
  };
  const mesh=new THREE.InstancedMesh(geometry,material,Math.max(1,capacity));
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);mesh.frustumCulled=false;mesh.count=0;mesh.userData.glyphs=[];
  return mesh;
}
function setVisitNumbers(mesh,tiles,angle=0) {
  const glyphs=mesh.userData.glyphs=[];
  for(const {x,z,count} of tiles) {
    if(!count)continue;
    const digits=String(count),scale=Math.min(1,2/(digits.length*.56+.22));
    for(let i=0;i<digits.length;i++)glyphs.push({x,z,scale,offset:(i-(digits.length-1)/2)*.56*scale,digit:Number(digits[i])});
  }
  mesh.count=glyphs.length;
  const attribute=mesh.geometry.getAttribute('visitDigit');
  glyphs.forEach((glyph,index)=>attribute.setX(index,glyph.digit));attribute.needsUpdate=true;
  orientVisitNumbers(mesh,angle);
}
function orientVisitNumbers(mesh,angle) {
  const sin=Math.sin(angle),cos=Math.cos(angle);
  mesh.userData.glyphs.forEach(({x,z,scale,offset},index)=>{
    // The number sits near the edge facing the camera, clear of the player's feet.
    setMatrix(mesh,index,x+offset*cos+.75*sin,.035,z-offset*sin+.75*cos,.78*scale,1,1.04*scale,angle);
  });
  mesh.instanceMatrix.needsUpdate=true;
}
function disposeGroup(group) {
  if(!group)return;
  const geometries=new Set(),materials=new Set();
  group.traverse(item=>{
    if(item.geometry)geometries.add(item.geometry);
    if(item.material)(Array.isArray(item.material)?item.material:[item.material]).forEach(m=>materials.add(m));
  });
  for(const geometry of geometries)geometry.dispose();
  for(const material of materials){if(material.map)material.map.dispose();material.dispose();}
  scene.remove(group);
}

function buildMaze() {
  disposeGroup(mazeGroup);shadeMaterials.length=0;
  mazeGroup=new THREE.Group();scene.add(mazeGroup);
  const stage=session.stage,info=stageInfo(session.stageIndex);
  const floorMaterial=fogMaterial({color:0xffffff,roughness:.79,metalness:.02,emissive:0x230400,emissiveIntensity:.2});
  floorMesh=new THREE.InstancedMesh(new THREE.BoxGeometry(CELL-.035,.16,CELL-.035),floorMaterial,stage.cells.length);
  floorMesh.receiveShadow=true;floorMesh.frustumCulled=false;floorMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);mazeGroup.add(floorMesh);
  const sideMaterial=fogMaterial({color:0x405761,roughness:.8,metalness:.06});
  const topMaterial=fogMaterial({color:0x74868d,roughness:.9});
  const capMaterial=fogMaterial({color:0x91a4a8,roughness:.6,metalness:.1});
  wallData=[];
  const origin=logoPosition({...stage,cells:[[0,0]]},0,CELL);
  for(let i=0;i<stage.cells.length;i++) {
    const [x,z]=stage.cells[i];
    DIRECTIONS.forEach(([dx,dz],dir)=>{
      const neighbor=session.lookup.get(`${x+dx},${z+dz}`);
      if(stage.links[i].includes(neighbor))return;
      if(neighbor!==undefined && neighbor<i)return;
      wallData.push({x:origin[0]+(x+dx*.5)*CELL,z:origin[2]+(z+dz*.5)*CELL,angle:dir%2===0?0:Math.PI/2,cells:[i,neighbor],height:WALL_HEIGHT});
    });
  }
  wallMesh=new THREE.InstancedMesh(new THREE.BoxGeometry(CELL+.22,1,.26),[sideMaterial,sideMaterial,topMaterial,sideMaterial,sideMaterial,sideMaterial],wallData.length);
  wallMesh.castShadow=true;wallMesh.receiveShadow=true;wallMesh.frustumCulled=false;wallMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);mazeGroup.add(wallMesh);
  capMesh=new THREE.InstancedMesh(new THREE.BoxGeometry(CELL+.23,.055,.28),capMaterial,wallData.length);capMesh.frustumCulled=false;capMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);mazeGroup.add(capMesh);
  const visibleCapacity=Math.min(stage.cells.length,(2*Math.floor(VISION_RADIUS)+1)**2);
  visitNumbers=makeVisitNumbers(visibleCapacity*String(Number.MAX_SAFE_INTEGER).length);mazeGroup.add(visitNumbers);
  portal=new THREE.Group();portal.position.copy(cellPosition(stage.goal));
  const incoming=stage.links[stage.goal][0],[gx,gz]=stage.cells[stage.goal],[px,pz]=stage.cells[incoming];
  portal.rotation.y=gx!==px?Math.PI/2:0;
  const glow=new THREE.MeshStandardMaterial({color:info.color,emissive:info.color,emissiveIntensity:2.2,roughness:.28});
  for(const side of [-1,1]) {
    const pillar=new THREE.Mesh(new THREE.BoxGeometry(.14,2.6,.16),glow);pillar.position.set(side*1.15,1.3,0);portal.add(pillar);
  }
  const lintel=new THREE.Mesh(new THREE.BoxGeometry(2.43,.14,.16),glow);lintel.position.y=2.6;portal.add(lintel);
  portalRing=new THREE.Mesh(new THREE.TorusGeometry(.72,.026,8,64),glow);portalRing.position.y=1.38;portal.add(portalRing);
  const letter=makeTextSprite(stage.letter,162,info.color);letter.position.y=1.4;portal.add(letter);
  const beacon=new THREE.PointLight(info.color,16,8,2);beacon.position.set(0,1.6,0);portal.add(beacon);
  const landing=new THREE.Mesh(new THREE.RingGeometry(.82,.99,64),new THREE.MeshBasicMaterial({color:info.color,side:THREE.DoubleSide,transparent:true,opacity:.8}));landing.rotation.x=-Math.PI/2;landing.position.y=.02;portal.add(landing);mazeGroup.add(portal);
  playerPosition.copy(cellPosition(session.cell));followPosition.copy(playerPosition);
  targetYaw=-session.heading*Math.PI/2;yaw=targetYaw;avatar.position.copy(playerPosition);
  resetBallOrientation(ball,yaw);lastBallPosition.copy(playerPosition);
  explorerUniform.value.copy(playerPosition);updateVisibility();updateHUD();
}

function updateVisibility() {
  const [x,z]=session.stage.cells[session.cell];visibleCells=new Set();
  session.stage.cells.forEach(([cx,cz],i)=>{
    if(Math.hypot(cx-x,cz-z)<=VISION_RADIUS && hasLineOfSight(session.stage,session.lookup,session.cell,i))visibleCells.add(i);
  });
  // At the starting screen, show a few near walls to frame the explorer.
  session.stage.cells.forEach(([cx,cz],i)=>{
    const visible=visibleCells.has(i);
    const position=cellPosition(i);
    setMatrix(floorMesh,i,position.x,-.09,position.z,visible?1:0,visible?1:0,visible?1:0);
    floorMesh.setColorAt(i,session.visited.has(i)?visitedColor:unvisitedColor);
  });
  floorMesh.instanceMatrix.needsUpdate=true;floorMesh.instanceColor.needsUpdate=true;
  setVisitNumbers(visitNumbers,[...visibleCells].map(id=>{
    const {x,z}=cellPosition(id);return {x,z,count:session.visitCounts[id]};
  }),yaw);
  portal.visible=visibleCells.has(session.stage.goal);
  for(const wall of wallData)wall.visible=wall.cells.some(i=>visibleCells.has(i));
  updateWalls(1);
}

function updateWalls(dt) {
  const behindX=Math.sin(yaw),behindZ=Math.cos(yaw);
  for(let i=0;i<wallData.length;i++) {
    const wall=wallData[i];
    if(!wall.visible){setMatrix(wallMesh,i,0,-20,0,0,0,0);setMatrix(capMesh,i,0,-20,0,0,0,0);continue;}
    const dx=wall.x-playerPosition.x,dz=wall.z-playerPosition.z;
    const behind=dx*behindX+dz*behindZ,side=Math.abs(dx*behindZ-dz*behindX);
    // Only the short wall directly between the camera and explorer is cut away.
    // Forward and side walls stay tall and opaque, so unseen paths remain hidden.
    const cutaway=behind>.35&&behind<5.2&&side<2.0;
    const target=cutaway?.67:WALL_HEIGHT;
    wall.height=THREE.MathUtils.damp(wall.height,target,12,dt);
    setMatrix(wallMesh,i,wall.x,wall.height/2,wall.z,1,wall.height,1,wall.angle);
    setMatrix(capMesh,i,wall.x,wall.height+.012,wall.z,1,1,1,wall.angle);
  }
  wallMesh.instanceMatrix.needsUpdate=true;capMesh.instanceMatrix.needsUpdate=true;
}

function makeOverviewBeacon(stage,id,label,color,emphasis=false) {
  const marker=new THREE.Group();marker.position.fromArray(logoPosition(stage,id,CELL));
  const solid=new THREE.MeshBasicMaterial({color,fog:false,toneMapped:false,side:THREE.DoubleSide,transparent:true,opacity:1});
  const glow=new THREE.MeshBasicMaterial({color,fog:false,toneMapped:false,side:THREE.DoubleSide,transparent:true,opacity:.24,depthWrite:false,blending:THREE.AdditiveBlending});
  const ring=new THREE.Mesh(new THREE.RingGeometry(emphasis?2.2:1.3,emphasis?2.8:1.82,48),solid);
  ring.rotation.x=-Math.PI/2;ring.position.y=.26;marker.add(ring);
  const halo=new THREE.Mesh(new THREE.RingGeometry(emphasis?2.9:1.9,emphasis?4.15:2.85,48),glow);
  halo.rotation.x=-Math.PI/2;halo.position.y=.2;marker.add(halo);
  const height=emphasis?8:5.4;
  const stem=new THREE.Mesh(new THREE.CylinderGeometry(emphasis?.13:.09,emphasis?.13:.09,height,10),solid);
  stem.position.y=height/2;marker.add(stem);
  const text=makeTextSprite(label,emphasis?68:76,color);text.position.y=height+1.5;
  text.scale.set(emphasis?7.8:5.1,emphasis?7.8:5.1,1);text.material.fog=false;text.material.toneMapped=false;marker.add(text);
  return marker;
}

function buildOverview(entranceIndex=session.stageIndex+1) {
  disposeGroup(overviewGroup);overviewGroup=new THREE.Group();scene.add(overviewGroup);
  overviewGateways=[];overviewMarker=null;overviewCurrentMarker=null;
  logoBounds={minX:Infinity,maxX:-Infinity,minZ:Infinity,maxZ:-Infinity};
  const floorGeometry=new THREE.BoxGeometry(CELL-.035,.18,CELL-.035);
  const floorMaterial=new THREE.MeshBasicMaterial({color:0xffffff,fog:false,toneMapped:false});
  const wallGeometry=new THREE.BoxGeometry(CELL+.2,WALL_HEIGHT,.26);
  const wallMaterial=new THREE.MeshBasicMaterial({color:0x4b292a,fog:false,toneMapped:false});
  const visitedTiles=[];
  for(let index=0;index<session.stages.length;index++) {
    const stage=session.stages[index],seen=session.visitedByStage[index];
    const floors=new THREE.InstancedMesh(floorGeometry,floorMaterial,stage.cells.length);
    const lookup=new Map(stage.cells.map((c,i)=>[c.join(','),i]));
    const walls=[];
    stage.cells.forEach(([x,z],i)=>{
      const [wx,,wz]=logoPosition(stage,i,CELL);
      logoBounds.minX=Math.min(logoBounds.minX,wx-CELL/2);logoBounds.maxX=Math.max(logoBounds.maxX,wx+CELL/2);
      logoBounds.minZ=Math.min(logoBounds.minZ,wz-CELL/2);logoBounds.maxZ=Math.max(logoBounds.maxZ,wz+CELL/2);
      setMatrix(floors,i,wx,-.1,wz);floors.setColorAt(i,seen.has(i)?visitedColor:unvisitedColor);
      if(seen.has(i))visitedTiles.push({x:wx,z:wz,count:session.visitsByStage[index][i]});
      DIRECTIONS.forEach(([dx,dz],dir)=>{
        const neighbor=lookup.get(`${x+dx},${z+dz}`);
        if(stage.links[i].includes(neighbor)||(neighbor!==undefined&&neighbor<i))return;
        walls.push({x:wx+dx*CELL/2,z:wz+dz*CELL/2,angle:dir%2===0?0:Math.PI/2});
      });
    });
    floors.instanceMatrix.needsUpdate=true;floors.instanceColor.needsUpdate=true;overviewGroup.add(floors);
    const wallInstances=new THREE.InstancedMesh(wallGeometry,wallMaterial,walls.length);
    walls.forEach((wall,i)=>setMatrix(wallInstances,i,wall.x,WALL_HEIGHT/2,wall.z,1,1,1,wall.angle));
    wallInstances.instanceMatrix.needsUpdate=true;overviewGroup.add(wallInstances);
    const entrance=makeOverviewBeacon(stage,stage.start,copy().entrance(stage.letter),'#8deeff');
    const exit=makeOverviewBeacon(stage,stage.goal,copy().exit(stage.letter),STAGE_COLORS[index]);
    overviewGateways.push(entrance,exit);overviewGroup.add(entrance,exit);
  }
  const overviewNumbers=makeVisitNumbers(visitedTiles.reduce((total,tile)=>total+String(tile.count).length,0));
  overviewNumbers.material.fog=false;setVisitNumbers(overviewNumbers,visitedTiles);overviewGroup.add(overviewNumbers);
  // The opening and stage transitions emphasize the entrance to enter next.
  if(Number.isInteger(entranceIndex)&&entranceIndex>=0&&entranceIndex<session.stages.length) {
    const nextStage=session.stages[entranceIndex];
    overviewMarker=makeOverviewBeacon(nextStage,nextStage.start,copy().nextMarker(nextStage.letter),'#ffffff',true);
    overviewGroup.add(overviewMarker);
  }
  overviewCurrentMarker=makeOverviewBeacon(session.stage,session.cell,copy().currentPosition,'#ffffff',true);
  overviewCurrentMarker.visible=false;overviewGroup.add(overviewCurrentMarker);
  overviewGroup.visible=false;
}

function getOverviewPose() {
  const pose=overviewCameraPose(logoBounds,camera.aspect,camera.fov);
  return {position:new THREE.Vector3(...pose.position),look:new THREE.Vector3(...pose.look)};
}
function getPlayPose() {
  const distance=mobileLayout?.distance??6.7;
  const look=playerPosition.clone();look.y=BALL_RADIUS;
  return {look,position:new THREE.Vector3(look.x+Math.sin(targetYaw)*distance,look.y+distance,look.z+Math.cos(targetYaw)*distance)};
}
function setCameraFrame(offsetY) {
  if(!offsetY){if(camera.view?.enabled)camera.clearViewOffset();return;}
  const view=camera.view;
  if(view?.enabled&&view.fullWidth===innerWidth&&view.fullHeight===innerHeight&&Math.abs(view.offsetY-offsetY)<.01)return;
  camera.setViewOffset(innerWidth,innerHeight,0,offsetY,innerWidth,innerHeight);
}
function beginCameraMove(type,destination) {
  const from=camera.position.clone(),to=destination.position.clone();
  const movingOut=type==='out'||type==='peekOut',movingIn=type==='in'||type==='peekIn';
  const lift=Math.max(38,Math.abs(to.y-from.y)*.32);
  const control1=from.clone().add(new THREE.Vector3(0,movingOut?lift:25,0));
  const control2=to.clone().add(new THREE.Vector3(0,movingIn?lift:25,0));
  cameraTween={type,progress:0,duration:reducedMotion?.35:type.startsWith('peek')?1.65:type==='out'?2.8:2.6,
    fromFrame:camera.view?.enabled?camera.view.offsetY:0,toFrame:movingIn?(mobileLayout?.offsetY??0):0,
    curve:new THREE.CubicBezierCurve3(from,control1,control2,to),fromLook:cameraLook.clone(),toLook:destination.look.clone()};
}
function updateCameraMove(dt) {
  const move=cameraTween;move.progress=Math.min(1,move.progress+dt/move.duration);
  const t=move.progress,ease=t*t*t*(t*(t*6-15)+10);
  setCameraFrame(THREE.MathUtils.lerp(move.fromFrame,move.toFrame,ease));
  camera.position.copy(move.curve.getPoint(ease));cameraLook.lerpVectors(move.fromLook,move.toLook,ease);camera.lookAt(cameraLook);
  if(move.type==='out'&&t>=.5&&!fireworksStarted) {
    fireworksStarted=true;$('celebration').classList.remove('revealing');
    burst(innerWidth*.5,innerHeight*.31,STAGE_COLORS[session.stageIndex],110);
    [523.25,659.25,783.99,1046.5].forEach((frequency,i)=>playTone(frequency,.3,.1,'triangle',i*.12));
  }
  if(t<1)return;
  cameraTween=null;
  if(move.type==='out') {
    mode='celebration';celebrationHold=0;shownCountdown=-1;$('next').disabled=false;$('next').focus();
  } else if(move.type==='peekOut') {
    mode='peekHold';peekHold=0;
  } else {
    mode='play';mazeGroup.visible=true;overviewGroup.visible=false;scene.fog=playFog;
    document.body.classList.remove('overview-mode');
    $('overview').disabled=false;
    if(move.type==='peekIn') {
      $('overview-peek-label').hidden=true;
      toast(copy().peekDone);
    } else toast(copy().entered(session.stage.letter));
  }
}

function updateHUD() {
  const info=stageInfo(session.stageIndex);
  $('zone-index').textContent=`ZONE 0${session.stageIndex+1} / 04`;
  $('zone-letter').textContent=session.stage.letter;$('zone-name').textContent=info.name;$('zone-objective').textContent=info.objective;
  $('steps').textContent=session.steps.toLocaleString();
  $('stage-timer').textContent=formatPreciseTime(session.stageElapsed);
  $('compass-needle').style.transform=`rotate(${session.heading*-90}deg)`;
  $('heading-label').textContent=copy().facing(copy().directions[session.heading]);
  document.querySelectorAll('.stage').forEach((el,i)=>{
    el.classList.toggle('current',i===session.stageIndex&&!session.finished);el.classList.toggle('complete',session.cleared.includes(i));
    el.setAttribute('aria-label',copy().stageStatus('KIST'[i],session.cleared.includes(i)?'complete':i===session.stageIndex?'current':'future'));
    if(i===session.stageIndex)el.setAttribute('aria-current','step');else el.removeAttribute('aria-current');
    el.querySelector('time').textContent=session.stageTimes[i]===null?'--:--':formatPreciseTime(session.stageTimes[i]);
  });
  document.querySelectorAll('.journey i').forEach((el,i)=>el.classList.toggle('done',session.cleared.includes(i)));
}
function formatTime(value) {
  const seconds=Math.floor(value);return `${Math.floor(seconds/60).toString().padStart(2,'0')}:${(seconds%60).toString().padStart(2,'0')}`;
}
function formatPreciseTime(value) {
  const centiseconds=Math.round(value*100),seconds=Math.floor(centiseconds/100);
  return `${Math.floor(seconds/60).toString().padStart(2,'0')}:${(seconds%60).toString().padStart(2,'0')}.${(centiseconds%100).toString().padStart(2,'0')}`;
}
function toast(message) {
  clearTimeout(toastTimeout);$('toast').textContent=message;$('toast').classList.add('show');
  toastTimeout=setTimeout(()=>$('toast').classList.remove('show'),2800);
}
function directionYaw(direction) {
  let angle=-direction*Math.PI/2;
  while(angle-targetYaw>Math.PI)angle-=Math.PI*2;
  while(angle-targetYaw<-Math.PI)angle+=Math.PI*2;
  return angle;
}
function step(direction,now) {
  if(mode!=='play'||animation)return;
  const result=session.move(direction);targetYaw=directionYaw(direction);
  $('compass-needle').style.transform=`rotate(${session.heading*-90}deg)`;
  $('heading-label').textContent=copy().facing(copy().directions[session.heading]);
  if(!result.allowed) {
    if(now-lastCollisionAt>.95){toast(copy().blocked);playTone(95,.08,.05,'sine');lastCollisionAt=now;}
    nextInputAt=now+.27;return;
  }
  animation={start:cellPosition(result.from),end:cellPosition(result.to),progress:0,duration:.34,complete:result.complete};
  playTone(140+Math.random()*25,.045,.025,'sine');updateHUD();
}
function press(relative,source) {
  if(mode!=='play')return;
  if(held?.source===source)return;
  const direction=relativeDirection(relative,session.heading);
  held={direction,source};queued=direction;nextInputAt=0;
}
function release(source) {if(held?.source===source)held=null;}
function clearInputs() {held=null;queued=null;document.querySelectorAll('.dpad button').forEach(b=>b.classList.remove('pressed'));}

function startGame() {
  if(!loaded||mode!=='intro')return;
  mode='zoomIn';document.body.classList.add('playing','overview-mode');$('start').blur();
  if(overviewMarker)overviewMarker.visible=false;
  beginCameraMove('in',getPlayPose());frameTime=performance.now()/1000;
}
function showOverview() {
  if(mode!=='play'||animation||cameraTween)return;
  clearInputs();mode='peekOut';$('overview').disabled=true;
  buildOverview(null);overviewCurrentMarker.visible=true;overviewGroup.visible=true;mazeGroup.visible=false;scene.fog=null;
  document.body.classList.add('overview-mode');$('overview-peek-label').hidden=false;$('toast').classList.remove('show');
  beginCameraMove('peekOut',getOverviewPose());
}
function openDialog(id) {
  if(!loaded||['celebration','zoomOut','zoomIn','peekOut','peekHold','peekIn'].includes(mode))return;
  if(!document.querySelector('dialog[open]'))modeBeforeDialog=mode;
  clearInputs();mode='pause';$(id).showModal();
}
function closeDialog(id) {$(id).close();if(!document.querySelector('dialog[open]'))mode=modeBeforeDialog;clearInputs();}
function restartGame() {
  document.querySelectorAll('dialog[open]').forEach(d=>d.close());
  $('celebration').hidden=true;particles=[];rockets=[];clearInputs();animation=null;cameraTween=null;
  if(overviewGroup)overviewGroup.visible=false;scene.fog=playFog;document.body.classList.remove('overview-mode');
  session=createRandomSession();buildMaze();mode='play';modeBeforeDialog='play';document.body.classList.add('playing');
  $('overview').disabled=false;$('overview-peek-label').hidden=true;
  $('timer').textContent='00:00';toast(copy().newMaze);
}
function updateCelebrationCopy() {
  if(!session?.finished)return;
  const index=session.stageIndex,info=stageInfo(index);
  $('celebration-eyebrow').textContent=index===3?copy().allComplete:copy().zoneComplete(session.stage.letter);
  $('celebration-letter').textContent=session.stage.letter;
  $('celebration-title').textContent=info.reward;$('celebration-description').textContent=info.line;
  $('celebration-stats').textContent=index===3?copy().allStats(session.steps):copy().stageStats(session.stage.letter,formatPreciseTime(session.stageTimes[index]),session.stageSteps);
  $('final-times').hidden=index!==3;
  if(index===3) {
    $('final-total-time').textContent=formatPreciseTime(session.totalClearTime);
    $('final-stage-times').innerHTML=session.stageTimes.map((time,i)=>`<div><b>${'KIST'[i]}</b><time>${formatPreciseTime(time)}</time></div>`).join('');
  }
  const nextLetter='KIST'[index+1];
  const label=index===3?copy().replay:mode==='celebration'&&shownCountdown>=0?copy().nextCountdown(nextLetter,shownCountdown):copy().next(nextLetter);
  $('next').innerHTML=`${label} <span>${index===3?'↻':'→'}</span>`;
}
function completeStage() {
  mode='zoomOut';session.recordStage();clearInputs();updateHUD();
  $('overview').disabled=true;$('overview-peek-label').hidden=true;
  const index=session.stageIndex,info=stageInfo(index);
  $('celebration').style.setProperty('--lime',info.color);
  $('celebration').classList.toggle('all-complete',index===3);
  updateCelebrationCopy();
  $('next').disabled=true;$('celebration').classList.add('revealing');$('celebration').hidden=false;
  particles=[];rockets=[];fireworkTime=0;fireworkNext=0;
  fireworksStarted=false;buildOverview();overviewGroup.visible=true;mazeGroup.visible=false;scene.fog=null;
  document.body.classList.add('overview-mode');$('toast').classList.remove('show');
  beginCameraMove('out',getOverviewPose());
}
function continueJourney() {
  if(mode!=='celebration')return;
  if(session.stageIndex===3){session=createRandomSession();buildOverview(0);overviewGroup.visible=true;}
  else if(!session.nextStage())return;
  $('celebration').hidden=true;particles=[];rockets=[];animation=null;clearInputs();buildMaze();
  mazeGroup.visible=false;mode='zoomIn';$('next').blur();
  if(overviewMarker)overviewMarker.visible=false;
  beginCameraMove('in',getPlayPose());
}
function playTone(frequency,duration,volume=.05,type='sine',delay=0) {
  if(!soundEnabled)return;
  try {
    if(!audioContext)audioContext=new (window.AudioContext||window.webkitAudioContext)();
    if(audioContext.state==='suspended')audioContext.resume().catch(()=>{});
    const at=audioContext.currentTime+delay,osc=audioContext.createOscillator(),gain=audioContext.createGain();
    osc.type=type;osc.frequency.setValueAtTime(frequency,at);gain.gain.setValueAtTime(.0001,at);
    gain.gain.exponentialRampToValueAtTime(volume,at+.012);gain.gain.exponentialRampToValueAtTime(.0001,at+duration);
    osc.connect(gain);gain.connect(audioContext.destination);osc.start(at);osc.stop(at+duration+.02);
  }catch{}
}
function burst(x,y,color,count=70) {
  if(reducedMotion)count=24;
  for(let i=0;i<count;i++) {
    const angle=Math.PI*2*i/count+(Math.random()-.5)*.15,speed=60+Math.random()*240;
    particles.push({x,y,vx:Math.cos(angle)*speed,vy:Math.sin(angle)*speed,life:1.2+Math.random()*1.4,max:2.6,color,size:1.4+Math.random()*2});
  }
  playTone(60,.21,.07,'triangle');
}
function animateFireworks(dt) {
  const width=innerWidth,height=innerHeight;
  fx.clearRect(0,0,width,height);fireworkTime+=dt;
  if(fireworkTime>=fireworkNext&&(!reducedMotion||fireworkTime<.1)) {
    fireworkNext=fireworkTime+.45+Math.random()*.45;
    const palette=[STAGE_COLORS[session.stageIndex],'#ff7675','#9ce9ff','#ffffff','#dfafff'];
    rockets.push({x:width*(.12+Math.random()*.76),y:height+10,target:height*(.13+Math.random()*.36),speed:340+Math.random()*170,color:palette[Math.floor(Math.random()*palette.length)]});
  }
  for(let i=rockets.length-1;i>=0;i--) {
    const r=rockets[i];r.y-=r.speed*dt;
    fx.strokeStyle=r.color;fx.globalAlpha=.7;fx.lineWidth=2;fx.beginPath();fx.moveTo(r.x,r.y+20);fx.lineTo(r.x,r.y);fx.stroke();
    if(r.y<=r.target){burst(r.x,r.y,r.color);rockets.splice(i,1);}
  }
  fx.globalCompositeOperation='lighter';
  for(let i=particles.length-1;i>=0;i--) {
    const p=particles[i];const oldX=p.x,oldY=p.y;p.life-=dt;
    p.x+=p.vx*dt;p.y+=p.vy*dt;p.vx*=Math.exp(-.55*dt);p.vy=p.vy*Math.exp(-.55*dt)+48*dt;
    if(p.life<=0){particles.splice(i,1);continue;}
    fx.globalAlpha=Math.min(1,p.life*.9);fx.strokeStyle=p.color;fx.lineWidth=p.size;
    fx.beginPath();fx.moveTo(oldX,oldY);fx.lineTo(p.x,p.y);fx.stroke();
    fx.fillStyle=p.color;fx.beginPath();fx.arc(p.x,p.y,p.size*.6,0,Math.PI*2);fx.fill();
  }
  fx.globalCompositeOperation='source-over';fx.globalAlpha=1;
}

function resize() {
  if(!renderer)return;
  const width=innerWidth,height=innerHeight;renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix();
  const ratio=Math.min(devicePixelRatio,2);fireworks.width=width*ratio;fireworks.height=height*ratio;fx.setTransform(ratio,0,0,ratio,0,0);
  if(mobileUI.matches) {
    const headerBottom=document.querySelector('.topbar').getBoundingClientRect().bottom;
    const journeyBottom=document.querySelector('.journey').getBoundingClientRect().bottom;
    const top=Math.max(headerBottom,journeyBottom+(width<=700&&height>530?22:0))+12;
    const bottom=document.querySelector('.zone-card').getBoundingClientRect().top-12;
    mobileLayout=mobileCameraLayout(width,height,top,bottom);
    touchControls.style.width=`${mobileLayout.padWidth}px`;touchControls.style.height=`${mobileLayout.padHeight}px`;
  } else {mobileLayout=null;touchControls.style.removeProperty('width');touchControls.style.removeProperty('height');}
  if(cameraTween) {
    const destination=cameraTween.type==='out'||cameraTween.type==='peekOut'?getOverviewPose():getPlayPose();
    cameraTween.curve.v3.copy(destination.position);cameraTween.toLook.copy(destination.look);
    cameraTween.toFrame=cameraTween.type==='in'||cameraTween.type==='peekIn'?(mobileLayout?.offsetY??0):0;
  } else {
    setCameraFrame(mode==='play'||(mode==='pause'&&modeBeforeDialog==='play')?(mobileLayout?.offsetY??0):0);
  }
}
function animate(timestamp) {
  if(!loaded)return;
  requestAnimationFrame(animate);
  const now=timestamp/1000,elapsedSeconds=Math.max(now-frameTime,0),dt=Math.min(elapsedSeconds,.05);frameTime=now;
  if(document.hidden)return;
  if(mode==='celebration'&&session.stageIndex<3) {
    celebrationHold+=dt;
    const remaining=Math.max(0,Math.ceil(5-celebrationHold));
    if(remaining!==shownCountdown) {
      shownCountdown=remaining;
      $('next').innerHTML=`${copy().nextCountdown('KIST'[session.stageIndex+1],remaining)} <span>→</span>`;
    }
    if(celebrationHold>=5)continueJourney();
  }
  if(mode==='peekHold') {
    peekHold+=elapsedSeconds;
    if(peekHold>=1){mode='peekIn';beginCameraMove('peekIn',getPlayPose());}
  }
  if(mode==='play') {
    session.tick(elapsedSeconds);$('timer').textContent=formatTime(session.elapsed);$('stage-timer').textContent=formatPreciseTime(session.stageElapsed);
    if(animation) {
      animation.progress=Math.min(1,animation.progress+dt/animation.duration);
      const t=animation.progress,curve=t*t*(3-2*t);playerPosition.lerpVectors(animation.start,animation.end,curve);
      if(t>=1) {
        const done=animation.complete;animation=null;updateVisibility();nextInputAt=now+.045;
        if(done)completeStage();
      }
    }
    if(!animation&&mode==='play'&&now>=nextInputAt) {
      const direction=queued??held?.direction;
      if(direction!==undefined&&direction!==null){queued=null;step(direction,now);}
    }
  }
  rollBall(ball,lastBallPosition,playerPosition);lastBallPosition.copy(playerPosition);
  yaw=THREE.MathUtils.damp(yaw,targetYaw,8,dt);avatar.position.copy(playerPosition);
  followPosition.lerp(playerPosition,1-Math.exp(-12*dt));
  const introMode=mode==='intro'||(mode==='pause'&&modeBeforeDialog==='intro');
  if(cameraTween) updateCameraMove(dt);
  else if(mode==='celebration'||mode==='peekHold'||introMode) {
    setCameraFrame(0);
    const pose=getOverviewPose();camera.position.copy(pose.position);cameraLook.copy(pose.look);camera.lookAt(cameraLook);
  } else {
    const distance=mobileLayout?.distance??6.7;
    setCameraFrame(mobileLayout?.offsetY??0);cameraLook.copy(followPosition);cameraLook.y=BALL_RADIUS;
    camera.position.set(cameraLook.x+Math.sin(yaw)*distance,cameraLook.y+distance,cameraLook.z+Math.cos(yaw)*distance);camera.lookAt(cameraLook);
  }
  explorerUniform.value.copy(playerPosition);if(mazeGroup.visible)updateWalls(dt);
  playerRing.position.set(playerPosition.x,.025,playerPosition.z);
  playerRing.material.opacity=.27+Math.sin(now*1.7)*.055;
  if(portalRing){portalRing.rotation.z=now*.16;portalRing.scale.setScalar(1+Math.sin(now*2)*.045);}
  if(overviewMarker)overviewMarker.scale.setScalar(1+Math.sin(now*2)*.045);
  if(overviewCurrentMarker?.visible)overviewCurrentMarker.scale.setScalar(1+Math.sin(now*4)*.08);
  overviewGateways.forEach((marker,index)=>marker.scale.setScalar(1+Math.sin(now*2+index*.8)*.025));
  if(mazeGroup.visible)orientVisitNumbers(visitNumbers,yaw);
  renderer.render(scene,camera);
  updateTouchControls();
  if((mode==='celebration'||mode==='zoomOut')&&fireworksStarted)animateFireworks(dt);
}

function updateTouchControls() {
  if(!mobileLayout||mode!=='play')return;
  // Track the actual projected ball while the chase camera moves and turns.
  camera.updateMatrixWorld();projectedPlayer.copy(playerPosition);projectedPlayer.y+=BALL_RADIUS;projectedPlayer.project(camera);
  const {padWidth,padHeight,top,bottom}=mobileLayout;
  const x=THREE.MathUtils.clamp((projectedPlayer.x*.5+.5)*innerWidth,padWidth/2+12,innerWidth-padWidth/2-12);
  const y=THREE.MathUtils.clamp((-projectedPlayer.y*.5+.5)*innerHeight,top+padHeight/2,bottom-padHeight/2);
  touchControls.style.left=`${x.toFixed(1)}px`;touchControls.style.top=`${y.toFixed(1)}px`;
}

function setupInputs() {
  const keys={ArrowUp:'up',ArrowRight:'right',ArrowDown:'down',ArrowLeft:'left',w:'up',d:'right',s:'down',a:'left',W:'up',D:'right',S:'down',A:'left'};
  window.addEventListener('keydown',event=>{
    if(event.key==='Escape') {
      if(document.querySelector('dialog[open]'))return;
      if(mode==='play'){event.preventDefault();openDialog('pause-dialog');}return;
    }
    if(keys[event.key]&&mode==='play') {event.preventDefault();if(!event.repeat)press(keys[event.key],event.code);}
  });
  window.addEventListener('keyup',event=>{if(keys[event.key]){event.preventDefault();release(event.code);}});
  document.querySelectorAll('.dpad button').forEach(button=>{
    button.addEventListener('pointerdown',event=>{event.preventDefault();if(mode!=='play')return;button.setPointerCapture(event.pointerId);button.classList.add('pressed');press(button.dataset.dir,`pointer-${event.pointerId}`);});
    const up=event=>{button.classList.remove('pressed');release(`pointer-${event.pointerId}`);};
    button.addEventListener('pointerup',up);button.addEventListener('pointercancel',up);button.addEventListener('lostpointercapture',up);
    button.addEventListener('contextmenu',event=>event.preventDefault());
  });
  window.addEventListener('blur',()=>{clearInputs();if(mode==='play')openDialog('pause-dialog');});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){clearInputs();if(mode==='play')openDialog('pause-dialog');}});
  $('start').addEventListener('click',startGame);$('help').addEventListener('click',()=>openDialog('help-dialog'));
  $('overview').addEventListener('click',showOverview);
  $('pause').addEventListener('click',()=>openDialog('pause-dialog'));
  $('resume').addEventListener('click',()=>closeDialog('pause-dialog'));
  $('restart').addEventListener('click',()=>{$('pause-dialog').close();$('restart-dialog').showModal();});
  $('cancel-restart').addEventListener('click',()=>{$('restart-dialog').close();$('pause-dialog').showModal();});
  $('confirm-restart').addEventListener('click',restartGame);
  document.querySelectorAll('[data-close]').forEach(button=>button.addEventListener('click',()=>closeDialog(button.dataset.close)));
  document.querySelectorAll('dialog').forEach(dialog=>dialog.addEventListener('cancel',event=>{event.preventDefault();if(dialog.id==='restart-dialog'){$('restart-dialog').close();$('pause-dialog').showModal();}else closeDialog(dialog.id);}));
  $('next').addEventListener('click',continueJourney);
  $('sound').addEventListener('click',()=>{soundEnabled=!soundEnabled;$('sound').setAttribute('aria-pressed',String(soundEnabled));setLabel($('sound'),soundEnabled?(language==='en'?'Sound off':'소리 끄기'):(language==='en'?'Sound on':'소리 켜기'));if(soundEnabled)playTone(523,.13,.07);});
  document.querySelectorAll('[data-lang]').forEach(button=>button.addEventListener('click',()=>applyLanguage(button.dataset.lang)));
  window.addEventListener('resize',resize);
  mobileUI.addEventListener('change',resize);
  if(document.fonts)document.fonts.ready.then(resize);
}

async function init() {
  const [response,logoImage]=await Promise.all([fetch('./mazes.json?v=20260907-1'),new THREE.ImageLoader().loadAsync('./kist-60-original.png').catch(()=>{throw new Error(language==='ko'?'60주년 로고를 불러오지 못했어요. 다시 열어 주세요.':'The anniversary logo could not load. Please reload.');})]);if(!response.ok)throw new Error(copy().errorFetch);
  mazeTemplates=await response.json();session=createRandomSession();
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.75));renderer.setClearColor(0x10191e);
  renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.1;
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  renderer.domElement.setAttribute('aria-label','the KIST 60 ball exploring between the tall walls of a KIST-shaped maze.');
  $('game').appendChild(renderer.domElement);
  renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();mode='pause';clearInputs();$('error-message').textContent=copy().contextLost;$('load-error').hidden=false;});
  scene=new THREE.Scene();scene.background=new THREE.Color(0x10191e);playFog=new THREE.FogExp2(0x10191e,.038);scene.fog=playFog;
  camera=new THREE.PerspectiveCamera(48,innerWidth/innerHeight,.1,4000);
  makeEnvironment();scene.add(new THREE.HemisphereLight(0xe1f4fa,0x172631,2.5));
  const key=new THREE.DirectionalLight(0xf1faff,3.4);key.position.set(6,18,10);scene.add(key);
  const fill=new THREE.DirectionalLight(0xa4d1e3,1.6);fill.position.set(-10,8,-5);scene.add(fill);
  avatar=createAvatar(logoImage);scene.add(avatar);
  const lantern=new THREE.PointLight(0xd7e9b9,9,11,1.6);lantern.position.set(0,3.8,-.3);avatar.add(lantern);
  playerRing=new THREE.Mesh(new THREE.RingGeometry(1.1,1.12,64),new THREE.MeshBasicMaterial({color:0xe2f58a,side:THREE.DoubleSide,transparent:true,opacity:.3,depthWrite:false}));playerRing.rotation.x=-Math.PI/2;scene.add(playerRing);
  buildMaze();buildOverview(0);mazeGroup.visible=false;overviewGroup.visible=true;scene.fog=null;
  setupInputs();resize();const opening=getOverviewPose();camera.position.copy(opening.position);cameraLook.copy(opening.look);camera.lookAt(cameraLook);
  loaded=true;frameTime=performance.now()/1000;
  $('start').disabled=false;$('start').innerHTML=`<span>${copy().start}</span><span>↗</span>`;
  requestAnimationFrame(animate);
}
applyLanguage('en');
init().catch(error=>{console.error(error);$('error-message').textContent=error.message===copy().errorFetch||/anniversary logo|60주년 로고/.test(error.message)?error.message:copy().errorGraphics;$('load-error').hidden=false;});
