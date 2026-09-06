import * as THREE from './vendor/three.module.min.js';
import {MazeSession,DIRECTIONS,relativeDirection,hasLineOfSight,LOGO_COLOR,UNVISITED_COLOR,logoPosition,overviewCameraPose,mobileCameraLayout} from './maze-core.js?v=20260906-2';

const $=id=>document.getElementById(id);
const STAGE_INFO=[
  {name:'특허의 미로',objective:'출구를 찾아 특허를 출원하세요.',reward:'특허출원!',line:'첫 번째 발견을 세상에.',color:'#e2f58a'},
  {name:'논문의 미로',objective:'새로운 길에서 논문을 완성하세요.',reward:'논문출판!',line:'당신의 발견이 한 편의 논문으로.',color:'#93ddf5'},
  {name:'기술의 미로',objective:'기술이 세상으로 나갈 길을 찾으세요.',reward:'기술이전!',line:'연구실의 아이디어가 세상으로.',color:'#e3b1ff'},
  {name:'행운의 미로',objective:'마지막 출구에서 행운을 만나세요.',reward:'로또당첨!',line:'K → I → S → T, 네 글자를 모두 연결했어요!',color:'#ffcd75'}
];
const CELL=3.2,WALL_HEIGHT=3.3,VISION_RADIUS=5.3;
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)').matches;
const mobileUI=matchMedia('(max-width: 700px), (hover: none), (pointer: coarse)');
const touchControls=$('touch-controls'),projectedPlayer=new THREE.Vector3();
let mobileLayout=null;
let renderer,scene,camera,mazeGroup,avatar,body,legs=[],portal,portalRing,playerRing;
let floorMesh,wallMesh,capMesh,visitNumbers,wallData=[],visibleCells=new Set();
let overviewGroup,overviewMarker,cameraTween=null,playFog,fireworksStarted=false;
let celebrationHold=0,shownCountdown=-1;
let logoBounds={minX:Infinity,maxX:-Infinity,minZ:Infinity,maxZ:-Infinity};
const cameraLook=new THREE.Vector3();
const visitedColor=new THREE.Color(LOGO_COLOR),unvisitedColor=new THREE.Color(UNVISITED_COLOR);
let session,mode='intro',modeBeforeDialog='intro',animation=null,held=null,queued=null;
let playerPosition=new THREE.Vector3(),followPosition=new THREE.Vector3(),yaw=0,targetYaw=0;
let frameTime=0,walkPhase=0,nextInputAt=0,lastCollisionAt=-5,toastTimeout,loaded=false;
let audioContext,soundEnabled=false,fireworkTime=0,fireworkNext=0,particles=[],rockets=[];
const fireworks=$('fireworks'),fx=fireworks.getContext('2d');
const object=new THREE.Object3D(),explorerUniform={value:new THREE.Vector3()};
const temporary=new THREE.Vector3();
const shadeMaterials=[];

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

function createAvatar() {
  const character=new THREE.Group();
  const red=new THREE.MeshPhysicalMaterial({color:0xee080e,roughness:.17,metalness:.12,clearcoat:1,clearcoatRoughness:.08,envMapIntensity:1.55});
  function ellipsoid(parent,x,y,z,sx,sy,sz) {
    const mesh=new THREE.Mesh(new THREE.SphereGeometry(1,32,24),red);
    mesh.position.set(x,y,z);mesh.scale.set(sx,sy,sz);mesh.castShadow=true;mesh.receiveShadow=true;parent.add(mesh);return mesh;
  }
  body=new THREE.Group();character.add(body);
  const profile=[[0,.5],[.42,.56],[.65,.72],[.79,1.02],[.80,1.29],[.72,1.58],[.57,1.88],[.35,2.04],[0,2.08]].map(([x,y])=>new THREE.Vector2(x,y));
  const torso=new THREE.Mesh(new THREE.LatheGeometry(profile,48),red);torso.scale.z=.87;torso.position.z=.08;torso.castShadow=true;torso.receiveShadow=true;body.add(torso);
  ellipsoid(body,0,1.91,-.18,.33,.27,.33);
  const head=ellipsoid(body,0,2.27,-.24,.56,.57,.53);head.rotation.x=-.12;
  const armL=ellipsoid(body,-.64,1.23,-.09,.22,.52,.28);armL.rotation.z=-.17;
  const armR=ellipsoid(body,.64,1.23,-.09,.22,.52,.28);armR.rotation.z=.17;
  legs=[];
  for(const side of [-1,1]) {
    const leg=new THREE.Group();leg.position.set(side*.33,.68,.03);
    const calf=new THREE.Mesh(new THREE.CapsuleGeometry(.175,.34,6,16),red);calf.position.set(0,-.22,-.04);calf.castShadow=true;leg.add(calf);
    ellipsoid(leg,0,-.5,-.16,.19,.16,.31);character.add(leg);legs.push(leg);
  }
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
  const stage=session.stage,info=STAGE_INFO[session.stageIndex];
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
  targetYaw=-session.heading*Math.PI/2;yaw=targetYaw;avatar.position.copy(playerPosition);avatar.rotation.y=yaw;
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

function buildOverview(entranceIndex=session.stageIndex+1) {
  disposeGroup(overviewGroup);overviewGroup=new THREE.Group();scene.add(overviewGroup);
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
  }
  const overviewNumbers=makeVisitNumbers(visitedTiles.reduce((total,tile)=>total+String(tile.count).length,0));
  overviewNumbers.material.fog=false;setVisitNumbers(overviewNumbers,visitedTiles);overviewGroup.add(overviewNumbers);
  // The next entrance is marked on its actual letter, in the original logo layout.
  overviewMarker=null;
  if(entranceIndex<session.stages.length) {
    const nextStage=session.stages[entranceIndex];
    overviewMarker=new THREE.Group();overviewMarker.position.fromArray(logoPosition(nextStage,nextStage.start,CELL));
    const markerMaterial=new THREE.MeshBasicMaterial({color:0xe2f58a,fog:false,toneMapped:false,side:THREE.DoubleSide});
    const ring=new THREE.Mesh(new THREE.RingGeometry(2.6,3.2,48),markerMaterial);ring.rotation.x=-Math.PI/2;ring.position.y=3.5;overviewMarker.add(ring);
    const stem=new THREE.Mesh(new THREE.CylinderGeometry(.16,.16,8,8),markerMaterial);stem.position.y=7;overviewMarker.add(stem);
    const letter=makeTextSprite(nextStage.letter,150);letter.position.y=15;letter.scale.set(11,11,1);letter.material.fog=false;letter.material.toneMapped=false;overviewMarker.add(letter);
    overviewGroup.add(overviewMarker);
  }
  overviewGroup.visible=false;
}

function getOverviewPose() {
  const pose=overviewCameraPose(logoBounds,camera.aspect,camera.fov);
  return {position:new THREE.Vector3(...pose.position),look:new THREE.Vector3(...pose.look)};
}
function getPlayPose() {
  const distance=mobileLayout?.distance??6.7;
  const look=playerPosition.clone();look.y=mobileLayout?1.35:1.05;
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
  const lift=Math.max(38,Math.abs(to.y-from.y)*.32);
  const control1=from.clone().add(new THREE.Vector3(0,type==='out'?lift:25,0));
  const control2=to.clone().add(new THREE.Vector3(0,type==='in'?lift:25,0));
  cameraTween={type,progress:0,duration:reducedMotion?.35:type==='out'?2.8:2.6,
    fromFrame:camera.view?.enabled?camera.view.offsetY:0,toFrame:type==='in'?(mobileLayout?.offsetY??0):0,
    curve:new THREE.CubicBezierCurve3(from,control1,control2,to),fromLook:cameraLook.clone(),toLook:destination.look.clone()};
}
function updateCameraMove(dt) {
  const move=cameraTween;move.progress=Math.min(1,move.progress+dt/move.duration);
  const t=move.progress,ease=t*t*t*(t*(t*6-15)+10);
  setCameraFrame(THREE.MathUtils.lerp(move.fromFrame,move.toFrame,ease));
  camera.position.copy(move.curve.getPoint(ease));cameraLook.lerpVectors(move.fromLook,move.toLook,ease);camera.lookAt(cameraLook);
  if(move.type==='out'&&t>=.5&&!fireworksStarted) {
    fireworksStarted=true;$('celebration').classList.remove('revealing');
    burst(innerWidth*.5,innerHeight*.31,STAGE_INFO[session.stageIndex].color,110);
    [523.25,659.25,783.99,1046.5].forEach((frequency,i)=>playTone(frequency,.3,.1,'triangle',i*.12));
  }
  if(t<1)return;
  cameraTween=null;
  if(move.type==='out') {
    mode='celebration';celebrationHold=0;shownCountdown=-1;$('next').disabled=false;$('next').focus();
  } else {
    mode='play';mazeGroup.visible=true;overviewGroup.visible=false;scene.fog=playFog;
    document.body.classList.remove('overview-mode');
    toast(`${session.stage.letter} 입구에 도착했어요. 다음 출구를 찾아보세요.`);
  }
}

function updateHUD() {
  const info=STAGE_INFO[session.stageIndex];
  $('zone-index').textContent=`ZONE 0${session.stageIndex+1} / 04`;
  $('zone-letter').textContent=session.stage.letter;$('zone-name').textContent=info.name;$('zone-objective').textContent=info.objective;
  $('steps').textContent=session.steps.toLocaleString();
  $('stage-timer').textContent=formatPreciseTime(session.stageElapsed);
  $('compass-needle').style.transform=`rotate(${session.heading*-90}deg)`;
  $('heading-label').textContent=`${['북쪽','동쪽','남쪽','서쪽'][session.heading]}을 보는 중`;
  document.querySelectorAll('.stage').forEach((el,i)=>{
    el.classList.toggle('current',i===session.stageIndex&&!session.finished);el.classList.toggle('complete',session.cleared.includes(i));
    el.setAttribute('aria-label',`${'KIST'[i]} ${session.cleared.includes(i)?'완료':i===session.stageIndex?'탐험 중':'아직 도착하지 않음'}`);
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
  $('heading-label').textContent=`${['북쪽','동쪽','남쪽','서쪽'][session.heading]}을 보는 중`;
  if(!result.allowed) {
    if(now-lastCollisionAt>.95){toast('막힌 길이에요. 다른 방향을 찾아보세요.');playTone(95,.08,.05,'sine');lastCollisionAt=now;}
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
function openDialog(id) {
  if(!loaded||['celebration','zoomOut','zoomIn'].includes(mode))return;
  if(!document.querySelector('dialog[open]'))modeBeforeDialog=mode;
  clearInputs();mode='pause';$(id).showModal();
}
function closeDialog(id) {$(id).close();if(!document.querySelector('dialog[open]'))mode=modeBeforeDialog;clearInputs();}
function restartGame() {
  document.querySelectorAll('dialog[open]').forEach(d=>d.close());
  $('celebration').hidden=true;particles=[];rockets=[];clearInputs();animation=null;cameraTween=null;
  if(overviewGroup)overviewGroup.visible=false;scene.fog=playFog;document.body.classList.remove('overview-mode');
  session.reset();buildMaze();mode='play';modeBeforeDialog='play';document.body.classList.add('playing');
  $('timer').textContent='00:00';toast('K 구역에서 새로운 탐험을 시작해요.');
}
function completeStage() {
  mode='zoomOut';session.recordStage();clearInputs();updateHUD();
  const index=session.stageIndex,info=STAGE_INFO[index];
  $('celebration').style.setProperty('--lime',info.color);
  $('celebration-eyebrow').textContent=index===3?'KIST · ALL ZONES COMPLETE':`${session.stage.letter} ZONE COMPLETE`;
  $('celebration-letter').textContent=session.stage.letter;
  $('celebration-title').textContent=info.reward;$('celebration-description').textContent=info.line;
  $('celebration-stats').textContent=index===3?`${session.steps.toLocaleString()}걸음 · 4개 구역 완주`:`${session.stage.letter} 통과 ${formatPreciseTime(session.stageTimes[index])} · ${session.stageSteps.toLocaleString()}걸음`;
  $('final-times').hidden=index!==3;
  $('celebration').classList.toggle('all-complete',index===3);
  if(index===3) {
    $('final-total-time').textContent=formatPreciseTime(session.totalClearTime);
    $('final-stage-times').innerHTML=session.stageTimes.map((time,i)=>`<div><b>${'KIST'[i]}</b><time>${formatPreciseTime(time)}</time></div>`).join('');
  }
  $('next').innerHTML=index===3?'다시 탐험하기 <span>↻</span>':`${'KIST'[index+1]} 입구로 들어가기 <span>→</span>`;
  $('next').disabled=true;$('celebration').classList.add('revealing');$('celebration').hidden=false;
  particles=[];rockets=[];fireworkTime=0;fireworkNext=0;
  fireworksStarted=false;buildOverview();overviewGroup.visible=true;mazeGroup.visible=false;scene.fog=null;
  document.body.classList.add('overview-mode');$('toast').classList.remove('show');
  beginCameraMove('out',getOverviewPose());
}
function continueJourney() {
  if(mode!=='celebration')return;
  if(session.stageIndex===3){session.reset();buildOverview(0);overviewGroup.visible=true;}
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
    const palette=[STAGE_INFO[session.stageIndex].color,'#ff7675','#9ce9ff','#ffffff','#dfafff'];
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
    const destination=cameraTween.type==='out'?getOverviewPose():getPlayPose();
    cameraTween.curve.v3.copy(destination.position);cameraTween.toLook.copy(destination.look);
    cameraTween.toFrame=cameraTween.type==='in'?(mobileLayout?.offsetY??0):0;
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
      $('next').innerHTML=`${'KIST'[session.stageIndex+1]} 입구로 이동 · ${remaining}<span>→</span>`;
    }
    if(celebrationHold>=5)continueJourney();
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
  const walking=mode==='play'&&!!animation;
  walkPhase+=dt*(walking?12:2);
  body.position.y=walking&&!reducedMotion?Math.abs(Math.sin(walkPhase))*.065:Math.sin(now*1.6)*.018;
  body.rotation.z=walking&&!reducedMotion?Math.sin(walkPhase)*.025:0;
  for(let i=0;i<legs.length;i++)legs[i].rotation.x=THREE.MathUtils.damp(legs[i].rotation.x,walking&&!reducedMotion?Math.sin(walkPhase+i*Math.PI)*.37:0,14,dt);
  yaw=THREE.MathUtils.damp(yaw,targetYaw,8,dt);avatar.rotation.y=yaw;avatar.position.copy(playerPosition);
  followPosition.lerp(playerPosition,1-Math.exp(-12*dt));
  const introMode=mode==='intro'||(mode==='pause'&&modeBeforeDialog==='intro');
  if(cameraTween) updateCameraMove(dt);
  else if(mode==='celebration'||introMode) {
    setCameraFrame(0);
    const pose=getOverviewPose();camera.position.copy(pose.position);cameraLook.copy(pose.look);camera.lookAt(cameraLook);
  } else {
    const distance=mobileLayout?.distance??6.7;
    setCameraFrame(mobileLayout?.offsetY??0);cameraLook.copy(followPosition);cameraLook.y=mobileLayout?1.35:1.05;
    camera.position.set(cameraLook.x+Math.sin(yaw)*distance,cameraLook.y+distance,cameraLook.z+Math.cos(yaw)*distance);camera.lookAt(cameraLook);
  }
  explorerUniform.value.copy(playerPosition);if(mazeGroup.visible)updateWalls(dt);
  playerRing.position.set(playerPosition.x,.025,playerPosition.z);
  playerRing.material.opacity=.27+Math.sin(now*1.7)*.055;
  if(portalRing){portalRing.rotation.z=now*.16;portalRing.scale.setScalar(1+Math.sin(now*2)*.045);}
  if(overviewMarker)overviewMarker.scale.setScalar(1+Math.sin(now*2)*.045);
  if(mazeGroup.visible)orientVisitNumbers(visitNumbers,yaw);
  renderer.render(scene,camera);
  updateTouchControls();
  if((mode==='celebration'||mode==='zoomOut')&&fireworksStarted)animateFireworks(dt);
}

function updateTouchControls() {
  if(!mobileLayout||mode!=='play')return;
  // Track the actual projected body while the chase camera moves and turns.
  camera.updateMatrixWorld();projectedPlayer.copy(playerPosition);projectedPlayer.y+=1.35;projectedPlayer.project(camera);
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
  $('pause').addEventListener('click',()=>openDialog('pause-dialog'));
  $('resume').addEventListener('click',()=>closeDialog('pause-dialog'));
  $('restart').addEventListener('click',()=>{$('pause-dialog').close();$('restart-dialog').showModal();});
  $('cancel-restart').addEventListener('click',()=>{$('restart-dialog').close();$('pause-dialog').showModal();});
  $('confirm-restart').addEventListener('click',restartGame);
  document.querySelectorAll('[data-close]').forEach(button=>button.addEventListener('click',()=>closeDialog(button.dataset.close)));
  document.querySelectorAll('dialog').forEach(dialog=>dialog.addEventListener('cancel',event=>{event.preventDefault();if(dialog.id==='restart-dialog'){$('restart-dialog').close();$('pause-dialog').showModal();}else closeDialog(dialog.id);}));
  $('next').addEventListener('click',continueJourney);
  $('sound').addEventListener('click',()=>{soundEnabled=!soundEnabled;$('sound').setAttribute('aria-pressed',String(soundEnabled));$('sound').setAttribute('aria-label',soundEnabled?'소리 끄기':'소리 켜기');$('sound').title=soundEnabled?'소리 끄기':'소리 켜기';if(soundEnabled)playTone(523,.13,.07);});
  window.addEventListener('resize',resize);
  mobileUI.addEventListener('change',resize);
  if(document.fonts)document.fonts.ready.then(resize);
}

async function init() {
  const response=await fetch('./mazes.json?v=20260906-2');if(!response.ok)throw new Error('미로 데이터를 불러오지 못했어요. 다시 열어 주세요.');
  const stages=await response.json();session=new MazeSession(stages);
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.75));renderer.setClearColor(0x10191e);
  renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.1;
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  renderer.domElement.setAttribute('aria-label','높은 벽 사이를 걷는 빨간 달콤한 뚱땡이. 방향키로 조작하세요.');
  $('game').appendChild(renderer.domElement);
  renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();mode='pause';clearInputs();$('error-message').textContent='3D 화면 연결이 잠시 끊겼어요. 다시 열기를 눌러 주세요.';$('load-error').hidden=false;});
  scene=new THREE.Scene();scene.background=new THREE.Color(0x10191e);playFog=new THREE.FogExp2(0x10191e,.038);scene.fog=playFog;
  camera=new THREE.PerspectiveCamera(48,innerWidth/innerHeight,.1,4000);
  makeEnvironment();scene.add(new THREE.HemisphereLight(0xe1f4fa,0x172631,2.5));
  const key=new THREE.DirectionalLight(0xf1faff,3.4);key.position.set(6,18,10);scene.add(key);
  const fill=new THREE.DirectionalLight(0xa4d1e3,1.6);fill.position.set(-10,8,-5);scene.add(fill);
  avatar=createAvatar();scene.add(avatar);
  const lantern=new THREE.PointLight(0xd7e9b9,9,11,1.6);lantern.position.set(0,3.8,-.3);avatar.add(lantern);
  playerRing=new THREE.Mesh(new THREE.RingGeometry(.94,.96,64),new THREE.MeshBasicMaterial({color:0xe2f58a,side:THREE.DoubleSide,transparent:true,opacity:.3,depthWrite:false}));playerRing.rotation.x=-Math.PI/2;scene.add(playerRing);
  buildMaze();buildOverview(0);mazeGroup.visible=false;overviewGroup.visible=true;scene.fog=null;
  setupInputs();resize();const opening=getOverviewPose();camera.position.copy(opening.position);cameraLook.copy(opening.look);camera.lookAt(cameraLook);
  loaded=true;frameTime=performance.now()/1000;
  $('start').disabled=false;$('start').innerHTML='<span>탐험 시작하기</span><span>↗</span>';
  requestAnimationFrame(animate);
}
init().catch(error=>{console.error(error);$('error-message').textContent=error.message.includes('미로')?error.message:'3D 화면을 시작하지 못했어요. WebGL을 지원하는 최신 브라우저에서 다시 열어 주세요.';$('load-error').hidden=false;});
