import {BakeryGame,STAGES,STAGE_SECONDS,COOKIE_SECONDS,PASS_SCORE,isCorrect} from './engine.js?v=family-snacks-4';
import {isGameFullscreen,enterGameDisplay,exitGameDisplay} from './display.js?v=family-snacks-4';
import {createPastryTiles,SNACK_MENUS} from './pastries.js?v=jeongan-magic-5';
import {PLAYERS,POSES,getPlayer,EATING_MOUTHS,SPRITE_RECTS,keySpriteMatte} from './family.js';

const $=id=>document.getElementById(id);
const game=new BakeryGame();
const PREP_SECONDS=5;
const canvas=$('game-canvas'),ctx=canvas.getContext('2d');
const sceneWrap=document.querySelector('.scene-wrap');
let displayPending=false,rotationRequired=false,fullscreenBlocked=false,fullscreenModalOpen=false;
let fullscreenModalBackup=null;
const currentCtx=$('current-cookie').getContext('2d'),targetCtx=$('target-cookie').getContext('2d');
const art={},crop=[[140,32,302,459],[613,31,307,460],[1112,33,304,458],[139,538,248,462],[608,491,363,509],[1079,492,424,508]];
let loaded=false,paused=false,helpOpen=false,transitionTime=0,clock=0,lastFrame=0,lastHud=-1,expression='neutral',expressionUntil=0,soundEnabled=false,audioCtx=null,returnFocus=null;
let cookieTiles=[],letteringStage=0;
let selectedPlayer=null,assetsLoading=false,assetError=false,familyFrames=[];
const playerStages=()=>SNACK_MENUS[selectedPlayer?.id]?.stages??STAGES;
const stageDetails=()=>playerStages()[game.stage-1];
const selectionHTML=$('modal-card').innerHTML;
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)').matches;
const demoCookies=[{id:0,arrival:0,rotation:1,flipped:false},{id:1,arrival:2,rotation:3,flipped:true},{id:2,arrival:4,rotation:2,flipped:false}];

function loadImage(name,path){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>{art[name]=img;resolve();};img.onerror=()=>reject(new Error(path));img.src=path;});}
async function loadAssets(){
 if(assetsLoading)return;assetsLoading=true;assetError=false;refreshSelection();$('help-button').disabled=true;
 try{
  await Promise.all([loadImage('background','./assets/bakery-background.webp'),loadImage('characters','./assets/bakery-characters.webp'),loadImage('family','./assets/family-characters.png'),loadImage('jeonganHappy','./assets/jeongan-happy.webp').catch(()=>{})]);
  cookieTiles=createPastryTiles('bakery-preview',selectedPlayer?.id??'dad');
  prepareFamilyFrames();loaded=true;assetsLoading=false;$('help-button').disabled=false;refreshSelection();
  drawLettering();
 }catch{
  loaded=false;assetsLoading=false;assetError=true;refreshSelection();
 }
}

function prepareFamilyFrames(){
 familyFrames=SPRITE_RECTS.map(([sx,sy,right,bottom])=>{
  const cell=document.createElement('canvas');cell.width=right-sx;cell.height=bottom-sy;const c=cell.getContext('2d',{willReadFrequently:true});c.drawImage(art.family,sx,sy,cell.width,cell.height,0,0,cell.width,cell.height);
  const pixels=c.getImageData(0,0,cell.width,cell.height);keySpriteMatte(pixels.data,cell.width,cell.height);c.putImageData(pixels,0,0);return cell;
 });
 // The original Jeongan celebration contains tears; replace only that pose.
 const jeongan=getPlayer('jeongan'),happyIndex=POSES.happy*PLAYERS.length+jeongan.column;
 familyFrames[happyIndex]=familyFrames[jeongan.column];
 if(art.jeonganHappy){
  const cell=document.createElement('canvas');cell.width=863;cell.height=1206;
  const c=cell.getContext('2d',{willReadFrequently:true});c.drawImage(art.jeonganHappy,198,18,863,1206,0,0,863,1206);
  const pixels=c.getImageData(0,0,cell.width,cell.height);keySpriteMatte(pixels.data,cell.width,cell.height);c.putImageData(pixels,0,0);
  familyFrames[happyIndex]=cell;
 }
}
function refreshSelection(){
 const button=$('start-button'),note=$('load-note');if(!button)return;
 button.disabled=assetsLoading||(!assetError&&(!loaded||!selectedPlayer));
 button.textContent=assetError?'그림 다시 불러오기':assetsLoading?'가족을 불러오는 중…':selectedPlayer?`${selectedPlayer.as} 시작하기 →`:'이름을 선택해 주세요';
 if(note)note.textContent=assetError?'그림을 불러오지 못했어요. 다시 눌러 주세요.':assetsLoading?'함께할 가족을 불러오고 있어요…':'시작하면 가로 · 전체화면으로 열려요 · 30개 중 25개 성공!';
 for(const input of document.querySelectorAll('input[name="player"]')){input.disabled=!loaded;input.checked=input.value===selectedPlayer?.id;}
 for(const portrait of document.querySelectorAll('[data-portrait]')){
  const c=portrait.getContext('2d');c.clearRect(0,0,160,160);const player=getPlayer(portrait.dataset.portrait);if(!loaded||!player)continue;
  const sprite=familyFrames[player.column],w=sprite.width,h=sprite.height;const headHeight=h*.47;const size=Math.max(w,headHeight);c.drawImage(sprite,-(size-w)/2,0,size,size,0,0,160,160);
 }
 $('player-caption').textContent=selectedPlayer?`${selectedPlayer.name}의 달콤한 도전`:'조금 삐뚤어도, 맛있을 거야!';
}
function selectPlayer(id){const player=getPlayer(id);if(!loaded||game.phase!=='ready'||!player||helpOpen)return;selectedPlayer=player;cookieTiles=createPastryTiles('bakery-preview',player.id);refreshSelection();drawLettering();updateHud(true);$('announcer').textContent=`${player.name} 선택. 포장할 간식: ${SNACK_MENUS[player.id]?.label??'버터 과자'}. 시작 버튼을 눌러 주세요.`;}
function showCharacterSelection(){
 document.querySelector('.bakery-app').classList.remove('game-started');fullscreenModalOpen=false;fullscreenModalBackup=null;game.reset();selectedPlayer=null;paused=false;helpOpen=false;expression='neutral';expressionUntil=0;transitionTime=0;$('scene-feedback').textContent='';
 if(loaded)cookieTiles=createPastryTiles('bakery-preview','dad');
 $('stage-intro').classList.add('hidden');$('tasting').classList.add('hidden');sceneWrap.classList.add('choosing-player');document.querySelector('.game-shell').classList.add('choosing-player');showModal(selectionHTML,'character-select');refreshSelection();updateHud(true);updateDisplayState();
}
function familyCharacter(c,pose,cx,bottom,height){
 if(!loaded||!selectedPlayer)return null;const sprite=familyFrames[pose*4+selectedPlayer.column];const w=height*sprite.width/sprite.height;const x=cx-w/2,y=bottom-height;c.drawImage(sprite,x,y,w,height);return {x,y,w,h:height};
}

function rounded(c,x,y,w,h,r,fill,stroke){c.beginPath();c.roundRect(x,y,w,h,r);if(fill){c.fillStyle=fill;c.fill();}if(stroke){c.strokeStyle=stroke;c.lineWidth=2;c.stroke();}}
function character(c,index,cx,bottom,height){if(!art.characters)return;const [sx,sy,sw,sh]=crop[index];const w=height*sw/sh;c.drawImage(art.characters,sx,sy,sw,sh,cx-w/2,bottom-height,w,height);}
function cookie(c,stage,x,y,size,o={rotation:0,flipped:false},alpha=1,animate=true){
 if(!cookieTiles.length)return;c.save();c.translate(x,y);c.globalAlpha=alpha;
 let angle=o.rotation*Math.PI/2,sy=o.flipped?-1:1;
 const a=o.animation,progress=a?Math.min(1,(game.elapsed-a.at)/.14):1;
 if(animate&&a&&progress<1&&!reducedMotion){
  const ease=1-(1-progress)**3;sy=a.fromFlipped?-1:1;
  if(a.action==='left'||a.action==='right')angle=(a.fromRotation+(a.action==='left'?-1:1)*ease)*Math.PI/2;
  else{
   angle=a.fromRotation*Math.PI/2;
   // Reflect the screen axis before applying the pastry's original orientation.
   const fold=Math.cos(Math.PI*ease);
   c.scale(a.action==='down'?fold:1,a.action==='up'?fold:1);
  }
 }
 c.rotate(angle);c.scale(1,sy);c.drawImage(cookieTiles[stage-1],-size/2,-size/2,size,size);c.restore();
}
function box(c,x,y,size=108,ghost=false,stage=game.stage){
 c.save();rounded(c,x-size/2,y-size*.36+13,size,size*.73,5,'#b88150','#8f5d39');rounded(c,x-size/2,y-size*.43,size,size*.75,5,'#e9c99e','#a47649');rounded(c,x-size*.42,y-size*.36,size*.84,size*.6,3,'#f8e8ca','#ceaa7e');
 if(ghost)cookie(c,stage,x,y-3,size*.77,{rotation:0,flipped:false},.27,false);c.restore();
}
function drawFloor(){
 ctx.save();ctx.beginPath();ctx.moveTo(697,407);ctx.lineTo(985,407);ctx.lineTo(1030,579);ctx.lineTo(664,579);ctx.closePath();ctx.clip();
 ctx.fillStyle='#b7c6a9';ctx.fillRect(650,407,400,180);
 for(let row=0;row<4;row++)for(let col=0;col<6;col++){ctx.fillStyle=(row+col)%2?'#dce1bb':'#bbc6a6';ctx.fillRect(651+col*66,408+row*47,64,45);}
 ctx.restore();ctx.save();ctx.strokeStyle='#f6edc9';ctx.lineWidth=3;ctx.setLineDash([7,7]);ctx.strokeRect(710,437,264,100);ctx.restore();
 ctx.fillStyle='#a5714f';ctx.beginPath();ctx.ellipse(1050,491,90,31,0,0,Math.PI*2);ctx.fill();ctx.fillStyle='#422c26';ctx.beginPath();ctx.ellipse(1050,487,83,27,0,0,Math.PI*2);ctx.fill();ctx.fillStyle='#291e1b';ctx.beginPath();ctx.ellipse(1050,491,73,19,0,0,Math.PI*2);ctx.fill();
}
function drawBelt(t){
 for(const x of [65,405,695,1140]){rounded(ctx,x,411,17,90,4,'#805d47','#634835');rounded(ctx,x-10,491,39,9,4,'#75523e');}
 rounded(ctx,-40,347,1290,97,35,'#725b4c','#634635');rounded(ctx,-40,347,1290,75,35,'#bea88b','#674d3c');
 ctx.save();ctx.beginPath();ctx.roundRect(-40,350,1290,65,30);ctx.clip();ctx.fillStyle='#ad987d';ctx.fillRect(0,350,1200,66);ctx.fillStyle='#c0ad91';ctx.fillRect(0,354,1200,18);
 const offset=(t*125)%37;for(let x=-40+offset;x<1260;x+=37){ctx.strokeStyle='#89765e';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,349);ctx.lineTo(x,416);ctx.stroke();ctx.strokeStyle='#d5c2a2';ctx.beginPath();ctx.moveTo(x+3,349);ctx.lineTo(x+3,416);ctx.stroke();}
 ctx.fillStyle='#ede8a42b';ctx.fillRect(707,350,266,66);ctx.restore();
 rounded(ctx,-20,343,1240,9,4,'#ddc6a4','#85664d');rounded(ctx,-20,413,1240,10,4,'#d4b18b','#846044');
 for(let x=30;x<1200;x+=75){ctx.fillStyle='#d2b18e';ctx.beginPath();ctx.arc(x,434,3,0,Math.PI*2);ctx.fill();}
 ctx.strokeStyle='#fbefb7';ctx.lineWidth=4;ctx.setLineDash([7,5]);for(const x of [706,975]){ctx.beginPath();ctx.moveTo(x,348);ctx.lineTo(x,416);ctx.stroke();}ctx.setLineDash([]);
}
function drawScene(){
 ctx.clearRect(0,0,1200,600);
 if(art.background){ctx.drawImage(art.background,0,0,art.background.width,art.background.height,0,-55,1200,675);}else{ctx.fillStyle='#e8c49e';ctx.fillRect(0,0,1200,600);}
 if(!loaded)return;
 ctx.save();
 drawFloor();
 const t=game.cookies.length?(game.phase==='intro'?-transitionTime:game.elapsed):clock*.32;
 const playerPose=expressionUntil>clock?(POSES[expression]??POSES.neutral):POSES.neutral;
 character(ctx,((t%2+2)%2<.52)?5:4,196,427,311);
 const bob=reducedMotion||paused?0:Math.sin(clock*2.5)*2;
 familyCharacter(ctx,playerPose,873,424+bob,selectedPlayer?.height??253);
 if(selectedPlayer){const nameY=424-selectedPlayer.height-34;rounded(ctx,827,nameY,92,27,13,'#fff7e4e8','#b98c61');ctx.font='600 17px sans-serif';ctx.textAlign='center';ctx.fillStyle='#87513b';ctx.fillText(selectedPlayer.name,873,nameY+19);}
 box(ctx,1100,307,99,false);box(ctx,1095,292,99,false);box(ctx,1102,276,99,true);
 ctx.font='600 14px sans-serif';ctx.textAlign='center';ctx.fillStyle='#fff5dc';ctx.fillText('상자 속 정답',1100,219);
 drawBelt(t);
 const list=game.cookies.length?game.cookies:demoCookies;
 for(const c of list){
  let x=720+125*((game.cookies.length?t:0)-c.arrival),y=381;
  if(x<(game.phase==='intro'?45:145)||x>1290)continue;
  let scale=1,opacity=1;
  const failure=c.result==='failure';
  if(failure&&x>1005){const fall=(x-1005)/125;x=1005+Math.min(fall*90,45);y+=fall*130+fall*fall*290;scale=Math.max(.15,1-fall*.75);opacity=Math.max(0,1-fall*.8);if(y>523)continue;}
  ctx.save();ctx.globalAlpha=opacity;ctx.translate(x,y);ctx.scale(scale,scale);
  if(c.result)box(ctx,0,0,109,false);
  else {ctx.fillStyle='#63402a20';ctx.beginPath();ctx.ellipse(0,21,39,13,0,0,Math.PI*2);ctx.fill();}
  cookie(ctx,game.stage,0,-4,c.result?90:102,c,1);
  if(c.result){ctx.save();ctx.globalAlpha=opacity;ctx.font='900 61px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.lineWidth=7;ctx.strokeStyle='#fff8e8';ctx.strokeText(c.result==='success'?'O':'X',0,-14);ctx.fillStyle=c.result==='success'?'#438655':'#c34c45';ctx.fillText(c.result==='success'?'O':'X',0,-14);ctx.restore();}
  else if(c===game.active){ctx.strokeStyle='#fffbda';ctx.lineWidth=3;ctx.setLineDash([5,5]);ctx.beginPath();ctx.ellipse(0,0,60,49,0,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);}
  ctx.restore();
 }
 // Steam floats above the baker's newly placed pastry.
 if(!reducedMotion){ctx.save();for(let i=0;i<3;i++){const p=(clock*.4+i/3)%1;ctx.globalAlpha=(1-p)*.45;ctx.strokeStyle='#fff7e3';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(239+i*11,337-p*43);ctx.quadraticCurveTo(226+i*11,321-p*43,242+i*11,310-p*43);ctx.stroke();}ctx.restore();}
 ctx.restore();
}

function drawPreviews(){
 currentCtx.clearRect(0,0,180,180);targetCtx.clearRect(0,0,180,180);
 const i=Math.min(29,Math.floor(game.elapsed/2));const c=game.cookies[i]??demoCookies[0];
 cookie(currentCtx,game.stage,90,90,166,c);cookie(targetCtx,game.stage,90,90,166,{rotation:0,flipped:false},.62,false);
}
function drawLettering(){
 const c=$('stage-lettering').getContext('2d');c.clearRect(0,0,1000,190);const text=`STAGE ${game.stage}`;
 c.font='900 145px Georgia,serif';c.textAlign='center';c.textBaseline='middle';c.lineJoin='round';c.strokeStyle='#8c542e';c.lineWidth=15;c.strokeText(text,500,102);c.strokeStyle='#edbd73';c.lineWidth=7;c.strokeText(text,500,97);
 const g=c.createLinearGradient(0,20,0,170);g.addColorStop(0,'#f8d893');g.addColorStop(.55,stageDetails().color);g.addColorStop(1,'#b77939');c.fillStyle=g;c.fillText(text,500,97);
 if(cookieTiles.length){const pattern=c.createPattern(cookieTiles[game.stage-1],'repeat');c.globalAlpha=.62;c.fillStyle=pattern;c.fillText(text,500,97);c.globalAlpha=1;}
 c.save();c.globalCompositeOperation='source-atop';c.fillStyle='#7b3f24';for(let i=0;i<56;i++){const x=55+(i*113)%900,y=40+(i*37)%116;c.beginPath();c.ellipse(x,y,2.5,1.6,i,0,Math.PI*2);c.fill();}c.restore();
 $('stage-lettering').setAttribute('aria-label',`STAGE ${game.stage}`);$('intro-flavor').textContent=stageDetails().name;letteringStage=game.stage;
}
function drawTasting(){
 const c=$('tasting-canvas').getContext('2d');c.clearRect(0,0,460,440);
 const frame=familyCharacter(c,POSES.eating,230,680,690);if(!frame)return;
 const mouth=EATING_MOUTHS[selectedPlayer.column];cookie(c,game.stage,frame.x+frame.w*mouth.x,frame.y+frame.h*mouth.y,110,{rotation:0,flipped:false},1,false);
}
function updateHud(force=false){
 if(!force&&clock-lastHud<.07)return;lastHud=clock;
 $('stage-number').textContent=String(game.stage).padStart(2,'0');$('stage-flavor').textContent=stageDetails().name;
 const seconds=Math.max(0,Math.ceil(STAGE_SECONDS-game.elapsed));$('time').textContent=`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;$('time').classList.toggle('low-time',seconds<=10);
 $('score').textContent=game.score;$('score-fill').style.width=`${Math.min(100,game.score/PASS_SCORE*100)}%`;$('score-progress').setAttribute('aria-valuenow',Math.min(PASS_SCORE,game.score));
 [...$('hearts').children].forEach((h,i)=>h.classList.toggle('lost',i>=game.lives));$('hearts').setAttribute('aria-label',`남은 기회 ${game.lives}번`);
 const preparing=game.phase==='intro',active=game.active;const waiting=game.phase==='playing'&&!active;const remaining=preparing?transitionTime:active?Math.max(0,active.deadline-game.elapsed):waiting?0:2;
 $('window-time').textContent=preparing?String(Math.ceil(remaining)):remaining.toFixed(1);$('window-fill').style.width=`${remaining/(preparing?PREP_SECONDS:COOKIE_SECONDS)*100}%`;$('window-fill').style.background=preparing?'#4b8461':remaining<.6?'#b54f40':'#be9154';
 $('window-label').textContent=preparing?'시작까지':'이번 과자';$('current-cookie-label').textContent=preparing?'첫 과자':'지금 내 과자';
 if(preparing){$('prep-countdown').textContent=Math.ceil(remaining);$('stage-intro').classList.toggle('compact',remaining<=PREP_SECONDS-1);}
 const previous=game.cookies[Math.min(29,Math.floor(game.elapsed/2))];const correct=active?isCorrect(active):previous?.result==='success';
 $('match-badge').textContent=preparing?'미리 살펴보세요':correct?'모양이 딱 맞아요!':waiting?'다음 과자 준비 중':'돌려 주세요';$('match-badge').style.color=correct?'#438655':'#9d795d';
 $('status-tip').innerHTML=preparing?'첫 과자가 오는 동안 정답을 살펴보세요.':game.score>=25?'목표 달성! 끝까지 포장해요.':waiting?'다음 과자가 오고 있어요.':'상자 속 희미한 그림과<br>모양도, 장식도 똑같이!';
 $('pause-button').disabled=!['playing','intro','tasting'].includes(game.phase);
}

function tone(freq,duration=.08,type='sine',delay=0){if(!soundEnabled)return;try{audioCtx??=new(window.AudioContext||window.webkitAudioContext)();audioCtx.resume();const osc=audioCtx.createOscillator(),gain=audioCtx.createGain(),at=audioCtx.currentTime+delay;osc.type=type;osc.frequency.setValueAtTime(freq,at);gain.gain.setValueAtTime(.045,at);gain.gain.exponentialRampToValueAtTime(.001,at+duration);osc.connect(gain).connect(audioCtx.destination);osc.start(at);osc.stop(at+duration);}catch{soundEnabled=false;}}
function playSound(type){if(type==='move')tone(370,.04,'triangle');if(type==='success'){tone(660,.11);tone(880,.14,'sine',.075);}if(type==='failure'){tone(210,.14,'triangle');tone(155,.18,'triangle',.08);}if(type==='stage'){[523,659,784,1046].forEach((f,i)=>tone(f,.2,'sine',i*.11));}}
function showModal(html,extraClass=''){$('modal-card').className=`modal-card ${extraClass}`;$('modal-card').innerHTML=html;$('overlay').classList.remove('hidden');requestAnimationFrame(()=>$('modal-card').querySelector('input:checked,input:not([disabled]),button:not([disabled])')?.focus({preventScroll:true}));}
function hideModal(){$('overlay').classList.add('hidden');}
function startIntro(){
 paused=false;helpOpen=false;hideModal();$('tasting').classList.add('hidden');
 // Prepare the real first pastry now, so its preview stays identical at arrival.
 game.beginStage();game.phase='intro';expression='neutral';expressionUntil=0;$('scene-feedback').textContent='';transitionTime=PREP_SECONDS;
 drawLettering();$('stage-intro').classList.remove('hidden','compact');
 $('announcer').textContent=`스테이지 ${game.stage}. ${stageDetails().name}. 5초 동안 첫 과자와 정답을 살펴보세요. 과자가 도착하면 60초 동안 25개를 포장하세요.`;
 updateHud(true);playSound('stage');updateDisplayState();
}
async function newGame(){
 if(!loaded){loadAssets();return;}if(!selectedPlayer||game.phase!=='ready'||displayPending)return;
 displayPending=true;document.querySelector('.bakery-app').classList.add('game-started');
 try{await enterGameDisplay();}finally{displayPending=false;lastFrame=0;}
 if(!isGameFullscreen()){document.querySelector('.bakery-app').classList.remove('game-started');showFullscreenGate();return;}
 fullscreenModalOpen=false;fullscreenModalBackup=null;
 cookieTiles=createPastryTiles(globalThis.crypto?.randomUUID?.()??`${Date.now()}-${Math.random()}`,selectedPlayer.id);
 game.reset();sceneWrap.classList.remove('choosing-player');document.querySelector('.game-shell').classList.remove('choosing-player');startIntro();updateDisplayState();
}
function togglePause(){if(!['playing','intro','tasting'].includes(game.phase)||helpOpen||fullscreenBlocked)return;if(paused){paused=false;hideModal();$('pause-button').setAttribute('aria-label','일시정지');return;}paused=true;returnFocus=document.activeElement;$('pause-button').setAttribute('aria-label','계속하기');showModal('<div class="small-stamp">잠깐 쉬어 가요</div><h2 id="modal-title">오븐도 잠깐 휴식!</h2><p>시간과 컨베이어가 멈췄어요.<br>준비되면 이어서 포장해 주세요.</p><button class="primary-button" id="resume-button">계속하기 →</button>');}
function showHelp(){
 if(helpOpen||fullscreenBlocked)return;helpOpen=true;const wasPaused=paused;paused=true;returnFocus=document.activeElement;const previous=$('modal-card').innerHTML;const previousClass=$('modal-card').className;const wasHidden=$('overlay').classList.contains('hidden');
 showModal('<div class="small-stamp">제과점의 작은 안내서</div><h2 id="modal-title">이렇게 포장해요</h2><ol class="help-list"><li>가로 · 전체화면에서 진행해요. 새 게임마다 <b>단계 난이도는 같고 과자 모양은 달라져요.</b></li><li>아빠는 버터 과자, 엄마는 캐러멜·초콜릿, 수안은 젤리·마시멜로, 정안은 엉뚱한 마법 간식을 포장해요.</li><li>스테이지마다 <b>5초 준비 시간</b>이 있어요. 첫 과자가 다가오는 동안 정답을 살펴보세요. 도착하면 60초가 시작돼요.</li><li>초록 타일 위 <b>작업 구역</b>에 들어온 과자를 조작해요.</li><li><b>← →</b>는 90도 회전, <b>↑</b>는 위아래 뒤집기, <b>↓</b>는 좌우 뒤집기예요.</li><li>휴대폰을 <b>가로로</b> 돌리면 왼쪽 위는 위아래 뒤집기, 왼쪽 아래는 좌우 뒤집기예요. 오른쪽 위는 반시계, 아래는 시계 회전 버튼이에요.</li><li><b>상자 속 정답</b>과 모양·장식을 맞추면 자동으로 포장돼요. 제한 시간은 <b>2초</b>!</li><li>초록 O는 성공, 빨간 X는 실패. 1분 동안 <b>30개 중 25개</b> 이상 성공하면 다음 단계로 가요.</li><li>스테이지 실패 시 하트 하나를 잃고 재도전해요. <b>하트 3개, 총 10단계</b>에 도전해 보세요.</li></ol><button class="primary-button" id="close-help">알겠어요!</button>');
 $('close-help').onclick=()=>{helpOpen=false;paused=wasPaused;$('modal-card').innerHTML=previous;$('modal-card').className=previousClass;if(wasHidden)hideModal();refreshSelection();if(returnFocus?.isConnected)returnFocus.focus({preventScroll:true});else $('modal-card').querySelector('input:checked,input:not([disabled]),button:not([disabled])')?.focus({preventScroll:true});};
}
function showRetry(){showModal(`<div class="small-stamp">다시 구우면 더 맛있어질 거예요</div><h2 id="modal-title">한 번 더 해 볼까요?</h2><p>STAGE ${game.stage} · <b>${game.score} / ${game.total}개</b> 포장 성공<br>25개까지 ${25-game.score}개가 모자랐어요.</p><div class="start-rules"><span>남은 기회 <b>${'♥'.repeat(game.lives)}</b></span><span>같은 스테이지에 다시 도전!</span></div><button class="primary-button" id="retry-button">다시 도전하기 →</button>`);}
function showResults(){
 $('tasting').classList.add('hidden');$('stage-intro').classList.add('hidden');
 const allClear=game.phase==='complete',sum=game.history.reduce((s,r)=>s+r.success,0),total=game.history.reduce((s,r)=>s+r.total,0),cleared=game.history.filter(r=>r.passed).length;
 const rows=playerStages().map((s,i)=>{const attempts=game.history.filter(h=>h.stage===i+1);if(!attempts.length)return `<tr><td>${String(i+1).padStart(2,'0')} · ${s.name}</td><td>—</td><td>미도전</td></tr>`;return attempts.map(r=>`<tr><td>${String(i+1).padStart(2,'0')} · ${s.name}${r.attempt>1?` (${r.attempt}차)`:''}</td><td>${r.success} / ${r.total}</td><td class="${r.passed?'result-clear':'result-fail'}">${r.passed?'성공':'실패'}</td></tr>`).join('');}).join('');
 showModal(`<div class="small-stamp">${selectedPlayer.name}의 제과점 영업 기록</div><h2 id="modal-title">${allClear?'최고의 엉뚱한 제과장!':'오늘도 수고했어요!'}</h2><p>${allClear?'10개 스테이지를 모두 완성했어요. 달콤한 대성공!':'하트를 모두 사용했어요. 다음에는 더 잘할 수 있어요.'}</p><div class="results-summary"><span><b>${sum}/${total}</b>개 성공</span><span><b>${cleared}/10</b>단계 완료</span></div><div class="results-table-wrap" tabindex="0" aria-label="스테이지별 전체 기록"><table class="results-table"><thead><tr><th>스테이지</th><th>성공 / 전체</th><th>결과</th></tr></thead><tbody>${rows}</tbody></table></div><button class="primary-button" id="restart-button">가족을 선택하고 다시 굽기 →</button>`,'results-card');
 $('announcer').textContent=`게임 종료. ${total}개 중 ${sum}개 포장 성공, ${cleared}개 스테이지 완료.`;
}
function stageEnded(passed){
 updateHud(true);
 if(passed){expression='happy';expressionUntil=clock+2;drawTasting();$('tasting-subtitle').textContent=`STAGE ${game.stage} CLEAR · ${game.score} / 30`;$('tasting-title').textContent=game.stage===10?'이 맛에 제과장 하지!':'음~ 맛있다!';$('tasting').classList.remove('hidden');transitionTime=1;playSound('stage');}
 else{expression='crying';expressionUntil=clock+10;transitionTime=.8;}
}
function act(action){if(!loaded||paused||rotationRequired||displayPending||fullscreenBlocked||game.phase!=='playing')return;if(game.act(action)){for(const key of document.querySelectorAll(`[data-action="${action}"]`)){key.classList.add('pressed');setTimeout(()=>key.classList.remove('pressed'),140);}}}
function frame(now){
 const dt=lastFrame?Math.min(.25,(now-lastFrame)/1000):0;lastFrame=now;
 if(!paused&&!rotationRequired&&!displayPending&&!fullscreenBlocked&&!document.hidden){clock+=dt;
  if(game.phase==='playing'){game.tick(dt);for(const e of game.drainEvents()){playSound(e.type);if(e.type==='success'||e.type==='failure'){expression=e.type==='success'?'happy':'crying';expressionUntil=clock+1.05;$('scene-feedback').textContent=e.type==='success'?['딱 맞아요!','포장 완료!','잘했어요!'][e.cookie.id%3]:'앗, 아까워요!';$('scene-feedback').classList.toggle('bad',e.type==='failure');}if(e.type==='stageEnd')stageEnded(e.passed);}}
  else if(game.phase==='intro'){transitionTime=Math.max(0,transitionTime-dt);if(transitionTime===0){$('stage-intro').classList.add('hidden');game.phase='playing';game.elapsed=0;$('announcer').textContent='첫 과자 도착! 지금부터 60초 동안 포장해요.';updateHud(true);}}
  else if(game.phase==='tasting'){transitionTime-=dt;if(transitionTime<=0){$('tasting').classList.add('hidden');game.advance();if(game.phase==='complete')showResults();else startIntro();}}
  else if((game.phase==='retry'||game.phase==='gameover')&&transitionTime>0){transitionTime-=dt;if(transitionTime<=0){if(game.phase==='retry')showRetry();else showResults();}}
  if(clock>expressionUntil)$('scene-feedback').textContent='';
 }
 drawScene();drawPreviews();updateHud();requestAnimationFrame(frame);
}

$('overlay').addEventListener('change',e=>{if(e.target.matches('input[name="player"]'))selectPlayer(e.target.value);});
$('overlay').addEventListener('click',e=>{const id=e.target.closest('button')?.id;if(id==='enter-fullscreen-button'){if(game.phase==='ready')newGame();else restoreFullscreen();return;}if(id==='choose-family-button'){showCharacterSelection();return;}if(id==='start-button')newGame();if(id==='restart-button')showCharacterSelection();if(id==='retry-button')startIntro();if(id==='resume-button'){togglePause();returnFocus?.focus({preventScroll:true});}});
$('pause-button').addEventListener('click',togglePause);$('help-button').addEventListener('click',showHelp);
$('sound-button').addEventListener('click',()=>{soundEnabled=!soundEnabled;$('sound-button').classList.toggle('sound-on',soundEnabled);$('sound-button').setAttribute('aria-label',soundEnabled?'소리 끄기':'소리 켜기');$('sound-button').title=soundEnabled?'소리 끄기':'소리 켜기';tone(660,.09);});
for(const key of document.querySelectorAll('[data-action]')){
 key.addEventListener('pointerdown',e=>{if(e.pointerType==='mouse'&&e.button!==0)return;e.preventDefault();key.setPointerCapture?.(e.pointerId);act(key.dataset.action);});
 for(const event of ['pointerup','pointercancel','lostpointercapture'])key.addEventListener(event,()=>key.classList.remove('pressed'));
 key.addEventListener('click',e=>{if(e.detail===0)act(key.dataset.action);});key.addEventListener('contextmenu',e=>e.preventDefault());
}
document.addEventListener('keydown',e=>{
 const mapping={ArrowLeft:'left',ArrowRight:'right',ArrowUp:'up',ArrowDown:'down'};
 if(mapping[e.key]&&game.phase==='playing'&&!paused&&!rotationRequired&&!displayPending&&!fullscreenBlocked){e.preventDefault();if(!e.repeat)act(mapping[e.key]);}
 if((e.code==='KeyP'||e.code==='Escape')&&!e.repeat){if(!fullscreenBlocked){if(helpOpen){$('close-help')?.click();}else togglePause();}}
 if(e.code==='Enter'&&game.phase==='ready'&&!helpOpen&&e.target.tagName!=='BUTTON'){e.preventDefault();newGame();}
 if(e.key==='Tab'&&!$('overlay').classList.contains('hidden')){const candidates=[...$('modal-card').querySelectorAll('button:not([disabled]),input:not([disabled]),[tabindex="0"]')];const radio=candidates.find(el=>el.type==='radio'&&el.checked)||candidates.find(el=>el.type==='radio');const focusables=candidates.filter(el=>el.type!=='radio'||el===radio);const first=focusables[0],last=focusables.at(-1);const active=document.activeElement?.type==='radio'?radio:document.activeElement;if(e.shiftKey&&(active===first||!$('modal-card').contains(document.activeElement))){e.preventDefault();last?.focus();}else if(!e.shiftKey&&(active===last||!$('modal-card').contains(document.activeElement))){e.preventDefault();first?.focus();}}
});
document.addEventListener('visibilitychange',()=>{lastFrame=0;if(document.hidden&&!paused&&['playing','intro','tasting'].includes(game.phase))togglePause();});
window.addEventListener('blur',()=>{if(!displayPending&&!paused&&!helpOpen&&['playing','intro','tasting'].includes(game.phase))togglePause();});
function showFullscreenGate(){
 if(fullscreenModalOpen)return;
 fullscreenModalBackup={html:$('modal-card').innerHTML,className:$('modal-card').className,hidden:$('overlay').classList.contains('hidden')};fullscreenModalOpen=true;
 const supported=Boolean(document.documentElement.requestFullscreen||document.documentElement.webkitRequestFullscreen);
 showModal(`<div class="small-stamp">가로 · 전체화면 전용</div><h2 id="modal-title">전체화면으로 시작해요</h2><p>${supported?'캐릭터와 작업 구역이 모두 보이도록<br>전체화면에서 게임을 진행해요.':'이 브라우저는 게임 전체화면을 지원하지 않아요.<br>전체화면을 지원하는 브라우저로 열어 주세요.'}</p>${supported?'<button class="primary-button" id="enter-fullscreen-button">전체화면으로 시작하기 →</button>':''}<button class="secondary-button" id="choose-family-button">가족 다시 선택하기</button>`);
}
async function restoreFullscreen(){
 if(displayPending)return;displayPending=true;
 try{await enterGameDisplay();}finally{displayPending=false;updateDisplayState();}
}
function updateDisplayState(){
 const fullscreen=isGameFullscreen();const button=$('fullscreen-button');
 button.setAttribute('aria-pressed',String(fullscreen));button.setAttribute('aria-label',fullscreen?'전체화면 나가기':'전체화면 보기');button.title=fullscreen?'전체화면 나가기':'전체화면 보기';
 fullscreenBlocked=game.phase!=='ready'&&!fullscreen;
 if(fullscreenBlocked){if(helpOpen)$('close-help')?.click();showFullscreenGate();}
 else if(fullscreen&&fullscreenModalOpen){const previous=fullscreenModalBackup;fullscreenModalOpen=false;fullscreenModalBackup=null;if(previous){$('modal-card').innerHTML=previous.html;$('modal-card').className=previous.className;if(previous.hidden)hideModal();else refreshSelection();}}
 rotationRequired=matchMedia('(orientation: portrait)').matches&&['intro','playing','tasting'].includes(game.phase);
 $('rotate-prompt').classList.toggle('hidden',!rotationRequired);lastFrame=0;
}
$('fullscreen-button').addEventListener('click',async()=>{
 if(displayPending)return;displayPending=true;
 try{if(isGameFullscreen())await exitGameDisplay();else await enterGameDisplay();}finally{displayPending=false;updateDisplayState();}
});
for(const event of ['fullscreenchange','webkitfullscreenchange'])document.addEventListener(event,updateDisplayState);
window.addEventListener('resize',updateDisplayState);
window.visualViewport?.addEventListener('resize',updateDisplayState);
updateDisplayState();
loadAssets();requestAnimationFrame(frame);
