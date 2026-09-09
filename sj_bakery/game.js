import {BakeryGame,STAGES,QUESTION_SECONDS,QUESTIONS_PER_STAGE,PASS_SCORE} from './engine.js?v=box-puzzle-9';
import {isGameFullscreen,enterGameDisplay,exitGameDisplay} from './display.js?v=family-snacks-4';
import {createPastryTiles,SNACK_MENUS} from './pastries.js?v=odd-designs-8';
import {PLAYERS,POSES,getPlayer,EATING_MOUTHS,SPRITE_RECTS,keySpriteMatte} from './family.js';

const $=id=>document.getElementById(id);
const game=new BakeryGame();
const PREP_SECONDS=5;
const canvas=$('game-canvas'),ctx=canvas.getContext('2d');
const sceneWrap=document.querySelector('.scene-wrap');
const answerButtons=[...document.querySelectorAll('[data-answer]')];
const answerCanvases=answerButtons.map(button=>button.querySelector('canvas'));
let displayPending=false,rotationRequired=false,fullscreenBlocked=false,fullscreenModalOpen=false,fullscreenModalBackup=null;
const art={},crop=[[140,32,302,459],[613,31,307,460],[1112,33,304,458],[139,538,248,462],[608,491,363,509],[1079,492,424,508]];
let loaded=false,paused=false,helpOpen=false,transitionTime=0,clock=0,lastFrame=0,lastHud=-1,expression='neutral',expressionUntil=0,soundEnabled=false,audioCtx=null,returnFocus=null;
let cookieTiles=[],letteringStage=0,renderedQuestion=null;
let selectedPlayer=null,assetsLoading=false,assetError=false,familyFrames=[];
const playerStages=()=>SNACK_MENUS[selectedPlayer?.id]?.stages??STAGES;
const stageDetails=()=>playerStages()[game.stage-1];
const selectionHTML=$('modal-card').innerHTML;
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)').matches;
const ACTION_LABELS={
 left:{symbol:'⟲',name:'반시계'},
 right:{symbol:'⟳',name:'시계'},
 up:{symbol:'↕',name:'위아래'},
 down:{symbol:'↔',name:'좌우'}
};

function loadImage(name,path){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>{art[name]=img;resolve();};img.onerror=()=>reject(new Error(path));img.src=path;});}
async function loadAssets(){
 if(assetsLoading)return;assetsLoading=true;assetError=false;refreshSelection();$('help-button').disabled=true;
 try{
  await Promise.all([loadImage('background','./assets/bakery-background.webp'),loadImage('characters','./assets/bakery-characters.webp'),loadImage('family','./assets/family-characters.png'),loadImage('jeonganHappy','./assets/jeongan-happy.webp').catch(()=>{})]);
  cookieTiles=createPastryTiles('bakery-preview',selectedPlayer?.id??'dad');prepareFamilyFrames();loaded=true;assetsLoading=false;$('help-button').disabled=false;refreshSelection();drawLettering();
 }catch{loaded=false;assetsLoading=false;assetError=true;refreshSelection();}
}
function prepareFamilyFrames(){
 familyFrames=SPRITE_RECTS.map(([sx,sy,right,bottom])=>{const cell=document.createElement('canvas');cell.width=right-sx;cell.height=bottom-sy;const c=cell.getContext('2d',{willReadFrequently:true});c.drawImage(art.family,sx,sy,cell.width,cell.height,0,0,cell.width,cell.height);const pixels=c.getImageData(0,0,cell.width,cell.height);keySpriteMatte(pixels.data,cell.width,cell.height);c.putImageData(pixels,0,0);return cell;});
 const jeongan=getPlayer('jeongan'),happyIndex=POSES.happy*PLAYERS.length+jeongan.column;familyFrames[happyIndex]=familyFrames[jeongan.column];
 if(art.jeonganHappy){const cell=document.createElement('canvas');cell.width=863;cell.height=1206;const c=cell.getContext('2d',{willReadFrequently:true});c.drawImage(art.jeonganHappy,198,18,863,1206,0,0,863,1206);const pixels=c.getImageData(0,0,cell.width,cell.height);keySpriteMatte(pixels.data,cell.width,cell.height);c.putImageData(pixels,0,0);familyFrames[happyIndex]=cell;}
}
function refreshSelection(){
 const button=$('start-button'),note=$('load-note');if(!button)return;
 button.disabled=assetsLoading||(!assetError&&(!loaded||!selectedPlayer));
 button.textContent=assetError?'그림 다시 불러오기':assetsLoading?'가족을 불러오는 중…':selectedPlayer?`${selectedPlayer.as} 시작하기 →`:'이름을 선택해 주세요';
 if(note)note.textContent=assetError?'그림을 불러오지 못했어요. 다시 눌러 주세요.':assetsLoading?'함께할 가족을 불러오고 있어요…':'시작하면 가로 · 전체화면으로 열려요 · 20개 중 15개 성공!';
 for(const input of document.querySelectorAll('input[name="player"]')){input.disabled=!loaded;input.checked=input.value===selectedPlayer?.id;}
 for(const portrait of document.querySelectorAll('[data-portrait]')){const c=portrait.getContext('2d');c.clearRect(0,0,160,160);const player=getPlayer(portrait.dataset.portrait);if(!loaded||!player)continue;const sprite=familyFrames[player.column],w=sprite.width,h=sprite.height,headHeight=h*.47,size=Math.max(w,headHeight);c.drawImage(sprite,-(size-w)/2,0,size,size,0,0,160,160);}
 $('player-caption').textContent=selectedPlayer?`${selectedPlayer.name}의 달콤한 도전`:'조금 삐뚤어도, 맛있을 거야!';
}
function selectPlayer(id){const player=getPlayer(id);if(!loaded||game.phase!=='ready'||!player||helpOpen)return;selectedPlayer=player;cookieTiles=createPastryTiles('bakery-preview',player.id);refreshSelection();drawLettering();updateHud(true);$('announcer').textContent=`${player.name} 선택. 포장할 간식: ${SNACK_MENUS[player.id]?.label??'버터 과자'}. 시작 버튼을 눌러 주세요.`;}
function showCharacterSelection(){
 document.querySelector('.bakery-app').classList.remove('game-started');fullscreenModalOpen=false;fullscreenModalBackup=null;game.reset();selectedPlayer=null;paused=false;helpOpen=false;expression='neutral';expressionUntil=0;transitionTime=0;renderedQuestion=null;$('scene-feedback').textContent='';
 if(loaded)cookieTiles=createPastryTiles('bakery-preview','dad');$('stage-intro').classList.add('hidden');$('tasting').classList.add('hidden');sceneWrap.classList.add('choosing-player');document.querySelector('.game-shell').classList.add('choosing-player');showModal(selectionHTML,'character-select');refreshSelection();updateHud(true);updateDisplayState();
}
function familyCharacter(c,pose,cx,bottom,height){if(!loaded||!selectedPlayer)return null;const sprite=familyFrames[pose*4+selectedPlayer.column],w=height*sprite.width/sprite.height,x=cx-w/2,y=bottom-height;c.drawImage(sprite,x,y,w,height);return {x,y,w,h:height};}
function rounded(c,x,y,w,h,r,fill,stroke){c.beginPath();c.roundRect(x,y,w,h,r);if(fill){c.fillStyle=fill;c.fill();}if(stroke){c.strokeStyle=stroke;c.lineWidth=2;c.stroke();}}
function character(c,index,cx,bottom,height){if(!art.characters)return;const [sx,sy,sw,sh]=crop[index],w=height*sw/sh;c.drawImage(art.characters,sx,sy,sw,sh,cx-w/2,bottom-height,w,height);}
function cookie(c,stage,x,y,size,o={rotation:0,flipped:false},alpha=1){if(!cookieTiles.length)return;c.save();c.translate(x,y);c.rotate(o.rotation*Math.PI/2);c.scale(1,o.flipped?-1:1);c.globalAlpha=alpha;c.drawImage(cookieTiles[stage-1],-size/2,-size/2,size,size);c.restore();}

function drawScene(){
 ctx.clearRect(0,0,1200,600);
 if(art.background)ctx.drawImage(art.background,0,0,art.background.width,art.background.height,0,-55,1200,675);else{ctx.fillStyle='#e8c49e';ctx.fillRect(0,0,1200,600);}
 if(!loaded)return;
 const pose=expressionUntil>clock?(POSES[expression]??POSES.neutral):POSES.neutral,bob=reducedMotion||paused?0:Math.sin(clock*2.5)*2;
 character(ctx,4,174,430,320);familyCharacter(ctx,pose,1070,440+bob,selectedPlayer?.height??253);
 if(selectedPlayer){const nameY=440-selectedPlayer.height-31;rounded(ctx,1024,nameY,92,27,13,'#fff7e4e8','#b98c61');ctx.font='600 17px sans-serif';ctx.textAlign='center';ctx.fillStyle='#87513b';ctx.fillText(selectedPlayer.name,1070,nameY+19);}
 rounded(ctx,54,146,292,188,18,'#fff0d5e8','#a97048');rounded(ctx,70,160,260,145,12,'#e6c59a','#a97048');
 ctx.fillStyle='#593b2c26';ctx.beginPath();ctx.ellipse(200,271,64,17,0,0,Math.PI*2);ctx.fill();
 const question=game.question;
 if(question&&game.phase!=='intro')cookie(ctx,game.stage,200,249,150,question.initial);
 ctx.font='700 17px sans-serif';ctx.fillStyle='#704633';ctx.textAlign='center';ctx.fillText(game.phase==='intro'?'첫 문제를 준비하는 중…':'처음 놓인 방향',200,358);
 if(!reducedMotion){ctx.save();for(let i=0;i<3;i++){const p=(clock*.4+i/3)%1;ctx.globalAlpha=(1-p)*.42;ctx.strokeStyle='#fff8e8';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(172+i*25,202-p*40);ctx.quadraticCurveTo(157+i*25,186-p*40,178+i*25,170-p*40);ctx.stroke();}ctx.restore();}
}
function clearAnswerState(){
 const row=$('answer-row');row.classList.remove('answered','success','failure','timeout');
 answerButtons.forEach(button=>{button.classList.remove('selected','correct','wrong');button.disabled=false;});
}
function drawQuestion(){
 const question=game.question;
 if(!question||game.phase==='intro'){clearAnswerState();$('instruction-steps').innerHTML='<span class="instruction-wait">시작하면 명령이 나타나요</span>';answerCanvases.forEach(c=>c.getContext('2d').clearRect(0,0,180,180));answerButtons.forEach(button=>button.disabled=true);return;}
 $('instruction-steps').innerHTML=question.instructions.map((action,index)=>{const label=ACTION_LABELS[action];return `<span class="instruction-step"><b>${index+1}</b><i>${label.symbol}</i><small>${label.name}</small></span>`;}).join('<em>›</em>');
 clearAnswerState();
 question.candidates.forEach((orientation,index)=>{const c=answerCanvases[index].getContext('2d');c.clearRect(0,0,180,180);cookie(c,game.stage,90,90,160,orientation);answerButtons[index].setAttribute('aria-label',`${index+1}번 상자, 과자 방향 후보`);});
}
function showAnswerResult(question){
 const row=$('answer-row');row.classList.add('answered',question.result);if(question.result==='failure'&&question.selectedIndex===null)row.classList.add('timeout');
 answerButtons.forEach((button,index)=>{button.disabled=true;if(question.result==='success'&&index===question.correctIndex)button.classList.add('selected','correct');if(question.result==='failure'&&index===question.selectedIndex)button.classList.add('selected','wrong');});
}
function drawLettering(){
 const c=$('stage-lettering').getContext('2d');c.clearRect(0,0,1000,190);const text=`STAGE ${game.stage}`;
 c.font='900 145px Georgia,serif';c.textAlign='center';c.textBaseline='middle';c.lineJoin='round';c.strokeStyle='#8c542e';c.lineWidth=15;c.strokeText(text,500,102);c.strokeStyle='#edbd73';c.lineWidth=7;c.strokeText(text,500,97);
 const g=c.createLinearGradient(0,20,0,170);g.addColorStop(0,'#f8d893');g.addColorStop(.55,stageDetails().color);g.addColorStop(1,'#b77939');c.fillStyle=g;c.fillText(text,500,97);
 if(cookieTiles.length){const pattern=c.createPattern(cookieTiles[game.stage-1],'repeat');c.globalAlpha=.62;c.fillStyle=pattern;c.fillText(text,500,97);c.globalAlpha=1;}
 $('stage-lettering').setAttribute('aria-label',`STAGE ${game.stage}`);$('intro-flavor').textContent=`${stageDetails().name} · 명령 ${game.stage}번`;letteringStage=game.stage;
}
function drawTasting(){const c=$('tasting-canvas').getContext('2d');c.clearRect(0,0,460,440);const frame=familyCharacter(c,POSES.eating,230,680,690);if(!frame)return;const mouth=EATING_MOUTHS[selectedPlayer.column];cookie(c,game.stage,frame.x+frame.w*mouth.x,frame.y+frame.h*mouth.y,110,{rotation:0,flipped:false});}
function updateHud(force=false){
 if(!force&&clock-lastHud<.05)return;lastHud=clock;$('stage-number').textContent=String(game.stage).padStart(2,'0');$('stage-flavor').textContent=stageDetails().name;
 $('question-number').textContent=String(Math.min(QUESTIONS_PER_STAGE,game.total+1));$('score').textContent=game.score;$('score-fill').style.width=`${Math.min(100,game.score/PASS_SCORE*100)}%`;$('score-progress').setAttribute('aria-valuenow',Math.min(PASS_SCORE,game.score));
 [...$('hearts').children].forEach((h,i)=>h.classList.toggle('lost',i>=game.lives));$('hearts').setAttribute('aria-label',`남은 기회 ${game.lives}번`);
 const preparing=game.phase==='intro',remaining=preparing?transitionTime:game.active?Math.max(0,QUESTION_SECONDS-game.questionElapsed):0;
 $('window-time').textContent=preparing?String(Math.ceil(remaining)):remaining.toFixed(1);$('window-fill').style.width=`${remaining/(preparing?PREP_SECONDS:QUESTION_SECONDS)*100}%`;$('window-fill').style.background=preparing?'#4b8461':remaining<.8?'#b54f40':'#be9154';$('window-label').textContent=preparing?'시작까지':'고를 시간';
 if(preparing){$('prep-countdown').textContent=Math.ceil(remaining);$('stage-intro').classList.toggle('compact',remaining<=PREP_SECONDS-1);}
 $('status-tip').innerHTML=preparing?`STAGE ${game.stage} · 명령 ${game.stage}번`:game.score>=PASS_SCORE?'목표 달성! 끝까지 풀어봐요.':'명령을 따라간 뒤<br>알맞은 상자를 고르세요!';
 $('pause-button').disabled=!['playing','intro','tasting'].includes(game.phase);
 if(renderedQuestion!==game.question){renderedQuestion=game.question;drawQuestion();}
}
function tone(freq,duration=.08,type='sine',delay=0){if(!soundEnabled)return;try{audioCtx??=new(window.AudioContext||window.webkitAudioContext)();audioCtx.resume();const osc=audioCtx.createOscillator(),gain=audioCtx.createGain(),at=audioCtx.currentTime+delay;osc.type=type;osc.frequency.setValueAtTime(freq,at);gain.gain.setValueAtTime(.045,at);gain.gain.exponentialRampToValueAtTime(.001,at+duration);osc.connect(gain).connect(audioCtx.destination);osc.start(at);osc.stop(at+duration);}catch{soundEnabled=false;}}
function playSound(type){if(type==='success'){tone(660,.11);tone(880,.14,'sine',.075);}if(type==='failure'){tone(210,.14,'triangle');tone(155,.18,'triangle',.08);}if(type==='stage')[523,659,784,1046].forEach((f,i)=>tone(f,.2,'sine',i*.11));}
function showModal(html,extraClass=''){$('modal-card').className=`modal-card ${extraClass}`;$('modal-card').innerHTML=html;$('overlay').classList.remove('hidden');requestAnimationFrame(()=>$('modal-card').querySelector('input:checked,input:not([disabled]),button:not([disabled])')?.focus({preventScroll:true}));}
function hideModal(){$('overlay').classList.add('hidden');}
function startIntro(){
 paused=false;helpOpen=false;hideModal();$('tasting').classList.add('hidden');game.beginStage();game.phase='intro';expression='neutral';expressionUntil=0;transitionTime=PREP_SECONDS;renderedQuestion=null;$('scene-feedback').textContent='';drawLettering();drawQuestion();$('stage-intro').classList.remove('hidden','compact');
 $('announcer').textContent=`스테이지 ${game.stage}. ${stageDetails().name}. 각 문제에는 명령 ${game.stage}개가 나오며, 3초 안에 세 상자 중 하나를 고릅니다.`;updateHud(true);playSound('stage');updateDisplayState();
}
async function newGame(){
 if(!loaded){loadAssets();return;}if(!selectedPlayer||game.phase!=='ready'||displayPending)return;displayPending=true;document.querySelector('.bakery-app').classList.add('game-started');
 try{await enterGameDisplay();}finally{displayPending=false;lastFrame=0;}
 if(!isGameFullscreen()){document.querySelector('.bakery-app').classList.remove('game-started');showFullscreenGate();return;}
 fullscreenModalOpen=false;fullscreenModalBackup=null;cookieTiles=createPastryTiles(globalThis.crypto?.randomUUID?.()??`${Date.now()}-${Math.random()}`,selectedPlayer.id);game.reset();sceneWrap.classList.remove('choosing-player');document.querySelector('.game-shell').classList.remove('choosing-player');startIntro();updateDisplayState();
}
function togglePause(){if(!['playing','intro','tasting'].includes(game.phase)||helpOpen||fullscreenBlocked)return;if(paused){paused=false;hideModal();$('pause-button').setAttribute('aria-label','일시정지');return;}paused=true;returnFocus=document.activeElement;$('pause-button').setAttribute('aria-label','계속하기');showModal('<div class="small-stamp">잠깐 쉬어 가요</div><h2 id="modal-title">오븐도 잠깐 휴식!</h2><p>문제 시간도 멈췄어요.<br>준비되면 이어서 풀어 주세요.</p><button class="primary-button" id="resume-button">계속하기 →</button>');}
function showHelp(){
 if(helpOpen||fullscreenBlocked)return;helpOpen=true;const wasPaused=paused;paused=true;returnFocus=document.activeElement;const previous=$('modal-card').innerHTML,previousClass=$('modal-card').className,wasHidden=$('overlay').classList.contains('hidden');
 showModal('<div class="small-stamp">제과점의 작은 안내서</div><h2 id="modal-title">이렇게 포장해요</h2><ol class="help-list"><li>주방장 아저씨가 과자를 비뚤게 놓아요.</li><li>과자 위에 나오는 <b>회전·뒤집기 명령</b>을 왼쪽부터 따라가세요.</li><li>명령을 모두 적용한 모양을 생각하고 <b>세 상자 중 정답</b>을 누르세요. PC에서는 1·2·3 키도 쓸 수 있어요.</li><li>스테이지 번호만큼 명령이 나와요. 3단계는 3번, 10단계는 10번이에요.</li><li>각 문제의 제한 시간은 <b>3초</b>예요.</li><li>맞으면 O와 함께 상자가 천장으로 날아가고, 틀리면 X와 함께 바닥으로 떨어져요.</li><li>스테이지마다 <b>20문제 중 15문제</b>를 맞히면 통과해요.</li><li>하트는 3개, 스테이지는 모두 10개예요.</li></ol><button class="primary-button" id="close-help">알겠어요!</button>');
 $('close-help').onclick=()=>{helpOpen=false;paused=wasPaused;$('modal-card').innerHTML=previous;$('modal-card').className=previousClass;if(wasHidden)hideModal();refreshSelection();returnFocus?.focus?.({preventScroll:true});};
}
function showRetry(){showModal(`<div class="small-stamp">다시 생각하면 풀 수 있어요</div><h2 id="modal-title">한 번 더 해 볼까요?</h2><p>STAGE ${game.stage} · <b>${game.score} / ${game.total}개</b> 정답<br>15개까지 ${15-game.score}개가 모자랐어요.</p><div class="start-rules"><span>남은 기회 <b>${'♥'.repeat(game.lives)}</b></span><span>같은 난이도로 다시 도전!</span></div><button class="primary-button" id="retry-button">다시 도전하기 →</button>`);}
function showResults(){
 $('tasting').classList.add('hidden');$('stage-intro').classList.add('hidden');const allClear=game.phase==='complete',sum=game.history.reduce((s,r)=>s+r.success,0),total=game.history.reduce((s,r)=>s+r.total,0),cleared=game.history.filter(r=>r.passed).length;
 const rows=playerStages().map((s,i)=>{const attempts=game.history.filter(h=>h.stage===i+1);if(!attempts.length)return `<tr><td>${String(i+1).padStart(2,'0')} · ${s.name}</td><td>—</td><td>미도전</td></tr>`;return attempts.map(r=>`<tr><td>${String(i+1).padStart(2,'0')} · ${s.name}${r.attempt>1?` (${r.attempt}차)`:''}</td><td>${r.success} / ${r.total}</td><td class="${r.passed?'result-clear':'result-fail'}">${r.passed?'성공':'실패'}</td></tr>`).join('');}).join('');
 showModal(`<div class="small-stamp">${selectedPlayer.name}의 제과점 영업 기록</div><h2 id="modal-title">${allClear?'최고의 엉뚱한 제과장!':'오늘도 수고했어요!'}</h2><p>${allClear?'10개 스테이지를 모두 완성했어요. 달콤한 대성공!':'하트를 모두 사용했어요. 다음에는 더 잘할 수 있어요.'}</p><div class="results-summary"><span><b>${sum}/${total}</b>개 정답</span><span><b>${cleared}/10</b>단계 완료</span></div><div class="results-table-wrap" tabindex="0" aria-label="스테이지별 전체 기록"><table class="results-table"><thead><tr><th>스테이지</th><th>정답 / 전체</th><th>결과</th></tr></thead><tbody>${rows}</tbody></table></div><button class="primary-button" id="restart-button">가족을 선택하고 다시 풀기 →</button>`,'results-card');
}
function stageEnded(passed){updateHud(true);if(passed){expression='happy';expressionUntil=clock+2;drawTasting();$('tasting-subtitle').textContent=`STAGE ${game.stage} CLEAR · ${game.score} / 20`;$('tasting-title').textContent=game.stage===10?'이 맛에 제과장 하지!':'음~ 맛있다!';$('tasting').classList.remove('hidden');transitionTime=1;playSound('stage');}else{expression='crying';expressionUntil=clock+10;transitionTime=.8;}}
function chooseBox(index){if(!loaded||paused||rotationRequired||displayPending||fullscreenBlocked||game.phase!=='playing')return;if(game.choose(index))showAnswerResult(game.question);}
function frame(now){
 const dt=lastFrame?Math.min(.25,(now-lastFrame)/1000):0;lastFrame=now;
 if(!paused&&!rotationRequired&&!displayPending&&!fullscreenBlocked&&!document.hidden){clock+=dt;
  if(game.phase==='playing'){game.tick(dt);for(const e of game.drainEvents()){if(e.type==='question'){renderedQuestion=null;$('scene-feedback').textContent='';}if(e.type==='success'||e.type==='failure'){playSound(e.type);showAnswerResult(e.question);expression=e.type==='success'?'happy':'crying';expressionUntil=clock+1.05;$('scene-feedback').textContent=e.type==='success'?'O  정답!':'X  아까워요!';$('scene-feedback').classList.toggle('bad',e.type==='failure');}if(e.type==='stageEnd')stageEnded(e.passed);}}
  else if(game.phase==='intro'){transitionTime=Math.max(0,transitionTime-dt);if(transitionTime===0){$('stage-intro').classList.add('hidden');game.phase='playing';game.questionElapsed=0;renderedQuestion=null;$('announcer').textContent='첫 문제 시작! 3초 안에 알맞은 상자를 고르세요.';}}
  else if(game.phase==='tasting'){transitionTime-=dt;if(transitionTime<=0){$('tasting').classList.add('hidden');game.advance();if(game.phase==='complete')showResults();else startIntro();}}
  else if((game.phase==='retry'||game.phase==='gameover')&&transitionTime>0){transitionTime-=dt;if(transitionTime<=0){if(game.phase==='retry')showRetry();else showResults();}}
  if(clock>expressionUntil)$('scene-feedback').textContent='';
 }
 drawScene();updateHud();requestAnimationFrame(frame);
}

$('overlay').addEventListener('change',e=>{if(e.target.matches('input[name="player"]'))selectPlayer(e.target.value);});
$('overlay').addEventListener('click',e=>{const id=e.target.closest('button')?.id;if(id==='enter-fullscreen-button'){if(game.phase==='ready')newGame();else restoreFullscreen();return;}if(id==='choose-family-button'){showCharacterSelection();return;}if(id==='start-button')newGame();if(id==='restart-button')showCharacterSelection();if(id==='retry-button')startIntro();if(id==='resume-button'){togglePause();returnFocus?.focus({preventScroll:true});}});
answerButtons.forEach(button=>button.addEventListener('click',()=>chooseBox(Number(button.dataset.answer))));
$('pause-button').addEventListener('click',togglePause);$('help-button').addEventListener('click',showHelp);
$('sound-button').addEventListener('click',()=>{soundEnabled=!soundEnabled;$('sound-button').classList.toggle('sound-on',soundEnabled);$('sound-button').setAttribute('aria-label',soundEnabled?'소리 끄기':'소리 켜기');tone(660,.09);});
document.addEventListener('keydown',e=>{
 const index={'Digit1':0,'Numpad1':0,'Digit2':1,'Numpad2':1,'Digit3':2,'Numpad3':2}[e.code];if(index!==undefined&&!e.repeat){e.preventDefault();chooseBox(index);}
 if((e.code==='KeyP'||e.code==='Escape')&&!e.repeat&&!fullscreenBlocked){if(helpOpen)$('close-help')?.click();else togglePause();}
 if(e.code==='Enter'&&game.phase==='ready'&&!helpOpen&&e.target.tagName!=='BUTTON'){e.preventDefault();newGame();}
});
document.addEventListener('visibilitychange',()=>{lastFrame=0;if(document.hidden&&!paused&&['playing','intro','tasting'].includes(game.phase))togglePause();});
window.addEventListener('blur',()=>{if(!displayPending&&!paused&&!helpOpen&&['playing','intro','tasting'].includes(game.phase))togglePause();});
function showFullscreenGate(){
 if(fullscreenModalOpen)return;fullscreenModalBackup={html:$('modal-card').innerHTML,className:$('modal-card').className,hidden:$('overlay').classList.contains('hidden')};fullscreenModalOpen=true;
 const supported=Boolean(document.documentElement.requestFullscreen||document.documentElement.webkitRequestFullscreen);showModal(`<div class="small-stamp">가로 · 전체화면 전용</div><h2 id="modal-title">전체화면으로 시작해요</h2><p>${supported?'세 상자와 명령이 모두 보이도록<br>전체화면에서 게임을 진행해요.':'이 브라우저는 게임 전체화면을 지원하지 않아요.<br>전체화면을 지원하는 브라우저로 열어 주세요.'}</p>${supported?'<button class="primary-button" id="enter-fullscreen-button">전체화면으로 시작하기 →</button>':''}<button class="secondary-button" id="choose-family-button">가족 다시 선택하기</button>`);}
async function restoreFullscreen(){if(displayPending)return;displayPending=true;try{await enterGameDisplay();}finally{displayPending=false;updateDisplayState();}}
function updateDisplayState(){
 const fullscreen=isGameFullscreen(),button=$('fullscreen-button');button.setAttribute('aria-pressed',String(fullscreen));button.setAttribute('aria-label',fullscreen?'전체화면 나가기':'전체화면 보기');fullscreenBlocked=game.phase!=='ready'&&!fullscreen;
 if(fullscreenBlocked){if(helpOpen)$('close-help')?.click();showFullscreenGate();}else if(fullscreen&&fullscreenModalOpen){const previous=fullscreenModalBackup;fullscreenModalOpen=false;fullscreenModalBackup=null;if(previous){$('modal-card').innerHTML=previous.html;$('modal-card').className=previous.className;if(previous.hidden)hideModal();else refreshSelection();}}
 rotationRequired=matchMedia('(orientation: portrait)').matches&&['intro','playing','tasting'].includes(game.phase);$('rotate-prompt').classList.toggle('hidden',!rotationRequired);lastFrame=0;
}
$('fullscreen-button').addEventListener('click',async()=>{if(displayPending)return;displayPending=true;try{if(isGameFullscreen())await exitGameDisplay();else await enterGameDisplay();}finally{displayPending=false;updateDisplayState();}});
for(const event of ['fullscreenchange','webkitfullscreenchange'])document.addEventListener(event,updateDisplayState);window.addEventListener('resize',updateDisplayState);window.visualViewport?.addEventListener('resize',updateDisplayState);
updateDisplayState();loadAssets();requestAnimationFrame(frame);
