export const STAGE_SECONDS=60;
export const COOKIE_SECONDS=2;
export const COOKIES_PER_STAGE=30;
export const PASS_SCORE=25;
export const STAGES=[
 {name:'버터의 첫걸음',color:'#dba04a'}, {name:'초코칩이 콕콕',color:'#cd8f3f'},
 {name:'초콜릿 반쪽의 비밀',color:'#8b5133'}, {name:'지그재그 초코 산책',color:'#875232'},
 {name:'딸기의 등장',color:'#cb6352'}, {name:'딸기와 크림의 춤',color:'#d87f69'},
 {name:'크림으로 그린 낙서',color:'#c3a074'}, {name:'달콤한 삼각관계',color:'#b96b4a'},
 {name:'빙글빙글 크림 미로',color:'#ba885d'}, {name:'엉뚱한 제과장의 걸작',color:'#b44740'}
];
export const mod4=n=>((n%4)+4)%4;
export function transform(o,action){
 if(action==='left')return {rotation:mod4(o.rotation-1),flipped:o.flipped};
 if(action==='right')return {rotation:mod4(o.rotation+1),flipped:o.flipped};
 if(action==='up'||action==='down')return {rotation:mod4(-o.rotation),flipped:!o.flipped};
 return {...o};
}
export const isCorrect=o=>o.rotation===0&&!o.flipped;
export class BakeryGame{
 constructor(random=Math.random){this.random=random;this.reset();}
 reset(){this.stage=1;this.lives=3;this.history=[];this.phase='ready';this.elapsed=0;this.cookies=[];this.score=0;this.total=0;this.attempt=0;this.events=[];}
 beginStage(){this.attempt++;this.elapsed=0;this.score=0;this.total=0;this.cookies=Array.from({length:COOKIES_PER_STAGE},(_,i)=>{let rotation=Math.floor(this.random()*4);let flipped=this.random()<(.25+(this.stage-1)*.045);if(rotation===0&&!flipped)rotation=1;return {id:i,arrival:i*COOKIE_SECONDS,deadline:(i+1)*COOKIE_SECONDS,rotation,flipped,result:null,resolvedAt:null,matchSince:null,animation:null};});this.phase='playing';this.events=[];}
 get active(){if(this.phase!=='playing')return null;const i=Math.min(COOKIES_PER_STAGE-1,Math.floor(this.elapsed/COOKIE_SECONDS));const c=this.cookies[i];return c&&!c.result?c:null;}
 act(action){const c=this.active;if(!c||this.elapsed>=c.deadline)return false;const next=transform(c,action);c.animation={fromRotation:c.rotation,fromFlipped:c.flipped,action,at:this.elapsed};c.rotation=next.rotation;c.flipped=next.flipped;c.matchSince=isCorrect(c)?this.elapsed:null;this.events.push({type:'move',action});return true;}
 resolve(c){if(c.result)return;const success=isCorrect(c);c.result=success?'success':'failure';c.resolvedAt=this.elapsed;this.total++;if(success)this.score++;this.events.push({type:c.result,cookie:c});}
 tick(dt){if(this.phase!=='playing')return;this.elapsed=Math.min(STAGE_SECONDS,this.elapsed+Math.max(0,dt));for(const c of this.cookies){if(c.result||c.arrival>this.elapsed)continue;if(c.deadline<=this.elapsed)this.resolve(c);else if(c.matchSince!==null&&this.elapsed-c.matchSince>=.16&&isCorrect(c))this.resolve(c);}if(this.elapsed>=STAGE_SECONDS)this.finishStage();}
 finishStage(){if(this.phase!=='playing')return;const passed=this.score>=PASS_SCORE;this.history.push({stage:this.stage,attempt:this.attempt,success:this.score,total:this.total,passed});if(!passed)this.lives--;this.phase=passed?'tasting':this.lives>0?'retry':'gameover';this.events.push({type:'stageEnd',passed});}
 advance(){if(this.phase!=='tasting')return;if(this.stage===10){this.phase='complete';return;}this.stage++;this.attempt=0;this.phase='intro';}
 drainEvents(){return this.events.splice(0);}
}
