export const STAGE_SECONDS=60;
export const QUESTION_SECONDS=3;
export const COOKIE_SECONDS=QUESTION_SECONDS;
export const QUESTIONS_PER_STAGE=20;
export const COOKIES_PER_STAGE=QUESTIONS_PER_STAGE;
export const PASS_SCORE=15;
export const PASTRIES=[
 {name:'버터의 첫걸음',color:'#dba04a'}, {name:'초코칩이 콕콕',color:'#cd8f3f'},
 {name:'초콜릿 반쪽의 비밀',color:'#8b5133'}, {name:'지그재그 초코 산책',color:'#875232'},
 {name:'딸기의 등장',color:'#cb6352'}, {name:'딸기와 크림의 춤',color:'#d87f69'},
 {name:'크림으로 그린 낙서',color:'#c3a074'}, {name:'달콤한 삼각관계',color:'#b96b4a'},
 {name:'빙글빙글 크림 미로',color:'#ba885d'}, {name:'엉뚱한 제과장의 걸작',color:'#b44740'}
];
const EASY_PASTRIES=[
 {name:'세모 버터의 첫걸음',color:'#e8b968'},
 {name:'네모난 버터 친구',color:'#e1ad58'},
 {name:'다섯 모서리 산책',color:'#dba04a'}
];
export const getPastryLevel=stage=>Math.max(1,Math.min(10,stage));
export const STAGES=Array.from({length:10},(_,index)=>{
 const pastryIndex=Math.max(0,index-3);
 return {...(index<3?EASY_PASTRIES[index]:PASTRIES[pastryIndex]),shapeLevel:index+1,pastryIndex};
});
export const mod4=n=>((n%4)+4)%4;
export function transform(o,action){
 if(action==='left')return {rotation:mod4(o.rotation-1),flipped:o.flipped};
 if(action==='right')return {rotation:mod4(o.rotation+1),flipped:o.flipped};
 if(action==='up')return {rotation:mod4(-o.rotation),flipped:!o.flipped};
 if(action==='down')return {rotation:mod4(2-o.rotation),flipped:!o.flipped};
 return {...o};
}
export const sameOrientation=(a,b)=>a.rotation===b.rotation&&a.flipped===b.flipped;
export const applyInstructions=(orientation,instructions)=>instructions.reduce(transform,{...orientation});

const ORIENTATIONS=Array.from({length:8},(_,index)=>({rotation:index%4,flipped:index>=4}));
const ACTIONS=['left','right','up','down'];
const shuffle=(values,random)=>{
 const copy=[...values];
 for(let i=copy.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[copy[i],copy[j]]=[copy[j],copy[i]];}
 return copy;
};

export class BakeryGame{
 constructor(random=Math.random){this.random=random;this.reset();}
 reset(){this.stage=1;this.lives=3;this.history=[];this.phase='ready';this.elapsed=0;this.questionElapsed=0;this.cookies=[];this.score=0;this.total=0;this.attempt=0;this.events=[];this.question=null;this.questionIndex=0;this.cooldown=0;}
 makeQuestion(id){
  const initial={rotation:Math.floor(this.random()*4),flipped:this.random()<.5};
  const instructions=Array.from({length:this.stage},()=>ACTIONS[Math.floor(this.random()*ACTIONS.length)]);
  const answer=applyInstructions(initial,instructions);
  const wrong=shuffle(ORIENTATIONS.filter(o=>!sameOrientation(o,answer)),this.random).slice(0,2);
  const candidates=shuffle([{...answer},...wrong],this.random);
  return {id,initial,instructions,answer,candidates,correctIndex:candidates.findIndex(o=>sameOrientation(o,answer)),selectedIndex:null,result:null,resolvedAt:null};
 }
 beginStage(){this.attempt++;this.elapsed=0;this.questionElapsed=0;this.score=0;this.total=0;this.questionIndex=0;this.cooldown=0;this.cookies=[];this.events=[];this.question=this.makeQuestion(0);this.phase='playing';}
 get active(){return this.phase==='playing'&&this.question&&!this.question.result?this.question:null;}
 choose(index){const question=this.active;if(!question||index<0||index>2)return false;question.selectedIndex=index;this.resolve(question,index===question.correctIndex);return true;}
 resolve(question,success=false){
  if(question.result)return;question.result=success?'success':'failure';question.resolvedAt=this.elapsed;this.total++;if(success)this.score++;this.cooldown=.62;
  this.cookies.push(question);this.events.push({type:question.result,cookie:question,question});
 }
 nextQuestion(){
  if(this.total>=QUESTIONS_PER_STAGE){this.finishStage();return;}
  this.questionIndex=this.total;this.questionElapsed=0;this.question=this.makeQuestion(this.questionIndex);this.events.push({type:'question',question:this.question});
 }
 tick(dt){
  if(this.phase!=='playing')return;dt=Math.max(0,dt);this.elapsed+=dt;
  if(this.question?.result){this.cooldown-=dt;if(this.cooldown<=0)this.nextQuestion();return;}
  this.questionElapsed=Math.min(QUESTION_SECONDS,this.questionElapsed+dt);
  if(this.questionElapsed>=QUESTION_SECONDS)this.resolve(this.question,false);
 }
 finishStage(){if(this.phase!=='playing')return;const passed=this.score>=PASS_SCORE;this.history.push({stage:this.stage,attempt:this.attempt,success:this.score,total:this.total,passed});if(!passed)this.lives--;this.phase=passed?'tasting':this.lives>0?'retry':'gameover';this.events.push({type:'stageEnd',passed});}
 advance(){if(this.phase!=='tasting')return;if(this.stage===10){this.phase='complete';return;}this.stage++;this.attempt=0;this.phase='intro';}
 drainEvents(){return this.events.splice(0);}
}
