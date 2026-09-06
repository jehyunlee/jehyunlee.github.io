export const DIRECTIONS = [[0,-1],[1,0],[0,1],[-1,0]];
export const RELATIVE_DIRECTIONS = {up:0,right:1,down:2,left:3};
export const LOGO_COLOR = '#ef1702';
export function lightenHex(hex,fraction) {
  return '#'+[1,3,5].map(index=>{
    const value=parseInt(hex.slice(index,index+2),16);
    return Math.round(value+(255-value)*fraction).toString(16).padStart(2,'0');
  }).join('');
}
export const UNVISITED_COLOR = lightenHex(LOGO_COLOR,.2);
export function logoPosition(stage,id,cellSize=3.2) {
  const [x,z]=stage.cells[id];
  return [(x+(stage.logoBox[0]-258)/12)*cellSize,0,(z+(stage.logoBox[1]-300)/12)*cellSize];
}
export function overviewCameraPose(bounds,aspect,fov=48) {
  const width=bounds.maxX-bounds.minX+10,depth=bounds.maxZ-bounds.minZ+10;
  const span=Math.max(width/(.86*aspect),depth/.5);
  const distance=span/(2*Math.tan(fov*Math.PI/360));
  const look=[(bounds.minX+bounds.maxX)/2,0,(bounds.minZ+bounds.maxZ)/2+depth*.15];
  return {look,position:[look[0],distance*Math.cos(Math.PI/12),look[2]+distance*Math.sin(Math.PI/12)]};
}
export function relativeDirection(input, heading) {
  return (heading + RELATIVE_DIRECTIONS[input]) % 4;
}
export class MazeSession {
  constructor(stages) { this.stages=stages; this.reset(); }
  reset() {
    this.stageIndex=0; this.steps=0; this.elapsed=0; this.cleared=[];
    this.stageTimes=this.stages.map(()=>null);
    this.visitedByStage=this.stages.map(()=>new Set());
    this.loadStage();
  }
  loadStage() {
    this.stage=this.stages[this.stageIndex]; this.cell=this.stage.start;
    this.visited=this.visitedByStage[this.stageIndex];this.visited.add(this.cell);
    this.stageSteps=0; this.stageElapsed=0;
    this.finished=false;
    this.lookup=new Map(this.stage.cells.map((c,i)=>[c.join(','),i]));
    const [x,z]=this.stage.cells[this.cell], [nx,nz]=this.stage.cells[this.stage.links[this.cell][0]];
    this.heading=DIRECTIONS.findIndex(([dx,dz])=>x+dx===nx && z+dz===nz);
  }
  move(direction) {
    if(this.finished) return {allowed:false,complete:true};
    const from=this.cell, [x,z]=this.stage.cells[from], [dx,dz]=DIRECTIONS[direction];
    const to=this.lookup.get(`${x+dx},${z+dz}`);
    this.heading=direction;
    if(to===undefined || !this.stage.links[from].includes(to)) return {allowed:false,complete:false,from};
    this.cell=to; this.steps++; this.stageSteps++; this.visited.add(to);
    this.finished=to===this.stage.goal;
    if(this.finished) this.cleared.push(this.stageIndex);
    return {allowed:true,from,to,complete:this.finished};
  }
  nextStage() {
    if(!this.finished || this.stageIndex===this.stages.length-1) return false;
    this.recordStage();
    this.stageIndex++; this.loadStage(); return true;
  }
  recordStage() {
    if(!this.finished)return false;
    if(this.stageTimes[this.stageIndex]===null)this.stageTimes[this.stageIndex]=Math.round(this.stageElapsed*100)/100;
    return true;
  }
  get totalClearTime() { return this.stageTimes.reduce((total,time)=>total+(time??0),0); }
  tick(seconds) {
    if(this.stageTimes[this.stageIndex]!==null)return;
    this.elapsed+=seconds; this.stageElapsed+=seconds;
  }
}
// Grid traversal checks every crossed wall, including both sides of a corner.
export function hasLineOfSight(stage, lookup, from, to) {
  if(from===to) return true;
  let [x,z]=stage.cells[from]; const [tx,tz]=stage.cells[to];
  const dx=tx-x,dz=tz-z,sx=Math.sign(dx),sz=Math.sign(dz);
  const deltaX=dx===0?Infinity:1/Math.abs(dx),deltaZ=dz===0?Infinity:1/Math.abs(dz);
  let tX=deltaX/2,tZ=deltaZ/2,here=from;
  const linked=(a,b)=>a!==undefined && b!==undefined && stage.links[a].includes(b);
  for(let n=0;n<100;n++) {
    if(x===tx&&z===tz) return true;
    if(Math.abs(tX-tZ)<1e-8) {
      const a=lookup.get(`${x+sx},${z}`), b=lookup.get(`${x},${z+sz}`);
      const end=lookup.get(`${x+sx},${z+sz}`);
      if(!linked(here,a)||!linked(here,b)||!linked(a,end)||!linked(b,end)) return false;
      x+=sx;z+=sz;here=end;tX+=deltaX;tZ+=deltaZ;
    } else if(tX<tZ) {
      const next=lookup.get(`${x+sx},${z}`);
      if(!linked(here,next)) return false;
      x+=sx;here=next;tX+=deltaX;
    } else {
      const next=lookup.get(`${x},${z+sz}`);
      if(!linked(here,next)) return false;
      z+=sz;here=next;tZ+=deltaZ;
    }
  }
  return false;
}
