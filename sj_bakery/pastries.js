// Code-native pastry art: one reproducible recipe per stage, renewed per game.
const OUTLINES = [
  [[-96,-58],[96,-58],[-86,60]],
  [[-96,-64],[50,-64],[96,-18],[-74,67]],
  [[-22,-91],[69,-45],[81,30],[-12,79],[-85,12]],
  [[-24,-91],[23,-61],[80,-44],[60,14],[71,47],[-9,82],[-78,24]],
  [[-34,-87],[30,-58],[82,-31],[59,25],[34,78],[-29,61],[-80,7]],
  [[-29,-89],[19,-59],[75,-57],[61,-2],[84,38],[10,79],[-33,52],[-80,11]],
  [[-38,-87],[25,-65],[70,-29],[56,16],[81,51],[3,77],[-42,49],[-82,-4]],
  [[-25,-91],[20,-58],[71,-60],[59,-12],[85,30],[31,57],[-8,82],[-49,47],[-80,-8]],
  [[-34,-89],[11,-65],[66,-69],[58,-16],[83,23],[48,67],[-9,81],[-40,47],[-82,1]],
  [[-24,-92],[16,-60],[61,-68],[54,-24],[85,11],[57,38],[35,77],[-11,63],[-44,80],[-82,2]]
];

function randomFrom(seed) {
  let state = 2166136261;
  for (const character of String(seed)) state = Math.imul(state ^ character.charCodeAt(0), 16777619);
  return () => {
    state = (state + 0x6D2B79F5) | 0;
    let value = Math.imul(state ^ state >>> 15, 1 | state);
    value ^= value + Math.imul(value ^ value >>> 7, 61 | value);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

function inside(points, x, y) {
  let result = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i], [xj, yj] = points[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) result = !result;
  }
  return result;
}

function interiorPoint(points, random, margin = 14) {
  for (let attempt = 0; attempt < 200; attempt++) {
    const x = 77 + random() * 100, y = 70 + random() * 111;
    if ([[0,0],[margin,0],[-margin,0],[0,margin],[0,-margin]].every(([dx,dy]) => inside(points,x+dx,y+dy))) return { x, y };
  }
  return { x: 128, y: 128 };
}

function distinctOrientations(points) {
  const centered=points.map(([x,y])=>[x-128,y-128]);
  const distance=(from,to)=>Math.max(...from.map(([x,y])=>Math.min(...to.map(([a,b])=>Math.hypot(x-a,y-b)))));
  for(let state=1;state<8;state++) {
    const transformed=centered.map(([x,y])=>{
      if(state>=4)x=-x;
      for(let rotation=0;rotation<state%4;rotation++)[x,y]=[-y,x];
      return [x,y];
    });
    if(Math.max(distance(centered,transformed),distance(transformed,centered))<20)return false;
  }
  return true;
}

function distinctSimpleSilhouette(points) {
  // A long point alone can pass a vertex-distance check while most of the
  // cookie still looks mirrored. Early levels must differ over their full area.
  const masks=Array.from({length:8},(_,state)=>{
    const oriented=points.map(([x,y])=>{
      x-=128;y-=128;if(state>=4)x=-x;
      for(let rotation=0;rotation<state%4;rotation++)[x,y]=[-y,x];
      return [x+128,y+128];
    });
    return Array.from({length:32*32},(_,index)=>inside(oriented,(index%32)*8+4,Math.floor(index/32)*8+4));
  });
  for(let state=1;state<8;state++) {
    let different=0,occupied=0;
    for(let pixel=0;pixel<masks[0].length;pixel++) {
      if(masks[0][pixel]||masks[state][pixel])occupied++;
      if(masks[0][pixel]!==masks[state][pixel])different++;
    }
    if(different/occupied<(points.length===3?.48:.42))return false;
  }
  return true;
}

function makeShape(outline,random,simple=false) {
  const sourcePoints=outline.map(([x,y])=>[128+x+(random()-.5)*9,128+y+(random()-.5)*9]);
  for(let attempt=0;attempt<48;attempt++) {
    // Unequal stretches and shear change the silhouette itself, beyond rotation.
    const shapeTransform=[.84+random()*.34,0,(random()-.5)*.52,.86+random()*.27];
    const warp=([x,y])=>[128+(x-128)*shapeTransform[0]+(y-128)*shapeTransform[2],128+(y-128)*shapeTransform[3]];
    const extent=Math.max(...sourcePoints.flatMap(point=>warp(point).map(value=>Math.abs(value-128))));
    if(extent>100)for(let i=0;i<shapeTransform.length;i++)shapeTransform[i]*=100/extent;
    const points=sourcePoints.map(warp);
    const area=Math.abs(points.reduce((sum,[x,y],i)=>{const [a,b]=points[(i+1)%points.length];return sum+x*b-y*a;},0)/2);
    if(area>7200&&distinctOrientations(points)&&(!simple||distinctSimpleSilhouette(points)))return {points,sourcePoints,shapeTransform};
  }
  // Bounded fallback templates are individually checked for all eight states.
  const points=outline.map(([x,y])=>[128+x,128+y]);
  return {points,sourcePoints:points,shapeTransform:[1,0,0,1]};
}

export function createPastryRecipes(seed = 0) {
  return OUTLINES.map((outline, index) => {
    const level = index + 1, random = randomFrom(`${seed}:pastry:${level}`);
    const shape=makeShape(outline,random,level<=2),points=shape.sourcePoints;
    const recipe = { level,...shape,chips: [], strawberries: [], cream: [], chocolate: null, crumbs: [] };
    // Soft baked flecks add texture without obscuring the simple early shapes.
    recipe.crumbs = Array.from({length:6},() => ({...interiorPoint(points,random,5),radius:1.4+random()*1.8}));
    if(level<=2) {
      // Two unequal baked dimples point along the long edge. Their positions
      // belong to the dough, so rotating or flipping also transforms the cue.
      const weights=level===1?[[.69,.14,.17],[.48,.30,.22]]:[[.70,.13,.02,.15],[.46,.33,.06,.15]];
      recipe.bakeMarks=weights.map((weights,index)=>({
        x:points.reduce((sum,point,i)=>sum+point[0]*weights[i],0),
        y:points.reduce((sum,point,i)=>sum+point[1]*weights[i],0),
        radius:index===0?11:6
      }));
    }
    if (level === 5) {
      recipe.chips = Array.from({length:4},(_,i) => {
        const corners = [[102,99],[154,108],[146,157],[95,147]];
        return {x:corners[i][0]+(random()-.5)*12,y:corners[i][1]+(random()-.5)*12,size:7+random()*3,angle:random()*Math.PI};
      });
    }
    if (level === 6) recipe.chocolate = {kind:'half',tilt:-.3+random()*.25,offset:-7+random()*14};
    if (level === 7) {
      const shift = (random()-.5)*12;
      recipe.chocolate = {kind:'zigzag',paths:[Array.from({length:7},(_,i) => [86+i*14,108+(i%2?35:0)+shift+(random()-.5)*7])]};
    }
    if (level === 8 || level === 9) {
      recipe.strawberries = Array.from({length:level===8?1:2},(_,i) => ({
        x:(level===8?117:i===0?104:149)+(random()-.5)*10,
        y:(level===8?114:i===0?108:146)+(random()-.5)*10,
        size:level===8?1.25:1.02,angle:-.4+random()*.8
      }));
    }
    if (level === 9) {
      const shift = (random()-.5)*10;
      recipe.cream = [
        [[84,140+shift],[107,157+shift],[132,143+shift]],
        [[128,89+shift],[151,101+shift],[168,126+shift]]
      ];
    }
    if (level === 10) {
      const shift = (random()-.5)*10, bend = (random()-.5)*8;
      recipe.cream = [
        [[89,109+shift],[109,85+shift],[140,95+shift],[155,119+shift],[126,136+shift],[106,118+shift]],
        [[100,153+shift],[121,173+shift],[149,150+shift],[164,161+shift]],
        [[78,137+shift],[87+bend,130+shift],[96,139+shift]],
        [[155,89+shift],[161+bend,78+shift],[170,91+shift]]
      ];
    }
    return recipe;
  });
}

function pastryPath(context, points) {
  context.beginPath();
  for (let i = 0; i < points.length; i++) {
    const previous = points[(i+points.length-1)%points.length], current = points[i], next = points[(i+1)%points.length];
    const before = current.map((value,j) => value*.94+previous[j]*.06);
    const after = current.map((value,j) => value*.94+next[j]*.06);
    if (i===0) context.moveTo(...before); else context.lineTo(...before);
    context.quadraticCurveTo(...current,...after);
  }
  context.closePath();
}

function drawStroke(context, points, color, width) {
  context.beginPath();
  context.moveTo(...points[0]);
  for (let i=1;i<points.length-1;i++) {
    const middle = points[i].map((value,j) => (value+points[i+1][j])/2);
    context.quadraticCurveTo(...points[i],...middle);
  }
  context.lineTo(...points[points.length-1]);
  context.strokeStyle=color;context.lineWidth=width;context.lineCap='round';context.lineJoin='round';context.stroke();
}

function drawStrawberry(context, strawberry) {
  context.save();context.translate(strawberry.x,strawberry.y);context.rotate(strawberry.angle);context.scale(strawberry.size,strawberry.size);
  context.shadowColor='#66341855';context.shadowBlur=3;context.shadowOffsetY=2;
  context.beginPath();context.moveTo(0,20);context.bezierCurveTo(-29,1,-16,-18,0,-10);context.bezierCurveTo(19,-20,27,3,0,20);
  context.fillStyle='#df5144';context.fill();context.shadowColor='transparent';context.lineWidth=2;context.strokeStyle='#a73530';context.stroke();
  context.fillStyle='#fff0b0';
  for (const [x,y] of [[-7,-3],[7,-2],[-9,5],[3,6],[-2,14],[11,4]]) {context.beginPath();context.ellipse(x,y,1.2,2,.25,0,Math.PI*2);context.fill();}
  context.fillStyle='#68863c';context.beginPath();context.moveTo(0,-7);context.lineTo(-13,-16);context.lineTo(-3,-14);context.lineTo(1,-22);context.lineTo(5,-13);context.lineTo(15,-15);context.lineTo(7,-6);context.closePath();context.fill();context.restore();
}

function drawPastry(canvas, recipe) {
  const context = canvas.getContext('2d');
  context.save();context.translate(128,128);context.transform(...recipe.shapeTransform,0,0);context.translate(-128,-128);
  const dough = context.createRadialGradient(115,104,12,128,126,100);
  dough.addColorStop(0,'#f9d991');dough.addColorStop(.68,'#efbc6b');dough.addColorStop(1,'#cd853b');
  context.save();pastryPath(context,recipe.sourcePoints);context.shadowColor='#8a49164d';context.shadowBlur=8;context.shadowOffsetY=5;
  context.fillStyle=dough;context.fill();context.shadowColor='transparent';context.strokeStyle='#a6652d';context.lineWidth=3.7;context.stroke();
  context.clip();context.fillStyle='#bc762326';
  for (const crumb of recipe.crumbs) {context.beginPath();context.arc(crumb.x,crumb.y,crumb.radius,0,Math.PI*2);context.fill();}
  for (const mark of recipe.bakeMarks??[]) {
    context.beginPath();context.arc(mark.x,mark.y,mark.radius,0,Math.PI*2);
    context.fillStyle='#8a4f25';context.fill();context.strokeStyle='#f7cf80';context.lineWidth=2;context.stroke();
    context.beginPath();context.arc(mark.x-1,mark.y-1,mark.radius*.58,0,Math.PI*2);context.fillStyle='#6a391d';context.fill();
  }
  const chocolate=recipe.chocolate;
  if (chocolate?.kind==='half') {
    context.save();context.translate(128+chocolate.offset,128);context.rotate(chocolate.tilt);
    const glaze=context.createLinearGradient(0,-70,110,65);glaze.addColorStop(0,'#754329');glaze.addColorStop(1,'#4d2d21');
    context.fillStyle=glaze;context.fillRect(0,-170,170,340);context.strokeStyle='#ac795355';context.lineWidth=4;context.strokeRect(8,-160,150,320);context.restore();
  }
  if (chocolate?.kind==='zigzag') for (const path of chocolate.paths) {
    drawStroke(context,path,'#704129',12);drawStroke(context,path.map(([x,y])=>[x,y-1.5]),'#956143',5);
  }
  for (const chip of recipe.chips) {
    context.save();context.translate(chip.x,chip.y);context.rotate(chip.angle);context.beginPath();context.moveTo(-chip.size,chip.size*.7);context.lineTo(-chip.size*.65,-chip.size*.65);context.lineTo(chip.size*.5,-chip.size);context.lineTo(chip.size,chip.size*.4);context.closePath();context.fillStyle='#66412b';context.fill();context.strokeStyle='#96613c';context.lineWidth=1.5;context.stroke();context.restore();
  }
  for (const path of recipe.cream) {
    drawStroke(context,path.map(([x,y])=>[x,y+2]),'#bf965c77',11);drawStroke(context,path,'#fff7d9',9);drawStroke(context,path.map(([x,y])=>[x,y-1]),'#fffef3',3);
  }
  for (const strawberry of recipe.strawberries) drawStrawberry(context,strawberry);
  context.restore();context.restore();
}

export function createPastryTiles(seed = 0) {
  return createPastryRecipes(seed).map(recipe => {
    const canvas=document.createElement('canvas');canvas.width=256;canvas.height=256;drawPastry(canvas,recipe);return canvas;
  });
}
