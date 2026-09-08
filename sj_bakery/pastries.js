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

function createBakedRecipes(seed = 0) {
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

// Menus describe the actual material rendered below, including the three easy
// introductory silhouettes. The dad route retains the original baked recipes.
export const SNACK_MENUS = {
  mom: {label:'캐러멜 · 초콜릿', stages:[
    {name:'황금 캐러멜',color:'#d28a27'},
    {name:'밀크초콜릿 조각',color:'#86503e'},
    {name:'버터스카치 캔디',color:'#d59b38'},
    {name:'다크초콜릿 봉봉',color:'#654133'},
    {name:'초코칩 캐러멜',color:'#c18433'},
    {name:'반반 캐러멜 초콜릿',color:'#9c653c'},
    {name:'캐러멜 리본 초콜릿',color:'#875231'},
    {name:'솔티드 캐러멜 초콜릿',color:'#99613d'},
    {name:'캐러멜 소용돌이',color:'#bf8c3d'},
    {name:'초콜릿 캐러멜 미로',color:'#9e6c42'}
  ]},
  suan: {label:'젤리 · 마시멜로', stages:[
    {name:'딸기 젤리',color:'#e86c85'},
    {name:'바닐라 마시멜로',color:'#c783a1'},
    {name:'포도 젤리',color:'#a56dc9'},
    {name:'구름 마시멜로',color:'#72abc0'},
    {name:'과일 알갱이 젤리',color:'#61aea8'},
    {name:'반반 젤리 마시멜로',color:'#d580a4'},
    {name:'딸기 리본 마시멜로',color:'#ce81a2'},
    {name:'말랑말랑 젤리 구름',color:'#cb94b4'},
    {name:'두 빛깔 젤리 구름',color:'#9672bf'},
    {name:'무지개 젤리 미로',color:'#53a89d'}
  ]},
  jeongan: {label:'엉뚱한 마법 간식', stages:[
    {name:'초록 젤리 품은 빵',color:'#84a94a'},
    {name:'반은 초코 반은 빵',color:'#b08048'},
    {name:'보랏빛 젤리 파이',color:'#a282b8'},
    {name:'파란 별빛 초코빵',color:'#718ba5'},
    {name:'톡톡 캔디 초코빵',color:'#a17883'},
    {name:'초코빵 속 무지개 젤리',color:'#a58253'},
    {name:'초록 젤리 리본 빵',color:'#98ad5c'},
    {name:'딸기 젤리 용암빵',color:'#c97870'},
    {name:'마시멜로 구름 초코빵',color:'#a18bab'},
    {name:'엉뚱한 마법 간식 미로',color:'#9c82b4'}
  ]}
};

const THEMES = {
  mom: [
    {material:'caramel'}, {material:'milk'}, {material:'caramel',extra:'butterscotch'},
    {material:'dark'}, {material:'caramel',extra:'chips'},
    {material:'caramel',half:'dark'}, {material:'dark',extra:'ribbon'},
    {material:'dark',extra:'salted'}, {material:'caramel',extra:'swirl'},
    {material:'dark',extra:'maze'}
  ],
  suan: [
    {material:'berry'}, {material:'mallow'}, {material:'grape'},
    {material:'cloud'}, {material:'mint',extra:'fruit'},
    {material:'berry',half:'mallow'}, {material:'mallow',extra:'ribbon'},
    {material:'mallow',extra:'jelly'}, {material:'grape',extra:'pillows'},
    {material:'mint',extra:'maze'}
  ],
  jeongan: [
    {material:'bread',extra:'window',filling:'lime'}, {material:'bread',half:'dark'},
    {material:'bread',extra:'window',filling:'grape'},
    {material:'dark',half:'bread',extra:'stars'}, {material:'dark',extra:'fruit'},
    {material:'bread',half:'dark',extra:'window',filling:'rainbow'},
    {material:'bread',extra:'ribbon'}, {material:'bread',extra:'window',filling:'berry'},
    {material:'dark',half:'bread',extra:'pillows'}, {material:'bread',half:'dark',extra:'maze'}
  ]
};

export function createPastryRecipes(seed = 0, playerId = 'dad') {
  if (!THEMES[playerId]) return createBakedRecipes(seed);
  return createBakedRecipes(`${seed}:${playerId}`).map(recipe => {
    const random=randomFrom(`${seed}:${playerId}:finish:${recipe.level}`);
    const center=recipe.sourcePoints.reduce((sum,[x,y])=>[sum[0]+x/recipe.sourcePoints.length,sum[1]+y/recipe.sourcePoints.length],[0,0]);
    return {...recipe,playerId,theme:{...THEMES[playerId][recipe.level-1],
      center,tilt:-.24+random()*.32,shift:(random()-.5)*10,
      bubbles:Array.from({length:recipe.level<3?2:4},()=>({...interiorPoint(recipe.sourcePoints,random,10),radius:3+random()*4})),
      flecks:Array.from({length:4},()=>({...interiorPoint(recipe.sourcePoints,random,12),angle:random()*Math.PI}))
    }};
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


const MATERIALS = {
  bread:['#ffe0a0','#efb96b','#bb702f','#915125'],
  caramel:['#ffe49b','#d99028','#96501e','#824117'],
  milk:['#b58060','#80513e','#51302b','#40241e'],
  dark:['#8d6252','#56362f','#342223','#2d1b1b'],
  berry:['#ffb8ceeb','#f15191ee','#b52765f5','#a62760'],
  grape:['#e9b9ffeb','#a966d7ee','#6c3499f5','#653081'],
  mint:['#b9ffe4eb','#55cbbdee','#288c91f5','#247e7d'],
  lime:['#e9ff97eb','#91da49ee','#46a047f5','#488b38'],
  mallow:['#fffefa','#fff3f0','#e7bdd1','#b987a3'],
  cloud:['#fbffff','#e4f6ff','#9fc9e2','#739cb7']
};

function materialGradient(context,material) {
  const colors=MATERIALS[material];
  const gradient=context.createRadialGradient(94,85,8,127,127,119);
  gradient.addColorStop(0,colors[0]);gradient.addColorStop(.58,colors[1]);gradient.addColorStop(1,colors[2]);
  return gradient;
}

function surface(context,recipe,material,shadow=false) {
  const colors=MATERIALS[material],jelly=['berry','grape','mint','lime'].includes(material);
  const soft=material==='mallow'||material==='cloud';
  context.save();pastryPath(context,recipe.sourcePoints);
  if(shadow){context.shadowColor='#50332940';context.shadowBlur=7;context.shadowOffsetY=5;}
  context.fillStyle=materialGradient(context,material);context.fill();context.shadowColor='transparent';
  context.strokeStyle=colors[3];context.lineWidth=soft?3:4;context.stroke();context.clip();
  if(material==='bread') {
    context.fillStyle='#ac612449';
    for(const crumb of recipe.crumbs){context.beginPath();context.arc(crumb.x,crumb.y,crumb.radius,0,Math.PI*2);context.fill();}
  } else {
    if(soft){pastryPath(context,recipe.sourcePoints);context.strokeStyle=colors[2];context.lineWidth=13;context.stroke();}
    // A broad rounded bevel makes marshmallows pillowy, while the sharp inset
    // and specular glints make candy and gummy jelly read as different foods.
    const inset=recipe.sourcePoints.map(([x,y])=>[128+(x-128)*.86,128+(y-128)*.86]);
    pastryPath(context,inset);context.lineWidth=soft?13:jelly?8:4;
    context.strokeStyle=soft?'#ffffff95':jelly?'#ffffff47':'#ffe2bd45';context.stroke();
    const [a,b]=recipe.sourcePoints;
    const edge=[[a[0]*.72+b[0]*.28,a[1]*.72+b[1]*.28+13],[a[0]*.48+b[0]*.52,a[1]*.48+b[1]*.52+13]];
    drawStroke(context,edge,soft?'#ffffffcc':jelly?'#fffaffc9':'#fff0cb9c',soft?10:6);
    if(jelly) for(const bubble of recipe.theme.bubbles) {
      context.beginPath();context.arc(bubble.x,bubble.y,bubble.radius,0,Math.PI*2);
      context.fillStyle='#ffffff25';context.fill();context.lineWidth=1.5;context.strokeStyle='#ffffff80';context.stroke();
    }
  }
  context.restore();
}

function jellyOval(context,x,y,rx,ry,material='berry') {
  const colors=MATERIALS[material],gradient=context.createLinearGradient(x-rx,y-ry,x+rx,y+ry);
  gradient.addColorStop(0,colors[0]);gradient.addColorStop(.5,colors[1]);gradient.addColorStop(1,colors[2]);
  context.beginPath();context.ellipse(x,y,rx,ry,-.16,0,Math.PI*2);context.fillStyle=gradient;context.fill();
  context.lineWidth=2.4;context.strokeStyle=colors[3];context.stroke();
  context.beginPath();context.ellipse(x-rx*.25,y-ry*.37,rx*.44,ry*.18,-.2,0,Math.PI*2);context.fillStyle='#ffffffae';context.fill();
}

function pillow(context,x,y,size=1) {
  context.save();context.translate(x,y);context.rotate(-.28);context.scale(size,size);
  context.beginPath();context.roundRect(-17,-14,34,28,10);
  const gradient=context.createLinearGradient(0,-14,0,14);gradient.addColorStop(0,'#fffefa');gradient.addColorStop(.7,'#fff0ed');gradient.addColorStop(1,'#d4a7c6');
  context.shadowColor='#53385244';context.shadowBlur=3;context.shadowOffsetY=3;context.fillStyle=gradient;context.fill();context.shadowColor='transparent';context.strokeStyle='#bd8faf';context.lineWidth=1.7;context.stroke();
  drawStroke(context,[[-9,-7],[6,-7]],'#ffffff',4);context.restore();
}

function magicStar(context,x,y,color,size=10) {
  context.beginPath();
  for(let point=0;point<8;point++) {
    const angle=point*Math.PI/4-Math.PI/2,radius=point%2?size*.36:size;
    const px=x+Math.cos(angle)*radius,py=y+Math.sin(angle)*radius;
    if(point===0)context.moveTo(px,py);else context.lineTo(px,py);
  }
  context.closePath();context.fillStyle=color;context.fill();context.lineWidth=1.5;context.strokeStyle='#70557b';context.stroke();
}

function drawThemedPastry(canvas,recipe) {
  const context=canvas.getContext('2d'),theme=recipe.theme,{material,extra}=theme;
  context.save();context.translate(128,128);context.transform(...recipe.shapeTransform,0,0);context.translate(-128,-128);
  surface(context,recipe,material,true);
  context.save();pastryPath(context,recipe.sourcePoints);context.clip();
  if(theme.half) {
    context.save();context.translate(128,128);context.rotate(theme.tilt);
    context.beginPath();context.rect(0,-200,200,400);context.clip();context.rotate(-theme.tilt);context.translate(-128,-128);
    surface(context,recipe,theme.half);context.restore();
    const edge=[[128+Math.sin(theme.tilt)*85,43],[128-Math.sin(theme.tilt)*87,215]];
    drawStroke(context,edge,theme.half==='mallow'?'#fff4dc':'#d89a6255',3);
  }
  // Unequal, off-centre cues retain the easy stages' unmistakable orientation.
  for(const mark of recipe.bakeMarks??[]) {
    const light=['milk','dark','berry','grape','mint'].includes(material);
    context.beginPath();context.arc(mark.x,mark.y,mark.radius,0,Math.PI*2);
    context.fillStyle=light?'#ffecd1':recipe.playerId==='suan'?'#a55170':'#8b4d24';context.fill();
    context.strokeStyle=light?'#59324388':'#fff5db';context.lineWidth=2;context.stroke();
    context.beginPath();context.arc(mark.x-2,mark.y-2,mark.radius*.34,0,Math.PI*2);context.fillStyle='#ffffffbb';context.fill();
  }
  const [cx,cy]=theme.center,shift=theme.shift;
  const ribbon=recipe.playerId==='mom'?'#efbb66':recipe.playerId==='suan'?'#ed71a6':'#8bdd64';
  if(extra==='butterscotch') {
    for(let i=0;i<3;i++)drawStroke(context,[[70+i*37,62],[88+i*37,136],[108+i*37,205]],i===1?'#fae3a0':'#b26b31',10);
  }
  if(extra==='chips') for(const chip of recipe.chips) {
    context.save();context.translate(chip.x,chip.y);context.rotate(chip.angle);
    context.fillStyle='#51332b';context.fillRect(-chip.size,-chip.size,chip.size*2,chip.size*1.7);
    context.fillStyle='#bd9275';context.fillRect(-chip.size+2,-chip.size+2,chip.size*1.1,2);context.restore();
  }
  if(extra==='fruit') theme.flecks.forEach((fleck,index)=>{
    context.save();context.translate(fleck.x,fleck.y);context.rotate(fleck.angle);
    context.beginPath();context.roundRect(-8,-7,16,14,3);context.fillStyle=['#ffbc55','#ef6f95','#8be79d','#c496ed'][index];context.fill();
    context.strokeStyle='#ffffffa0';context.lineWidth=2;context.stroke();context.restore();
  });
  if(extra==='ribbon') {
    const path=[[73,94+shift],[106,130+shift],[133,99+shift],[166,142+shift],[187,121+shift]];
    drawStroke(context,path,'#53334140',15);drawStroke(context,path,ribbon,12);drawStroke(context,path.map(([x,y])=>[x,y-2]),'#ffffff65',3);
  }
  if(extra==='window') {
    // A thick golden crust and pale crumb ring visibly expose the jelly filling.
    const rx=recipe.level===1?27:36,ry=recipe.level===1?20:27;
    context.beginPath();context.ellipse(cx,cy+7,rx+8,ry+8,-.16,0,Math.PI*2);
    context.fillStyle='#fff0c4';context.fill();context.strokeStyle='#a46633';context.lineWidth=3;context.stroke();
    jellyOval(context,cx,cy+7,rx,ry,theme.filling==='rainbow'?'mint':theme.filling);
    if(theme.filling==='rainbow') {
      drawStroke(context,[[cx-24,cy+4],[cx,cy+8],[cx+24,cy]],'#fa7bac',9);
      drawStroke(context,[[cx-22,cy+14],[cx+3,cy+21],[cx+22,cy+11]],'#ffdc78',7);
    }
    if(recipe.level===8) {
      drawStroke(context,[[cx+15,cy+24],[cx+9,cy+47],[cx+24,cy+59]],'#de4779',12);
      drawStroke(context,[[cx+15,cy+24],[cx+9,cy+47],[cx+24,cy+59]],'#ff9ba2',4);
    }
  }
  if(extra==='jelly') jellyOval(context,cx-4,cy+2,34,26,'berry');
  if(extra==='salted') {
    jellyOval(context,cx-2,cy+3,36,29,'caramel');
    drawStroke(context,[[cx-19,cy-8],[cx-2,cy-13]],'#ffecb4b0',5);
    for(const [x,y] of [[-20,-1],[3,-11],[18,9],[-7,18]]){context.save();context.translate(cx+x,cy+y);context.rotate(.4);context.fillStyle='#fff5dc';context.fillRect(-2,-3,4,6);context.restore();}
  }
  if(extra==='pillows') {
    pillow(context,106+shift,107,1.08);pillow(context,151,148-shift,.91);
    if(recipe.playerId==='jeongan')magicStar(context,143,99,'#b7e9ff',9);
  }
  if(extra==='stars') {
    drawStroke(context,[[86,135],[119,99],[159,139]],'#76b9e8',13);
    magicStar(context,105,107,'#fff1a8',13);magicStar(context,157,142,'#bdeaff',10);
  }
  if(extra==='swirl') {
    const path=[[82,145],[90,100],[132,84],[163,119],[144,156],[106,145],[115,116],[142,119]];
    drawStroke(context,path,'#683c31',11);drawStroke(context,path.map(([x,y])=>[x,y-2]),'#f8ddb1',3);
  }
  if(extra==='maze') {
    const paths=recipe.cream.length?recipe.cream:[[[80,95],[112,137],[142,97],[178,132]],[[91,157],[124,144],[157,171]]];
    const colors=recipe.playerId==='mom'?['#e4a550','#fff1c8','#b17946','#f5dca7']:recipe.playerId==='suan'?['#ff7eb3','#fff3d8','#ffe477','#bd82e2']:['#ad82ee','#87e279','#ffb9d3','#9adefa'];
    paths.forEach((path,index)=>{
      drawStroke(context,path,'#56374e4d',13);drawStroke(context,path,colors[index%colors.length],10);
      drawStroke(context,path.map(([x,y])=>[x,y-2]),'#ffffff82',3);
    });
    if(recipe.playerId==='jeongan')magicStar(context,101,142,'#ffe790',11);
  }
  context.restore();context.restore();
}

export function createPastryTiles(seed = 0, playerId = 'dad') {
  return createPastryRecipes(seed,playerId).map(recipe => {
    const canvas=document.createElement('canvas');canvas.width=256;canvas.height=256;
    if(recipe.theme)drawThemedPastry(canvas,recipe);else drawPastry(canvas,recipe);
    return canvas;
  });
}
