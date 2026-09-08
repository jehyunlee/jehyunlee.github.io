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
    {name:'개구리 초콜릿',color:'#80533b'},
    {name:'보랏빛 젤리 파이',color:'#a282b8'},
    {name:'맨드레이크',color:'#80944d'},
    {name:'톡톡 캔디 초코빵',color:'#a17883'},
    {name:'모든 맛 젤리',color:'#bb79a8'},
    {name:'초록 젤리 리본 빵',color:'#98ad5c'},
    {name:'마법빗자루',color:'#b88343'},
    {name:'마시멜로 구름 초코빵',color:'#a18bab'},
    {name:'골든스니치',color:'#c99b38'}
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
    {material:'bread',extra:'window',filling:'lime'}, {material:'milk',extra:'frog'},
    {material:'bread',extra:'window',filling:'grape'},
    {material:'bread',extra:'mandrake'}, {material:'dark',extra:'fruit'},
    {material:'mint',extra:'beans'},
    {material:'bread',extra:'ribbon'}, {material:'caramel',extra:'broom'},
    {material:'dark',half:'bread',extra:'pillows'}, {material:'caramel',extra:'snitch'}
  ]
};

export function createPastryRecipes(seed = 0, playerId = 'dad') {
  if (!THEMES[playerId]) return createBakedRecipes(seed);
  return createBakedRecipes(`${seed}:${playerId}`).map(recipe => {
    const random=randomFrom(`${seed}:${playerId}:finish:${recipe.level}`);
    const center=recipe.sourcePoints.reduce((sum,[x,y])=>[sum[0]+x/recipe.sourcePoints.length,sum[1]+y/recipe.sourcePoints.length],[0,0]);
    const themed={...recipe,playerId,theme:{...THEMES[playerId][recipe.level-1],
      center,tilt:-.24+random()*.32,shift:(random()-.5)*10,
      bubbles:Array.from({length:recipe.level<3?2:4},()=>({...interiorPoint(recipe.sourcePoints,random,10),radius:3+random()*4})),
      flecks:Array.from({length:4},()=>({...interiorPoint(recipe.sourcePoints,random,12),angle:random()*Math.PI}))
    }};
    return playerId==='jeongan'&&[2,4,6,8,10].includes(recipe.level)?makeMagicRecipe(themed,seed):themed;
  });
}


// These storybook sweets have their own recognisable silhouettes. The
// deliberately unequal limbs and fixed colour landmarks move with the sweet,
// so the player can tell all eight rotations/reflections apart at belt size.
const FROG_OUTLINE = [[63,92],[64,63],[78,49],[98,48],[108,69],[124,54],[144,58],[152,81],[148,101],[170,116],[183,132],[203,116],[224,114],[223,136],[203,159],[173,174],[171,193],[152,208],[113,204],[98,184],[81,173],[67,188],[43,188],[35,173],[56,155],[70,146],[73,124],[49,135],[31,124],[38,109],[57,111]];
const MANDRAKE_ROOT = [[102,79],[143,82],[163,100],[162,123],[180,124],[199,148],[196,164],[182,162],[167,144],[158,158],[164,180],[191,208],[184,221],[157,207],[137,181],[128,166],[114,180],[99,218],[79,223],[72,213],[85,177],[94,161],[90,145],[70,155],[54,149],[55,136],[82,126],[87,100]];
const BEAN_OUTLINE = [[48,59],[82,45],[103,47],[129,31],[151,36],[156,76],[184,86],[199,112],[187,145],[190,167],[192,209],[166,219],[133,202],[113,196],[89,216],[61,205],[49,178],[56,144],[44,113]];
const BROOM_OUTLINE = [[74,40],[79,29],[96,27],[108,39],[110,61],[105,78],[119,109],[143,137],[167,146],[197,176],[217,203],[197,207],[183,223],[157,215],[140,229],[119,216],[101,222],[85,207],[95,182],[102,161],[113,145],[96,118],[83,88],[82,70],[88,51],[86,45]];
const SNITCH_OUTLINE = [[99,128],[74,118],[55,96],[37,66],[24,34],[48,54],[55,31],[74,65],[83,53],[99,84],[109,106],[128,102],[151,108],[166,128],[187,102],[209,87],[231,76],[216,105],[232,99],[216,128],[229,128],[203,150],[169,152],[162,176],[144,190],[121,193],[98,182],[83,162],[81,144]];

function makeMagicRecipe(recipe,seed) {
  const random=randomFrom(`${seed}:jeongan:storybook:${recipe.level}`);
  const outline=({2:FROG_OUTLINE,4:MANDRAKE_ROOT,6:BEAN_OUTLINE,8:BROOM_OUTLINE,10:SNITCH_OUTLINE})[recipe.level];
  // Keep the familiar pose and number of ingredients; small seeded stretches
  // change each new batch without changing the stage's visual difficulty.
  const shapeTransform=[.965+random()*.04,0,(random()-.5)*.04,.97+random()*.03];
  const sourcePoints=outline.map(([x,y])=>[x+(random()-.5)*1.4,y+(random()-.5)*1.4]);
  const points=sourcePoints.map(([x,y])=>[128+(x-128)*shapeTransform[0]+(y-128)*shapeTransform[2],128+(y-128)*shapeTransform[3]]);
  const paletteOffset=Math.floor(random()*6);
  return {...recipe,sourcePoints,points,shapeTransform,magic:{
    glint:random()*4-2,
    beans:[
      {x:75,y:89,rx:27,ry:40,angle:-.36},
      {x:129,y:67,rx:26,ry:35,angle:.42},
      {x:167,y:120,rx:28,ry:39,angle:-.15},
      {x:106,y:131,rx:28,ry:37,angle:.69},
      {x:78,y:178,rx:28,ry:36,angle:-.38},
      {x:162,y:184,rx:28,ry:39,angle:.53}
    ].map((bean,index)=>({...bean,x:bean.x+(random()-.5)*3,y:bean.y+(random()-.5)*3,
      angle:bean.angle+(random()-.5)*.12,palette:(index+paletteOffset)%6}))
  }};
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


function candyEllipse(context,x,y,rx,ry,fill,stroke,width=3,angle=0) {
  context.beginPath();context.ellipse(x,y,rx,ry,angle,0,Math.PI*2);
  context.fillStyle=fill;context.fill();
  if(stroke){context.strokeStyle=stroke;context.lineWidth=width;context.stroke();}
}

function drawChocolateFrog(context,recipe) {
  surface(context,recipe,'milk',true);
  context.save();pastryPath(context,recipe.sourcePoints);context.clip();
  // Moulded haunches, one outstretched foot, and an off-centre face make a
  // chocolate frog whose left and right sides are visibly different.
  const belly=context.createLinearGradient(97,115,160,197);
  belly.addColorStop(0,'#c28c65');belly.addColorStop(.55,'#a16b49');belly.addColorStop(1,'#79503b');
  candyEllipse(context,129,155,36,42,belly,'#5d392c',3,-.22);
  candyEllipse(context,151,167,20,29,'#976447','#553529',4,.2);
  drawStroke(context,[[144,192],[156,178],[162,160]],'#d6a279',4);
  drawStroke(context,[[94,147],[87,163],[64,177],[47,178]],'#543229',5);
  drawStroke(context,[[72,117],[57,122],[40,119]],'#523027',5);
  drawStroke(context,[[173,147],[191,145],[211,127]],'#583429',5);
  drawStroke(context,[[82,129],[92,140],[105,155]],'#e0ad7e',5);
  drawStroke(context,[[203,130],[211,123]],'#e7b883',4);
  drawStroke(context,[[206,138],[216,131]],'#e7b883',3);
  // Pale chocolate eyes stay large enough to read on a 64-pixel sweet.
  candyEllipse(context,83,73,15,17,'#f4dbac','#55332b',3,-.08);
  candyEllipse(context,133,77,14,15,'#f7dfb3','#55332b',3,.12);
  candyEllipse(context,85,76,6,8,'#40251e');
  candyEllipse(context,129,79,6,7,'#40251e');
  candyEllipse(context,83,73,2.3,2.3,'#fff8de');
  candyEllipse(context,127,76,2,2,'#fff8de');
  drawStroke(context,[[87,99],[107,119],[135,100]],'#47271f',6);
  drawStroke(context,[[94,106],[112,117],[128,109]],'#dfac79',2.5);
  candyEllipse(context,108,95,2.5,2.5,'#523027');
  candyEllipse(context,119,94,2.5,2.5,'#523027');
  // A single gold chocolate button is a second, simple reflection landmark.
  candyEllipse(context,116,158+recipe.magic.glint,8,8,'#f6d486','#7a4829',2);
  drawStroke(context,[[72,59],[84,55],[95,59]],'#e7b988',4);
  context.restore();
}

function mandrakeLeaf(context,base,tip,width,color) {
  const [x,y]=base,[tx,ty]=tip,dx=tx-x,dy=ty-y,length=Math.hypot(dx,dy);
  const nx=-dy/length*width,ny=dx/length*width;
  context.beginPath();context.moveTo(x,y);
  context.bezierCurveTo(x+dx*.22+nx,y+dy*.22+ny,tx+nx*.38,ty+ny*.38,tx,ty);
  context.bezierCurveTo(tx-nx*.6,ty-ny*.6,x+dx*.4-nx,y+dy*.4-ny,x,y);
  context.fillStyle=color;context.fill();context.strokeStyle='#3f642c';context.lineWidth=3;context.stroke();
  drawStroke(context,[[x,y],[x+dx*.58,y+dy*.58],[tx,ty]],'#b0d46a',3);
}

function drawMandrake(context,recipe) {
  context.save();context.shadowColor='#3f3d2840';context.shadowBlur=5;context.shadowOffsetY=4;
  mandrakeLeaf(context,[124,92],[65,30],22,'#78aa49');
  mandrakeLeaf(context,[125,91],[143,23],19,'#90b852');
  mandrakeLeaf(context,[130,94],[193,53],25,'#5f943d');
  context.restore();
  drawStroke(context,[[123,108],[126,78],[141,43]],'#557c33',8);
  surface(context,recipe,'bread',true);
  context.save();pastryPath(context,recipe.sourcePoints);context.clip();
  // Golden marzipan/biscuit root, with a face and uneven branching legs.
  const cheek=context.createRadialGradient(111,112,2,122,120,45);
  cheek.addColorStop(0,'#ffe9b6');cheek.addColorStop(1,'#f1c78000');
  candyEllipse(context,125,123,34,38,cheek);
  drawStroke(context,[[94,104],[104,98],[113,101]],'#714b2c',4);
  drawStroke(context,[[135,101],[145,104],[151,111]],'#714b2c',4);
  candyEllipse(context,106,115,6,8,'#4e3624');
  candyEllipse(context,142,117,5,7,'#4e3624');
  candyEllipse(context,104,112,1.8,2,'#fff5d7');
  candyEllipse(context,140,114,1.6,1.9,'#fff5d7');
  candyEllipse(context,125,137,12,16,'#60392b','#a76b39',2,-.14);
  candyEllipse(context,126,144,7,5,'#d99273',null,0,-.14);
  drawStroke(context,[[115,163],[106,184],[88,211]],'#a16b39',4);
  drawStroke(context,[[142,163],[153,182],[178,208]],'#a16b39',4);
  drawStroke(context,[[160,132],[177,139],[187,152]],'#b58043',4);
  drawStroke(context,[[92,136],[77,143],[64,145]],'#b58043',3);
  drawStroke(context,[[98,155],[107,159]],'#b78146',3);
  drawStroke(context,[[147,153],[156,151]],'#b78146',3);
  drawStroke(context,[[121,92],[116,98]],'#b78146',3);
  // One small sugar leaf on the left shoulder is deliberately unmatched.
  mandrakeLeaf(context,[93,127],[82,103],9,'#8eb35a');
  context.restore();
}

const BEAN_PALETTES = [
  ['#ffd0de','#ed6a9e','#b62f68','#90234f'],
  ['#eeffa9','#9dcd51','#508e3c','#3c692f'],
  ['#e7cbff','#aa79d4','#684490','#543474'],
  ['#ffe7a1','#efae48','#b96725','#8f511f'],
  ['#c4f7ff','#63bed9','#357baf','#2a648c'],
  ['#fff7df','#dac293','#a58b62','#816846']
];

function drawJellyBean(context,bean,index) {
  const [highlight,mid,shade,outline]=BEAN_PALETTES[bean.palette];
  context.save();context.translate(bean.x,bean.y);context.rotate(bean.angle);
  const {rx,ry}=bean;
  context.beginPath();context.moveTo(-rx*.72,-ry*.7);
  context.bezierCurveTo(-rx*1.32,-ry*.14,-rx*.87,ry*.86,-rx*.12,ry);
  context.bezierCurveTo(rx*.66,ry*1.04,rx*1.1,ry*.42,rx*.61,ry*.04);
  context.bezierCurveTo(rx*.23,-ry*.22,rx*.91,-ry*.57,rx*.5,-ry*.85);
  context.bezierCurveTo(rx*.16,-ry*1.1,-rx*.44,-ry*1.01,-rx*.72,-ry*.7);
  context.closePath();
  const glaze=context.createLinearGradient(-rx,-ry,rx,ry);
  glaze.addColorStop(0,highlight);glaze.addColorStop(.48,mid);glaze.addColorStop(1,shade);
  context.shadowColor='#41324155';context.shadowBlur=5;context.shadowOffsetY=4;
  context.fillStyle=glaze;context.fill();context.shadowColor='transparent';
  context.strokeStyle=outline;context.lineWidth=3.5;context.stroke();context.clip();
  drawStroke(context,[[-rx*.46,-ry*.66],[-rx*.64,-ry*.26],[-rx*.52,ry*.12]],'#ffffffb8',6);
  if(index===1||index===5)for(const [x,y,r] of [[-4,10,4],[10,-15,5],[-10,-16,3],[6,24,3]]){
    candyEllipse(context,x,y,r,r,bean.palette===5?'#88613caa':'#fff7d0bb');
  }
  if(index===3)drawStroke(context,[[-rx,ry*.15],[0,ry*.36],[rx,ry*.12]],'#fff3c9a8',6);
  context.restore();
}

function drawBroom(context,recipe) {
  // A bent chocolate pretzel handle joins a wide caramel brush. The hooked
  // handle, stepped bristle tips and unequal berry bow belong to the sweet;
  // their asymmetry survives both reflections even at conveyor-belt size.
  surface(context,recipe,'milk',true);
  context.save();pastryPath(context,recipe.sourcePoints);context.clip();
  drawStroke(context,[[82,37],[98,37],[101,57],[95,78],[108,110],[132,140]],'#ddb187',6);
  drawStroke(context,[[97,50],[102,51]],'#fff2c3',3);
  drawStroke(context,[[89,81],[99,79]],'#ffe6b2',4);
  drawStroke(context,[[98,106],[110,102]],'#ffe6b2',4);
  drawStroke(context,[[113,131],[124,123]],'#ffe6b2',4);
  pastryPath(context,[[115,139],[142,135],[167,146],[199,177],[222,205],[198,210],[185,226],[157,217],[140,233],[118,219],[98,225],[82,208],[95,179],[102,158]]);
  const caramel=context.createLinearGradient(112,145,176,224);
  caramel.addColorStop(0,'#fff0b0');caramel.addColorStop(.38,'#eebb57');caramel.addColorStop(1,'#b57026');
  context.fillStyle=caramel;context.fill();context.strokeStyle='#855025';context.lineWidth=4;context.stroke();
  const strands=[
    [[114,158],[109,181],[96,210]],
    [[122,157],[123,187],[121,217]],
    [[130,154],[143,188],[141,226]],
    [[138,153],[157,183],[157,212]],
    [[147,154],[173,184],[182,219]],
    [[153,152],[187,174],[211,202]]
  ];
  for(const path of strands) {
    drawStroke(context,path,'#955523',5);
    drawStroke(context,path.map(([x,y])=>[x-3,y-1]),'#ffe2a0',3);
  }
  drawStroke(context,[[105,161],[126,164],[155,151]],'#714331',17);
  drawStroke(context,[[105,157],[126,160],[155,147]],'#c3769f',11);
  drawStroke(context,[[106,154],[126,157],[154,144]],'#f4c3d3',3);
  // Three fixed sugar seeds and a single star vary in sheen with the seed.
  for(const [x,y,angle] of [[148,184,.3],[171,201,-.4],[132,201,.1]])
    candyEllipse(context,x,y+recipe.magic.glint,2.1,4.2,'#fff4ca',null,0,angle);
  magicStar(context,178,176,'#fff0b5',8);
  context.restore();
  context.save();context.shadowColor='#55344740';context.shadowBlur=3;context.shadowOffsetY=2;
  pastryPath(context,[[115,154],[89,132],[78,138],[84,159],[101,167]]);
  context.fillStyle='#b9618a';context.fill();context.strokeStyle='#754257';context.lineWidth=3;context.stroke();
  pastryPath(context,[[120,155],[137,152],[150,163],[140,172],[126,165]]);
  context.fillStyle='#d485ab';context.fill();context.stroke();
  context.shadowColor='transparent';
  drawStroke(context,[[88,142],[104,156]],'#f4c3d3',4);
  drawStroke(context,[[129,157],[141,164]],'#f5d0df',3);
  candyEllipse(context,118,157,8,9,'#e5a3c0','#754257',3,-.3);
  context.restore();
}

function snitchWing(context,points,veins,left) {
  pastryPath(context,points);
  const sugar=context.createLinearGradient(left?35:218,55,126,155);
  sugar.addColorStop(0,'#fffef1');sugar.addColorStop(.52,'#f6e9bb');sugar.addColorStop(1,'#c9a759');
  context.fillStyle=sugar;context.fill();context.lineWidth=3.5;context.strokeStyle='#9a782f';context.stroke();
  for(const path of veins){
    drawStroke(context,path,'#b19659',3.5);
    drawStroke(context,path.map(([x,y])=>[x-1.7,y-1]),'#fffdf0',1.8);
  }
}

function drawGoldenSnitch(context,recipe) {
  context.save();context.shadowColor='#69501e45';context.shadowBlur=5;context.shadowOffsetY=4;
  // One tall swept wing and one short outstretched wing prevent a mirrored
  // sphere from looking correct. The pale wings are moulded sugar wafers.
  snitchWing(context,[[103,139],[76,121],[55,98],[37,67],[24,34],[47,54],[55,31],[73,65],[83,53],[99,83],[110,113]],[
    [[99,126],[69,98],[39,52]],
    [[83,111],[71,80],[57,47]],
    [[95,113],[90,86],[83,67]],
    [[70,107],[60,95],[45,86]],
    [[97,130],[79,124],[66,116]]
  ],true);
  snitchWing(context,[[158,139],[186,104],[210,87],[231,76],[216,105],[232,99],[216,127],[229,128],[203,151],[174,153]],[
    [[169,137],[192,115],[219,87]],
    [[181,140],[204,124],[220,108]],
    [[187,146],[205,141],[218,133]],
    [[175,126],[190,115],[204,108]]
  ],false);
  context.restore();
  const gold=context.createRadialGradient(109,125,5,126,149,55);
  gold.addColorStop(0,'#fff5b2');gold.addColorStop(.28,'#f9d774');gold.addColorStop(.67,'#dea938');gold.addColorStop(1,'#9d6116');
  context.save();context.shadowColor='#68431955';context.shadowBlur=6;context.shadowOffsetY=5;
  candyEllipse(context,126,148,46,46,gold,'#886019',4);
  context.shadowColor='transparent';
  context.beginPath();context.arc(126,148,44,0,Math.PI*2);context.clip();
  // Chocolate mould grooves form unequal arcs and an off-centre raised stud.
  // They add the final stage's fine orientation cues without masking the orb.
  const grooves=[
    [[83,139],[105,146],[135,141],[165,119]],
    [[82,146],[105,154],[139,150],[172,128]],
    [[101,108],[111,125],[117,150],[112,188]],
    [[142,108],[151,129],[157,152],[151,187]],
    [[91,169],[119,174],[152,162],[173,148]]
  ];
  for(const path of grooves){
    drawStroke(context,path,'#a87223',4);
    drawStroke(context,path.map(([x,y])=>[x,y-2]),'#ffe49a',2);
  }
  candyEllipse(context,137,161+recipe.magic.glint,9,9,'#e8b444','#96651c',2);
  candyEllipse(context,135,159+recipe.magic.glint,3,3,'#fff0ad');
  drawStroke(context,[[97,127],[101,118],[113,114]],'#fff6cb',6);
  candyEllipse(context,131,117,2.5,2.5,'#fff8d1');
  for(const [x,y] of [[94,159],[102,164],[110,166]])candyEllipse(context,x,y,1.6,2.3,'#98631d');
  context.restore();
}

function drawMagicPastry(canvas,recipe) {
  const context=canvas.getContext('2d');
  context.save();context.translate(128,128);context.transform(...recipe.shapeTransform,0,0);context.translate(-128,-128);
  if(recipe.level===2)drawChocolateFrog(context,recipe);
  else if(recipe.level===4)drawMandrake(context,recipe);
  else if(recipe.level===6)recipe.magic.beans.forEach((bean,index)=>drawJellyBean(context,bean,index));
  else if(recipe.level===8)drawBroom(context,recipe);
  else drawGoldenSnitch(context,recipe);
  context.restore();
}

function drawThemedPastry(canvas,recipe) {
  if(recipe.magic){drawMagicPastry(canvas,recipe);return;}
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
