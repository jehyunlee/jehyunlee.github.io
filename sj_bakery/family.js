// Atlas columns follow the positions in the supplied family portrait.
export const PLAYERS=[
 {id:'dad',name:'아빠',as:'아빠로',column:0,height:278},
 {id:'mom',name:'엄마',as:'엄마로',column:1,height:272},
 {id:'suan',name:'수안',as:'수안으로',column:2,height:253},
 {id:'jeongan',name:'정안',as:'정안으로',column:3,height:246}
];
export const POSES={neutral:0,happy:1,crying:2,eating:3};
export const getPlayer=id=>PLAYERS.find(player=>player.id===id)??null;
export const EATING_MOUTHS=[{x:.577,y:.492},{x:.540,y:.475},{x:.525,y:.468},{x:.556,y:.456}];
export const SPRITE_RECTS=[
 [92,12,309,355],[387,23,615,358],[682,43,897,358],[988,66,1190,358],
 [76,359,327,692],[383,368,621,693],[670,381,905,694],[988,398,1196,694],
 [90,694,298,992],[386,701,606,995],[688,711,895,995],[993,730,1186,995],
 [110,989,266,1251],[404,996,593,1253],[700,1003,877,1253],[1002,1014,1162,1253]
];

// Render the atlas's neutral checker matte as transparency. Only matte connected
// to a cell edge is keyed, so enclosed white eyeglass lenses stay opaque.
export function keySpriteMatte(data,width,height){
 const seen=new Uint8Array(width*height),queue=new Uint32Array(width*height);let head=0,tail=0;
 const offer=i=>{if(i<0||i>=seen.length||seen[i])return;seen[i]=1;const p=i*4,r=data[p],g=data[p+1],b=data[p+2];if(Math.min(r,g,b)>=155&&Math.max(r,g,b)-Math.min(r,g,b)<=27)queue[tail++]=i;};
 for(let x=0;x<width;x++){offer(x);offer((height-1)*width+x);}for(let y=0;y<height;y++){offer(y*width);offer(y*width+width-1);}
 while(head<tail){const i=queue[head++];data[i*4+3]=0;const x=i%width;if(x>0)offer(i-1);if(x<width-1)offer(i+1);if(i>=width)offer(i-width);if(i<width*(height-1))offer(i+width);}
 return data;
}
