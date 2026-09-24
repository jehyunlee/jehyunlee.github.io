import * as THREE from './vendor/three.module.min.js';

export const BALL_RADIUS=1.02;
// Bounds in the supplied 427 × 346 original. Everything below y=200 is excluded.
export const LOGO_CROP={x:137,y:24,width:170,height:176};
const rollAxis=new THREE.Vector3(),rollRotation=new THREE.Quaternion();

// Chuseok 2026 special: the ball becomes a full moon from 2026-09-24 00:00 KST
// through 2026-09-27 24:00 KST. Fixed UTC epochs, so the window does not depend
// on the visitor's time zone. `?moon=1` / `?moon=0` force it for previewing.
export const MOON_EVENT={start:Date.UTC(2026,8,23,15),end:Date.UTC(2026,8,27,15)};
export const MOON_ASSETS={color:'./moon-color.jpg?v=20260924-moon',bump:'./moon-bump.jpg?v=20260924-moon'};
// Where the chalk graffiti was baked into the equirectangular map (pixels of 2048×1024).
export const MOON_GRAFFITI={x:512,y:392,size:600};

export function moonEventActive(now=Date.now(),search=globalThis.location?.search??'') {
  const forced=new URLSearchParams(search).get('moon');
  if(forced==='1')return true;
  if(forced==='0')return false;
  return now>=MOON_EVENT.start&&now<MOON_EVENT.end;
}

export async function loadMoonTextures(anisotropy=1) {
  const loader=new THREE.TextureLoader();
  const [map,bumpMap]=await Promise.all([loader.loadAsync(MOON_ASSETS.color),loader.loadAsync(MOON_ASSETS.bump)]);
  map.colorSpace=THREE.SRGBColorSpace;map.anisotropy=anisotropy;bumpMap.anisotropy=anisotropy;
  return {map,bumpMap};
}

export function createMoonMaterial({map,bumpMap}) {
  // Regolith: fully rough, no clearcoat. LOLA elevation drives crater relief;
  // a faint self-glow keeps the disc reading as a full moon inside the dark maze.
  return new THREE.MeshStandardMaterial({color:0xcfcfcf,map,bumpMap,bumpScale:.09,roughness:1,metalness:0,emissive:0xffffff,emissiveMap:map,emissiveIntensity:.12,envMapIntensity:.2});
}

export function drawMoonBadge(context,image,x,y,size) {
  const g=MOON_GRAFFITI,r=size/2;
  context.save();
  context.beginPath();context.arc(x+r,y+r,r,0,Math.PI*2);context.clip();
  context.drawImage(image,g.x-g.size/2,g.y-g.size/2,g.size,g.size,x,y,size,size);
  context.restore();
}

export function drawBallLogo(context,image,x,y,width,height,crop=LOGO_CROP) {
  crop=crop??{x:0,y:0,width:image.naturalWidth||image.width,height:image.naturalHeight||image.height};
  context.drawImage(image,crop.x,crop.y,crop.width,crop.height,x,y,width,height);
}

export function createBallTexture(image,anisotropy=1,crop=LOGO_CROP) {
  const canvas=document.createElement('canvas');canvas.width=2048;canvas.height=1024;
  const context=canvas.getContext('2d');
  context.fillStyle='#ffffff';context.fillRect(0,0,canvas.width,canvas.height);
  // A genuine UV map: the original emblem follows the surface and rolls with it.
  // Equal angular pixel density keeps its proportions at the equator.
  const source=crop??{width:image.naturalWidth||image.width,height:image.naturalHeight||image.height};
  const width=480,height=width*source.height/source.width;
  drawBallLogo(context,image,512-width/2,512-height/2,width,height,crop);
  const texture=new THREE.CanvasTexture(canvas);
  texture.colorSpace=THREE.SRGBColorSpace;texture.anisotropy=anisotropy;
  return texture;
}

export function resetBallOrientation(ball,yaw) {
  // u=.25 faces +Z. Tilt toward the 45° rear camera, then rotate into its heading.
  ball.quaternion.setFromEuler(new THREE.Euler(-Math.PI/4,yaw,0,'YXZ'));
}

export function rollBall(ball,from,to) {
  const dx=to.x-from.x,dz=to.z-from.z,distance=Math.hypot(dx,dz);
  if(distance<1e-9)return;
  rollAxis.set(dz/distance,0,-dx/distance);
  rollRotation.setFromAxisAngle(rollAxis,distance/BALL_RADIUS);
  ball.quaternion.premultiply(rollRotation).normalize();
}
