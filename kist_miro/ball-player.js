import * as THREE from './vendor/three.module.min.js';

export const BALL_RADIUS=1.02;
// Bounds in the supplied 427 × 346 original. Everything below y=200 is excluded.
export const LOGO_CROP={x:137,y:24,width:170,height:176};
const rollAxis=new THREE.Vector3(),rollRotation=new THREE.Quaternion();

export function drawAnniversaryLogo(context,image,x,y,width,height) {
  const crop=LOGO_CROP;
  context.drawImage(image,crop.x,crop.y,crop.width,crop.height,x,y,width,height);
}

export function createBallTexture(image,anisotropy=1) {
  const canvas=document.createElement('canvas');canvas.width=2048;canvas.height=1024;
  const context=canvas.getContext('2d');
  context.fillStyle='#ffffff';context.fillRect(0,0,canvas.width,canvas.height);
  // A genuine UV map: the original emblem follows the surface and rolls with it.
  // Equal angular pixel density keeps its proportions at the equator.
  const width=480,height=width*LOGO_CROP.height/LOGO_CROP.width;
  drawAnniversaryLogo(context,image,512-width/2,512-height/2,width,height);
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
