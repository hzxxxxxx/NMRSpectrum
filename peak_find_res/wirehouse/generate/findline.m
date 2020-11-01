clc;clear all;
load pure.mat;
x=data.data_x;
figure;plot(x);
n_x=length(x);
f_x=[1:n_x];
% [x0,y0]=invinterp(f_x,x,0.5);
[x0,y0]=invinterp(f_x(520:540),x(520:540),0.5*max(x(520:540)));
lw_x=abs(x0(1)-x0(2))*2000/2048;