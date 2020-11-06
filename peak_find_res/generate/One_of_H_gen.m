close all
clear
clc
sw = 5000;                        %谱宽
ss = 20;
c1 = 1;
AI= ceil(4*rand(1));
np = 8096;                        %采样点数
t = 0:1/sw:(np-1)/sw;
freq=[(ss/(2*np))-5:ss/np:ss-5];
signal = zeros(1,length(t));

wnum = ceil(3 + rand(1)*5);  %峰的个数
w = round(1000+rand(1, wnum)*3000);   %峰的位置
Jnum = randi([0, 3], 1, wnum);     %J的个数为0~3
label=zeros(1,np); %生成峰的位置谱
n = np/sw;

for ww=1:wnum
    T2=0.1+0.15*rand(1);  %每个峰T2的范围
    H=0.8*rand(1,wnum)+0.2; %每个峰的幅度系数
    label(round(w(ww)*n))=1; 
    fid1 = H(ww)*exp(1i*(2*pi*(w(ww)))*t).*exp(-t/T2);
    signal = signal + fid1;
    for j = 1:Jnum
        location = (w(ww)-(randsrc(1,1)*((randi([0,100],1,1)/20000)+0.002))*np);
        label(round(location*n))=1;
        fid2 = (H(ww)*((rand(1,1)/5)+0.8))*exp(1i*(2*pi*location)*t).*exp(-t/T2);%模拟耦合的峰
    signal = signal + fid2;
    end
end

Pnum = randi([0, 3], 1, 1);     %伪峰的个数为0~3
pw = round(rand(1, Pnum)*np);   %伪峰的位置

%给谱图添加上伪峰
for p=1:Pnum
    T2=0.05*rand(1);  %每个伪峰T2的范围
    H=0.4*rand(1,Pnum)+0.3; %每个峰的幅度系数
    fidp = H(p)*exp(1i*(2*pi*(pw(p)))*t).*exp(-t/T2);
    signal = signal + fidp;
end    

%输出峰位置标签图
figure();
plot(freq,label);
axis([-5,ss-5,-0.5,1]);

%生成纯净波谱
pure_real1(1, :) = real(fft(signal,np));
pure = pure_real1/max(max(pure_real1));
figure();
plot(freq,pure);%做频谱图
axis([-5,ss-5,-0.5,1]);

%生成不纯净波谱
signalnoise=awgn(signal,2,'measured');%加噪
noise(1, :) = signalnoise; 
impure = real((fft(noise, np)))/max(max(real((fft(noise, np)))));
figure();
plot(freq,impure);%做频谱图
axis([-5,ss-5,-0.5,1]);

data_peak_high=struct('peak_high',label);
save('../test/peak_high.mat');
data_pure=struct('pure',pure);
save('../test/pure.mat');
data_impure=struct('impure',impure);
save('../test/impure.mat');