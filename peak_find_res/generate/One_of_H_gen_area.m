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
peak = zeros(1,length(t));
wnum = ceil(3 + rand(1)*10);  %峰的个数
Jnum = randi(1, 3, wnum);     %J的个数为1~3
w = round(1000+rand(1, wnum)*3000);   %峰的位置
J = 10*rand(wnum, 3)+2;       %J的范围为2~12
pow = randi(20, wnum, 3) - ones(wnum, 3);  %不同的耦合
for ww=1:wnum
    T2=0.1+0.15*rand(1);  %每个c峰T2的范围
    M00=0.6*rand(1,wnum); %每个峰的幅度系数cc
    tt00 = M00(ww)*exp(1i*(2*pi*(w(ww)))*t).*exp(-t/T2);
%     for z=1:Jnum  %ctrl+r/t
%         tt00 = tt00.*cos(pi*J(ww, z)*t).^pow(ww, z);
%     end
    top(1, :) = real(fft(tt00,np));
    top = top/max(max(top));
    area_1 = find(top>0.005);
    peak(area_1) = 1;
    signal = signal + tt00;
end

%生成峰的位置谱
label=zeros(1,np);
n = np/sw;
for i=1:wnum
   label(round(w(i)*n))=1; 
end
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
signalnoise=awgn(signal,10,'measured');%加噪
noise(1, :) = signalnoise; 
impure = real((fft(noise, np)))/max(max(real((fft(noise, np)))));
figure();
plot(freq,impure);%做频谱图
axis([-5,ss-5,-0.5,1]);

data_peak_high=struct('peak_high',label);
save('../test/peak_high.mat');
data_peak=struct('peak',peak);
save('../test/peak.mat');
data_pure=struct('pure',pure);
save('../test/pure.mat');
data_impure=struct('impure',impure);
save('../test/impure.mat');