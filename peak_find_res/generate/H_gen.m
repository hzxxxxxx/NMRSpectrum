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

tt = ones(1, length(t));
spec_X=[];
spec_Y=[];
spec_L=[];
spec_P=[];
total_spec_X=[];
total_spec_Y=[];
total_spec_L=[];
total_spec_P=[];

for a=1:3000
signal = zeros(1,length(t));
peak = zeros(1,length(t));    %用来标记峰的范围
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

%生成纯净波谱
pure_real1(1, :) = real(fft(signal,np));
pure = pure_real1/max(max(pure_real1));

%生成不纯净波谱
signalnoise=awgn(signal,3,'measured');%加噪
noise(1, :) = signalnoise; 
impure = real((fft(noise, np)))/max(max(real((fft(noise, np)))));

 spec_X=[spec_X;impure];  
 spec_Y=[spec_Y;pure];
 spec_L=[spec_L;label];
 spec_P=[spec_P;peak];
 if rem(c1, 100) == 0
    total_spec_X=[total_spec_X;spec_X]; 
    total_spec_Y=[total_spec_Y;spec_Y]; 
    total_spec_L=[total_spec_L;spec_L]; 
    total_spec_P=[total_spec_P;spec_P]; 
    spec_X=[];
    spec_Y=[];
    spec_L=[];
    spec_P=[];
 end
c1=c1+1;
end

data=struct('peak',total_spec_L);
save('../data/data_peak.mat','data');
data=struct('peak_area',total_spec_P);
save('../data/data_peakarea.mat','data');
data=struct('data_x',total_spec_X);
save('../data/data_pure.mat','data');
data=struct('data_y',total_spec_Y);
save('../data/data_impure.mat','data');