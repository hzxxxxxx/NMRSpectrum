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

for a=1:100
signal = zeros(1,length(t));
peak = zeros(1,length(t));    %用来标记峰的范围
wnum = ceil(3 + rand(1)*10);  %峰的个数
Jnum = randi(1, 3, wnum);     %J的个数为1~3
w = round(1000+rand(1, wnum)*3000);   %峰的位置
for ww=1:wnum
    T2=0.1+0.15*rand(1);  %每个c峰T2的范围
    M00=0.6*rand(1,wnum); %每个峰的幅度系数cc
    tt00 = M00(ww)*exp(1i*(2*pi*(w(ww)))*t).*exp(-t/T2);
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

%生成纯净波谱
pure_real1(1, :) = real(fft(signal,np));
pure = pure_real1/max(max(pure_real1));

%生成不纯净波谱
signalnoise=awgn(signal,100,'measured');%加噪
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
save('../test/data_peak.mat','data');
data=struct('peak_area',total_spec_P);
save('../test/data_peakarea.mat','data');
data=struct('data_x',total_spec_X);
save('../test/data_pure.mat','data');
data=struct('data_y',total_spec_Y);
save('../test/data_impure.mat','data');