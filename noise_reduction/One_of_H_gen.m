close all
clear
clc

sw = 5000;                        %谱宽
AI= ceil(4*rand(1));
np = 2048;                        %采样点数
t = 0:1/sw:(np-1)/sw;
signal = zeros(1,length(t));
tt = ones(1, length(t));

wnum = ceil(3 + rand(1)*10);  %峰的个数
Jnum = randi(1, 3, wnum);     %J的个数为1~3
w = 1000+rand(1, wnum)*3000;   %峰的位置
J = 10*rand(wnum, 3)+2;       %J的范围为2~12
pow = randi(20, wnum, 3) - ones(wnum, 3);  %不同的耦合
signal00 = zeros(1,length(t));
for ww=1:wnum
    T2=0.1+0.15*rand(1);  %每个峰T2的范围
    M00=0.6*rand(1,wnum); %每个峰的幅度系数
    tt00 = M00(ww)*exp(1i*(2*pi*(w(ww)))*t).*exp(-t/T2);
    for z=1:Jnum
        tt00 = tt00.*cos(pi*J(ww, z)*t).^pow(ww, z);
    end
    signal00 = signal00 + tt00;
end

signal01=awgn(signal00,2,'measured');
signal02=awgn(signal00,4,'measured');
signal03=awgn(signal00,8,'measured');
signal04=awgn(signal00,16,'measured');
pure_real1(1, :) = real(fft(signal00,np));

noise01(1, :) = signal01;
noise02(1, :) = signal02;
noise03(1, :) = signal03;
noise04(1, :) = signal04;      

pure1 = pure_real1/max(max(pure_real1));

figure();
plot(pure1);%做频谱图
axis([0,np,-0.5,1]);
pure=pure1;


impure01 = real((fft(noise01, np)))/max(max(real((fft(noise01, np)))));
impure02 = real((fft(noise02, np)))/max(max(real((fft(noise02, np)))));
impure03 = real((fft(noise03, np)))/max(max(real((fft(noise03, np)))));
impure04 = real((fft(noise04, np)))/max(max(real((fft(noise04, np)))));



impure1=[impure01
         impure02
         impure03
         impure04];

figure();
plot(impure01);%做频谱图
axis([0,np,-0.5,1]);
impure=impure01;

data=struct('data_x',pure);
save('test/pure.mat','data');
data=struct('data_y',impure);
save('test/impure.mat','data');