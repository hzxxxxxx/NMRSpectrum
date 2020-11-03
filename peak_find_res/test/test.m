clear all,clc; close all;
load pure.mat;
b=pure;
load impure.mat;
a=impure;
load generate.mat;
c=generate;
pure=pure/max(max(pure));
impure=impure/max(max(impure));
generate=generate/max(max(generate));
figure()
subplot(3,1,1)
plot(b)
xlabel('位置')
ylabel('强度')
title('纯净谱')
axis([0,8096,-0.5,1.5]); 
subplot(3,1,2)
plot(a)
xlabel('位置')
ylabel('强度')
title('含噪谱')
axis([0,8096,-0.5,1.5]); 
subplot(3,1,3)
plot(c)
xlabel('位置')
ylabel('出现峰的可能性强度')
title('峰提取位置')
axis([0,8096,-0.5,1.5]); 
snr_a=1/std(pure(1:350));
snr_b=1/std(impure(1:350));
snr_c=1/std(generate(1:350));
mse=generate-pure;
MSE=sum(mse.^2)/sum(pure.^2)*100;
fprintf('the value of snr_a is %6.2f\n',snr_a)
fprintf('the value of snr_b is %6.2f\n',snr_b)
fprintf('the value of snr_c is %6.2f\n',snr_c)
fprintf('the value of MSE is %.6f\n',MSE)