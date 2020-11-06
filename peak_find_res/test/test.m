clear all,clc; close all;
load pure.mat;
load impure.mat;
load generate.mat;
load peak_high.mat;
pure=pure/max(max(pure));
impure=impure/max(max(impure));
generate=generate/max(max(generate));
label=label/max(max(label));
np = 8096;
figure()
subplot(4,1,1)
plot(pure)
xlabel('位置')
ylabel('强度')
title('纯净谱')
axis([0,np,-0.5,1.5]); 
subplot(4,1,2)
plot(impure)
xlabel('位置')
ylabel('强度')
title('含噪谱')
axis([0,np,-0.5,1.5]); 
subplot(4,1,3)
plot(generate)
xlabel('位置')
ylabel('出现峰的可能性强度')
title('峰提取位置')
axis([0,np,-0.5,1.5]); 
subplot(4,1,4)
plot(label)
xlabel('位置')
ylabel('出现峰的可能性强度')
title('峰标准位置')
axis([0,np,-0.5,1.5]); 
peak_total=length(find(label));
mse=generate-label;
error_label_ex=find((mse>0));
fprintf('the errorPeak of error_label_ex is %6.2f\n',error_label_ex)
error_label_lack=find((mse<0));
fprintf('the errorPeak of error_label_lack is %6.2f\n',error_label_lack)
acc=1-(length(error_label_lack)/peak_total);
fprintf('the accuracy of peak_find is%6.2f\n',acc)