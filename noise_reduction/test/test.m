clear all,clc
load pure.mat;
load impure.mat;
load generate.mat;
a=impure;
b=pure;
c=generate;
pure=pure/max(max(pure));
impure=impure/max(max(impure));
generate=generate/max(max(generate));
figure()
plot(a)
axis([0,2025,-0.5,1.5]); 
figure()
plot(b)
axis([0,2025,-0.5,1.5]); 
figure()
plot(c)
axis([0,2025,-0.5,1.5]); 
snr_a=1/std(pure(1:350));
snr_b=1/std(impure(1:350));
snr_c=1/std(generate(1:350));
mse=generate-pure;
MSE=sum(mse.^2)/sum(pure.^2)*100;
fprintf('the value of snr_a is %6.2f\n',snr_a)
fprintf('the value of snr_b is %6.2f\n',snr_b)
fprintf('the value of snr_c is %6.2f\n',snr_c)
fprintf('the value of MSE is %.6f\n',MSE)