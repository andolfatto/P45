clear
close all
clc
noise_A1 = 0.0;
noise_A2 = 3.3e-5;
noise_A3 = 2.9e-4;
noise_B1 = 1.1;
noise_B2 = 0.41;
noise_B3 = -14.55;
v_lim1 = 30000;
v_lim2 = 58000;
x1 = 0:v_lim1;
x2 = v_lim1+1:v_lim2;
x3 = v_lim2+1:2^16;
x = 0:2^16;
figure
hold on
grid on
plot(x1,noise_A1*x1+noise_B1)
plot(x2,noise_A2*x2+noise_B2)
plot(x3,noise_A3*x3+noise_B3)
plot(x,2.78e-24*x.^5+noise_B1)