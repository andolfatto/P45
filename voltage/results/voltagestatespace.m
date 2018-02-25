%%votage tension state system
clear
close all
clc
L = 3.6829e-4;
R = 2.428;
Km = -2.026;
stdr1 = 5e-5;%%50muA
stdr2 = 7e-9;%%7nm
stdw1 = 1e-5;%%10muV
stdw2 = 10;

A = [-R/L, 0, -Km/L;
     0,    0,     1;
     0,    0,     0];
B = [1/L, 0;
     0,   0;
     0,   1];
C = [1, 0, 0;
     0, 1, 0];
D = [1, 0;
     0, 1];
SYS = ss(A,B,C,D);
dSYS = c2d(SYS, 1/80000);

Q = [stdw1^2 , 0;
     0,        stdw2^2];
R = [stdr1^2 , 0;
     0,        stdr2^2];
 
[L,P,E]= dlqe(dSYS.a,dSYS.b,dSYS.c,Q,R);


 
 