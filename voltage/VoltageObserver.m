clear 
close all
clc
L = 3.6829e-4;
R = 2.428;
Ki = 2.025;
Rp = 20*R;

a = [-(R+Rp)/L -Ki/L;
     0      0   ];
b = [1/L, 0 ; 0, 1];
c = [1,0];
d = 0;
 
q = [(5e-5)^2,   0 ;
         0   ,    460^2];
% q = [(1e-6)^2,   0 ;
%          0   ,    10^2 ];
         
r = 5e-5^2;

dt = 1/80000;

sys = c2d(ss(a,b,c,d),dt);

[M,P,Z,E] = dlqe(sys.a,sys.b,sys.c,q,r);

%log(E)*80000/6.28;

%E,sqrt(Z);

load u.dat
load ud.dat
load vtrue.dat
load itrue.dat
load udtrue.dat

x = zeros(2,size(u,1));

for i = 1:size(u,1)-1
 x(:,i) =x(:,i)+M*(itrue(i,2)-x(1,i));
 x(:,i+1)=sys.a*x(:,i)+sys.b(1,1)*(vtrue(i,2));
end

figure()
plot(ud(:,1),ud(:,2),udtrue(:,1),udtrue(:,2))
hold on
plot(ud(:,1),x(2,:),'r')

