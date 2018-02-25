clear
close
clc
load ref.dat;
load refd.dat;
load u.dat;
load ud.dat;
load utrue.dat;
load udtrue.dat;
load time.dat;
minplus = zeros(size(time));
minplus2 = minplus;
minplus3 = minplus;
minplus4 = minplus;
for i =1:numel(time)
    if sign(ud(i)) == sign(udtrue(i))
        minplus(i) = 1;
    else minplus(i) = 0;
    end
end
% figure(1)
% subplot(311)
% plot(time,ref,'b',time,u,'r',time,utrue,'g',...
%     time,ref+2e-8,'--b',time,ref-2e-8, '--b',...
%     time, ref+1e-8,'--b',time,ref-1e-8,'--b')
% title('displacement')
% legend('ref', 'read', 'true')
% grid on
% subplot(312)
% plot(time,refd,'b',time,ud,'r',time,udtrue,'g')
% grid on
% title('velocity')
% legend('ref', 'read', 'true')
% subplot(313)
% plot(time,minplus,'*')
% title('sign of velocity')
% ylabel('+1 concordi,-1 discordi')
% grid on

%%stime di velocità
f = 40000;
dt = 1/80000;
a = 0.75;
A = [a, 1-a; 0, 1];
C = [1,0];
G = [0;1];
Q = 1;%%%  mante ha usato 0.5
R = 1;
[L,P,Z,E] = dlqe(A,G, C, Q, R);
% K = place(A',C',[1e-3,1e-4,1e-5]);
% eig(A-K'*C)
x = zeros(2,numel(time));
for i = 1:size(time)-2
    x(:,i+1) = x(:,i+1)+L*(u(i+1)-x(1,i+1));
    x(:,i+2) = A*x(:,i+1);
end
utemp = x(1,:);
ufilt = x(2,:);


udfilt = zeros(size(time));
for i = 2:numel(time)
    udfilt(i) = (ufilt(i)-ufilt(i-1))/dt;
end



x = zeros(2,numel(time));
for i = 1:size(time)-2
    x(:,i+1) = x(:,i+1)+L*(udfilt(i+1)-x(1,i+1));
    x(:,i+2) = A*x(:,i+1);
end

udfilt2 = x(2,:);

figure()
plot(time,ud,'b',time,udfilt,'r',time, udtrue,'g', time,udfilt2,'k');
grid on

for i =1:numel(time)
    if sign(udfilt(i)) == sign(udtrue(i))
        minplus2(i) = 1;
    else minplus2(i) = 0;
    end
end

for i =1:numel(time)
    if sign(udfilt2(i)) == sign(udtrue(i))
        minplus3(i) = 1;
    else minplus3(i) = 0;
    end
end
std(ud)
std(udfilt)
std(udfilt2)

sum(minplus)
sum(minplus2)
sum(minplus3)

udfilt3 = zeros(size(time));
for i = 2:numel(time)
    udfilt3(i) = (utemp(i)-utemp(i-1))/dt;
end

for i =1:numel(time)
    if sign(udfilt3(i)) == sign(udtrue(i))
        minplus4(i) = 1;
    else minplus4(i) = 0;
    end
end


sum(minplus4)