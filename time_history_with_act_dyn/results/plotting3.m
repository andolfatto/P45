%% questi dati si riferiscono a csi = 1e-3 a 1000Hz, con
%ni = 5, gp = 15k,gd = 15, filtro sulla posizione a 16kHz
% clear
% close
% clc
load ref3.dat;
load refd3.dat;
load u3.dat;
load ud3.dat;
load utrue3.dat;
load udtrue3.dat;
load time3.dat;
minplus = zeros(size(time3));
% minplus2 = minplus;
% minplus3 = minplus;
% minplus4 = minplus;
for i =1:numel(time3)
    if sign(ud3(i)) == sign(udtrue3(i))
        minplus(i) = 1;
    else minplus(i) = 0;
    end
end
figure(3)
subplot(211)
plot(time3,ref3,'b',time3,u3,'r',time3,utrue3,'g',...
    time3,ref3+2e-8,'--b',time3,ref3-2e-8, '--b',...
    time3, ref3+1e-8,'--b',time3,ref3-1e-8,'--b')
title('displacement')
legend('ref', 'read', 'true')
grid on
subplot(212)
plot(time3,refd3,'b',time3,ud3,'r',time3,udtrue3,'g')
grid on
title('velocity')
legend('ref', 'read', 'true')