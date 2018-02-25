clear
close
clc
load ref2.dat;
load refd2.dat;
load utemp.dat;
load udtemp.dat;
load utruetemp.dat;
load udtruetemp.dat;
load timetemp.dat;
load time2.dat;
% minplus = zeros(size(time));
% minplus2 = minplus;
% minplus3 = minplus;
% minplus4 = minplus;
% for i =1:numel(time)
%     if sign(ud(i)) == sign(udtrue(i))
%         minplus(i) = 1;
%     else minplus(i) = 0;
%     end
% end
figure(1)
subplot(211)
plot(time2,ref2(:,1),'b',timetemp,utemp(:,1),'r',timetemp,utruetemp(:,1),'g',...
    time2,ref2(:,1)+2e-8,'--b',time2,ref2(:,1)-2e-8, '--b',...
    time2, ref2(:,1)+1e-8,'--b',time2,ref2(:,1)-1e-8,'--b')
title('displacement')
legend('ref', 'read', 'true')
grid on
subplot(212)
plot(time2,refd2(:,1),'b',timetemp,udtemp(:,1),'r',timetemp,udtruetemp(:,1),'g')
grid on
title('velocity')
legend('ref', 'read', 'true')