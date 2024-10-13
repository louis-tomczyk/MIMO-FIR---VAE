rst


% Step 1: check Taylor series goodness of approximation
% of 1/(1-x) at the first order 1+x
% set the confidence level alpha (eg. alpha = 5%) and get
% the {x} such that it is within the confidence level
Nx      = 1000;
x       = linspace(-1,1,Nx);
y       = 1./(1-x);
yapprox = 1+x;
dy      = abs((y-yapprox)./y)*100;
alpha   = 10;

figure
hold on
plot(x,dy)
plot(x,ones(1,Nx)*alpha)
set(gca,'Yscale','log')
set(gcf, 'Position', [0.0198, 0.0009, 0.5255, 0.8824])


% Step 2: check when dtheta*tan(theta)<<1
% take the max of the {x} of Step 1: xstar_max
xstar_max   = 0.225;
Nt          = 720;
t           = linspace(-pi,pi,Nt);
dt          = linspace(-pi,pi,Nt)/2;

[T,DT]      = meshgrid(t,dt);
Z           = abs(DT.*tan(t));

figure
hold on
surf(T*180/pi,DT*180/pi,Z)
xlabel('$\theta$ [deg]')
ylabel('$d\theta$ [deg]')
colormap("gray")
shading flat
set(gca,"Zscale",'log')
zlim([0,xstar_max])
title('validity domain of $d\theta\tan(\theta)<0.225$')
set(gcf, 'Position', [0.0198, 0.0009, 0.5255, 0.8824])

