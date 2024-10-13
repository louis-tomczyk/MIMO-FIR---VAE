rst

Nt = 50;
Ny = 50;
t = linspace(-pi,pi,Nt);
y = linspace(-1,1,Ny);

[T,Y] = meshgrid(t,y);
Z = sin(T).^2-3*cos(T).*Y;

Z0 = zeros(Nt,Ny);



figure('Position', [0.0198, 0.0009, 0.5255, 0.8824])
hold on
surf(T*180/pi,Y,Z)
surf(T*180/pi,Y,Z0)
xlabel('$\theta$ [deg]')
ylabel('$y = \cos(\theta+d\theta)-\cos(\theta)$')
zlabel('$\Delta =\sin(\theta)^2-3\cos(\theta)*y$')
colormap("jet")
view(45,45)
legend('$\Delta(\theta,y)$','Z = 0')