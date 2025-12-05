% parameterize torus and plot
R = 0.5; r = 0.25;
[t, phi] = meshgrid(linspace(0,2*pi,50));
Xp = (R + r*cos(phi)).*cos(t);
Yp = r*sin(phi);
Zp = (R + r*cos(phi)).*sin(t);
figure;
surf(Xp, Yp, Zp,'FaceAlpha', 0.4, 'LineStyle',':'); hold on
axis equal; grid on
xlabel('x'); ylabel('y'); zlabel('z');
view(40, 25);
pts = reshape(x_prev, 3, []).';   % 10×3 matrix of points
X = pts(:,1);
Y = pts(:,2);
Z = pts(:,3);

% plot on existing figure
plot3(X, Y, Z, 'ro', 'MarkerSize', 8, 'LineWidth', 2);