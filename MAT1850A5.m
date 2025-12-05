% ----------------------- % PART A
f = @(x,y) (1.5-x+x*y)^2+(2.25-x+x*y^2)^2+(2.625-x+x*y^3)^2;
x = valder(1/2,[1 0]);
y = valder(1/4,[0 1]);
fvd = f(x,y);
[fvd.val,fvd.der]

% -------------------------------- % PART B 

u_prev = zeros(1,2);
alpha = 0.25;
beta = 0.8;
u_next = BackGD(alpha, beta, u_prev, f);
while norm(u_prev-u_next) > 10^-10
    u_prev = u_next;
    u_next = BackGD(alpha, beta, u_prev, f);
end
disp(u_prev)

% ----------------------------------------------% PART C
% Lagrange multipliers: minimize wrt primal x, maximize wrt dual lambda
% A = {[x1,y1,z1], [x2,y2,z2]...}, lambda = [d1,d2...d10]

alpha = 0.01; % set small alpha 
beta = 0.6;
gamma = 0.1;
% generate points on the outer Y radius of torus
R = 0.75; thet = linspace(0, 2*pi, 11); thet = thet(1:10);
x_prev = zeros(1,30);
x_prev(1:3:end) = R*cos(thet);
x_prev(3:3:end) = R*sin(thet);
x_prev(2:3:end) = 1e-4*randn(1,10); % perturb y because otherwise d/dy = 0

lambda_prev = zeros(1,10); %initial lambda=0 b.c. equality condition
x_next = BackGDPartial(alpha, beta, x_prev, lambda_prev, @lagrangian,1);
lambda_next = BackGDPartial(alpha, beta, x_next, lambda_prev, @lagrangian,2);
counter = 0;
while norm(x_prev-x_next) > 10^-5 || norm(lambda_prev-lambda_next) > 10^-5
    counter = counter+1;
    x_prev = x_next;
    x_next = BackGDPartial(alpha, beta, x_prev, lambda_next, @lagrangian,1);
    lambda_prev = lambda_next;
    lambda_next = BackGDPartial(alpha, beta, x_next, lambda_prev, @lagrangian,2);
    if mod(counter, 10) == 0
        % disp(["iteration, x diff, lambda diff", counter, norm(x_prev-x_next) norm(lambda_prev-lambda_next)]) 
        %compute lagrangian and max"g" constraint violations to track
        Lval = lagrangian(x_next, lambda_next);
        gmax = 0;
        for i = 0:9
            gval = sqrt((sqrt(x_next(3*i+1)^2 + x_next(3*i+3)^2)-0.5)^2 + x_next(3*i+2)^2) - 0.25;
            gmax = max(gmax, abs(gval));
        end
        fprintf('iter %d, ||dx||=%.2e, ||dl||=%.2e, L=%.6e, max|g|=%.2e\n', counter, norm(x_prev-x_next), norm(lambda_prev-lambda_next), Lval, gmax);
        drawnow;
    end
end
% final results, feed this to plotter file
disp(x_prev)
disp(lambda_prev)


function L = lagrangian(A, lambda) % A will be 3n length array (x1,y1,z1, x2,y2,z2..)
    n = length(lambda);
    E = 0;
    % compute E
    for j = 0:n-1
        for i = 0:j-1
            dist = (A(3*i+1) - A(3*j+1))^2 + (A(3*i+2) - A(3*j+2))^2 + (A(3*i+3) - A(3*j+3))^2;
            E = E + 1/sqrt(dist);
        end
    end
    F = 0;
    % create augmented lagrangian with penalty -- helps stabilize lambdas
    % more without x derailing initially
    FF = 0;
    for i = 0:n-1
        F = F + lambda(i+1)* (sqrt((sqrt(A(3*i+1)^2 + A(3*i+3)^2)-0.5)^2+A(3*i+2)^2)-0.25);
        FF = FF + (sqrt((sqrt(A(3*i+1)^2 + A(3*i+3)^2)-0.5)^2+A(3*i+2)^2)-0.25)^2;
    end

    L = E + F + 20*FF/2; % idk i just tried a larger penalty of 20 b.c. my lambda updates were small
end

function u = BackGDPartial(alpha, beta, v, lambda, f, mode) % f = f(v, lambda) 
    % mode 1 optimizes over [x1,y1,z1]....xn,yn,zn], mode 2 optimizes over
    % [d1...dn]
    if mode == 1 % gradient descent over primal variables
        t = 1;
        n = length(v);
        m = length(lambda);
        % "valderize" data in v
        var_c = cell(1, n);
        for i = 1:n
            onehot_i = zeros(1,n+m);
            onehot_i(i) = 1;
            var_c{i} = valder(v(i), onehot_i);
        end
        % valderize data in lambda
        lam_c = cell(1,m);
        for i = 1:m
            onehot_i = zeros(1,n+m);
            onehot_i(n+i) = 1;
            lam_c{i} = valder(lambda(i), onehot_i);
        end
        fvd = f([var_c{:}], [lam_c{:}]);
        grad = fvd.der(1:n); %only keep gradient of first variables
    
        while 1
            K = modalValderEmbedder(v,lambda,grad,t,1); % valderizes v-grad*t
            %disp([f([K{:}], [lam_c{:}]).val- f([var_c{:}], [lam_c{:}]).val - alpha*t*norm(grad).^2])
            if f([K{:}], [lam_c{:}]).val <= fvd.val - alpha*t*norm(grad).^2 || t < 10^-12
                break;
            end
            t = beta*t;
        end
        v = v - t*grad;
        u = v;
    elseif mode == 2 % gradient ascent of dual variables
        t = 1;
        n = length(v);
        m = length(lambda);
        var_c = cell(1, n);
        for i = 1:n
            onehot_i = zeros(1,n+m);
            onehot_i(i) = 1;
            var_c{i} = valder(v(i), onehot_i);
        end
        lam_c = cell(1,m);
        for i = 1:m
            onehot_i = zeros(1,n+m);
            onehot_i(n+i) = 1;
            lam_c{i} = valder(lambda(i), onehot_i);
        end
        fvd = f([var_c{:}], [lam_c{:}]);
        grad = fvd.der(n+1:end); % only keep gradients of last variables
        
        % added a damping factor of 0.1 to lambdas for stability issues
        while 1
            K = modalValderEmbedder(v,lambda,grad,t,2);
            % %disp(f([var_c{:}], [K{:}]).val - f([var_c{:}], [lam_c{:}]).val + alpha*t*norm(grad).^2)
            if f([var_c{:}], [K{:}]).val >= fvd.val + alpha*0.1*t*norm(grad).^2 || t < 10^-12
                break;
            end
            t = beta*t;
        end
        lambda = lambda + 0.1*t*grad;
        u = lambda;
    end
end

function embedding = modalValderEmbedder(v, lambda, grad, t, mode) % input grad will be grad of the variables u want to edit
    n = length(v);
    m = length(lambda);
    if mode == 1
        embedding = cell(1, n);
        for i = 1:n
            onehot_i = zeros(1,n+m);
            onehot_i(i) = 1;
            embedding{i} = valder(v(i) - t*grad(i), onehot_i);
        end
    elseif mode == 2
        embedding = cell(1, m);
        for i = 1:m
            onehot_i = zeros(1,n+m);
            onehot_i(n+i) = 1;
            embedding{i} = valder(lambda(i) + 0.1*t*grad(i), onehot_i);
        end
    end
end

function u = BackGD(alpha, beta, v, f) % f = f(v)
    t = 1;
    n = length(v);
    var_c = cell(1, n);
    for i = 1:n
        onehot_i = zeros(1,n);
        onehot_i(i) = 1;
        var_c{i} = valder(v(i), onehot_i);
    end
    fvd = f(var_c{:});
    grad = fvd.der;

    while 1
        K = valderEmbedder(v,grad,t);
        if f(K{:}).val <= f(var_c{:}).val - alpha*t*norm(grad).^2
            break;
        end
        t = beta*t;
    end
    v = v - t*grad;
    u = v;
end

function var_c = valderEmbedder(v, grad, t)
    n = length(v);
    var_c = cell(1, n);
    for i = 1:n
        onehot_i = zeros(1,n);
        onehot_i(i) = 1;
        var_c{i} = valder(v(i) - t*grad(i), onehot_i);
    end
end

% rewrite this for general n in R^n
% function u = BackGD(alpha, beta, x, y, f) % f = f(v)
%     t = 1;
%     xc = valder(x, [1 0]);
%     yc = valder(y, [0 1]);
%     fvd = f(xc,yc);
%     grad = fvd.der;
% 
%     while 1
%         %K = valderEmbedder([xc, yc],grad,t);
%         if f(valder(x-t*grad(1),[1 0]), valder(y-t*grad(2),[0 1])).val <= f(xc,yc).val - alpha*t*norm(grad).^2
%             break;
%         end
%         t = beta*t;
%     end
%     u = [x y] - t*grad;
% end

% failed parameterization
% R = 0.5; r = 0.25;
% [Theta, Phi] = meshgrid(linspace(0,2*pi,50));
% Xt = (R + r*cos(Phi)).*cos(Theta);
% Yt = r*sin(Phi);
% Zt = (R + r*cos(Phi)).*sin(Theta);
% C = floor(linspace(1, 50, 10));
% B = zeros(1,30);
% for i = 1:10
%     B(3*(i-1)+1:3*(i-1)+3) = [Xt(C(i),C(i)), Yt(C(i),C(i)), Zt(C(i),C(i))];
% end
% x_prev = B;