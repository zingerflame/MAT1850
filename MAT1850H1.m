A = 101^2 * (diag(2*ones(100,1)) + diag(-1*ones(99,1), 1) + diag(-1*ones(99,1),-1));
b = ones(1, 100);

true_y = zeros(100,1);
for k = 1:100
    true_y(k) = (k/101)*(1-(k/101))/2;
end
[x, errormatrix, iteration] = basicIterativeMethod(A,b, true_y,1);
[x2, errormatrix2, iteration2] = basicIterativeMethod(A,b, true_y, 2);
semilogy(1:iteration, errormatrix, 'LineWidth', 1.6, 'DisplayName','Jacobi');
hold on;
semilogy(1:iteration2, errormatrix2, 'LineWidth', 1.6, 'DisplayName','GaussSeidel');
hold on;
xlabel('iteration');
hold on;
ylabel('log error');
hold on;
grid on;

function [x, errormatrix, iteration] = basicIterativeMethod(A,b, true_y, meth)
    n = size(b,2);
    x = zeros(n,1); % initialization
    maxIterations = 1000;
    tolerance = 1e-8;
    %meth = 1; %1 = jacobi, 2 = G-S

    done = false;
    iteration = 0;
    error = 1e100;
    while ~done
        iteration = iteration + 1;
        xprev = x;
        prev_error = error;

        for i = 1:n
            indBehind = 1:i-1;
            indAhead = i+1:n;
            if meth==1
                x(i) = (b(i) - A(i,[indBehind,indAhead])*xprev([indBehind,indAhead]))/A(i,i);
            elseif meth==2
                x(i) = (b(i) - A(i,indBehind)*x(indBehind) - A(i, indAhead)*xprev(indAhead))/A(i,i);
            end
        end
        % convergence check
        % done = norm(x-xprev, Inf) < tolerance || iteration > maxIterations;
        error = max(abs(x-true_y));
        errormatrix(iteration) = error;
        fprintf('Iteration %d has error %d \n', iteration, error)
        done = error >= prev_error;
    end
end
