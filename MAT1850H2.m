A = [2,8,5;3,9,6;4,7,1];
[Q, R] = QRdecomp(A);

function [Q,R] = QRdecomp(A)
    n = size(A,1);
    Q = eye(n);
    for l = 1:n-1
        for k = n:-1:l+1
            [max_val, max_idx] = max(A(l:k-1,l));
            max_idx = max_idx+l-1; % offset the index to the actual row index in A
            [temp_a, temp_b, temp_c, temp_d] = deal(zeros(n,n), zeros(n,n), zeros(n,n), zeros(n,n)); 
            % make temp matrices for constructing G
            if max_val == 0
                [temp_a(k,k), temp_b(1,1), temp_c(1,k), temp_d(k,1)] = deal(-1, -1, 1, -1);
            else
                theta = atan(A(k,l)/A(max_idx,l));
                [temp_a(k,k), temp_b(max_idx,max_idx), temp_c(max_idx,k), temp_d(k,max_idx)] = deal(-1+cos(theta), -1+cos(theta), sin(theta), -sin(theta));
            end
            G = eye(n) + temp_a + temp_b + temp_c + temp_d;
            disp(G) % to get sequence
            A = G*A;
            disp(A)
            Q = Q * G.';
        end
    end
    R = A;
end