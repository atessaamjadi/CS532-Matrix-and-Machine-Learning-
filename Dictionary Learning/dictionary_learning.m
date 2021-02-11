function [D,R] = dictionary_learning(X, k, iteration, tolerance, debug_level)
    [U,S,V] = svds(X, k);
    D = U;
    R = S*V';
    for i = 1:iteration
        R = sparse_code(X, D, R);
        
        err = norm(X - D*R);  
        if debug_level >= 1
            fprintf('Iteration %.d, error is %.4g, nonzero is %.d', i, err, nnz(R));
        end
        
        [D, R] = update_dictionary(X, D, R);
        err = norm(X - D*R); 
        if debug_level >= 1
            fprintf('Iteration %.d, error is %.4g', i, err);
        end
        if err < tolerance
            break
        end
    end
end

function R = sparse_code(X, D, R)

R = proximal_gd(X, D, R, 0.2, 2 ,100);

end

function R = proximal_gd(X, D, R_init, tau, lam, it)
% compute it iterations of L1 proximal gradient descent starting at R_init
%
%
% step size tau

R = R_init;

for k = 1:it
    Z = R - tau*D'*(D*R - X);
    R = wthresh(Z,'s',lam*tau/2);
end

end

function [D,R] = update_dictionary(X, D, R)
    
% K-svd to update the dictionary.

size_D = size(D);
cols = size_D(2);

for k = 1:cols
    nonzero_index = find(R(k,:));
    [~, b] = size(nonzero_index);
    if b == 0
        continue;
    end
    D(:,k)=0;
    Ek = X - D*R;
    Ek = Ek(:, nonzero_index);
    [U,S,V] = svds(Ek,1);
    D(:,k) = reshape(U,[],1);
    R(k, nonzero_index) = S * V;
end

end