% This function aims to solve exactly or approximately the nonnegative
% least square (NNLS) problem of the form: 
% 
%         min_{H >= 0} ||X - WH||_F^2 
%
% using different algorithms: 
% (1) 'HALS' which uses block coordinat descent method (on the rows of H)
% (2) 'ANLS' which solves the problem exactly using an active-set method
% (3) 'MUUP' which uses multiplicative updates (this is the approach
%       popularized by Lee and Seung but initially proposed in 
%       Daube-Witherspoon, M. E., & Muehllehner, G. (1986). An iterative 
%       image space reconstruction algorithm suitable for volume ECT. IEEE 
%       Trans. Med. Imaging, 5, 61–66. 
% 
% For both HALS and MUUP, we use the accelerated variants, that is, we
% perform several updates. 

function [H,WTW,WTX] = NNLSalgo(W,X,H,algo) 

if algo == 'HALS' % Accelerated hierarchical alternating least squares
    % Gillis, N., & Glineur, F. (2012). Accelerated multiplicative updates 
    % and hierarchical ALS algorithms for nonnegative matrix factorization. 
    % Neural computation, 24(4), 1085-1105.
    [H,WTW,WTX] = nnlsHALSupdt(X,W,H); 
elseif algo == 'ANLS' % Alternating nonnegative least squares
    % Kim, J., & Park, H. (2011). Fast nonnegative matrix factorization: 
    % An active-set-like method and comparisons. 
    % SIAM Journal on Scientific Computing, 33(6), 3261-3281.
    % Sometimes, nnlsm_blockpivot fails and returns NaN solutions for rank
    % deficient matrices. This happens usually if a column of W is equal 
    % to zero, which we avoid by checking this case beforehand: 
    indi = find( sum(W) < 1e-12 ); 
    if ~isempty(indi)
        W(:,indi) = rand(size(W,1),length(indi)); 
    end
    [H,~,WTW,WTX] = nnlsm_blockpivot( W, X, 0, max(0,H) ); 
    % If nnlsm_blockpivot fails for rank deficient matrices with
    % non-zero columns (this is extremely rare in practice), then we
    % execute nnlsm_activeset
    if sum( isnan(H(:)) ) > 0
        [H,~,WTW,WTX] = nnlsm_activeset( W, X, 0, 0, max(0,H) ); 
    end
elseif algo == 'MUUP' % Multiplicative updates
    WTW = W'*W; 
    WTX = W'*X; 
    KX = sum( X(:) > 0 );
    [m,n] = size(X);
    [m,r] = size(W);
    alphaparam = 2; 
    maxiter = floor( 1 + alphaparam*(KX+m*r)/(n*r+n) ); 
    i = 1; 
    while i <= maxiter
        H = max(1e-16, H.*(WTX)./(WTW*H)); 
        i = i+1; 
    end
end