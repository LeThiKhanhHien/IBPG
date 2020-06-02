% Update of V <- HALS(M,U,V)
% i.e., optimizing min_{V >= 0} ||M-UV||_F^2 
% with an exact block-coordinate descent scheme
% 
% Code from 
% Gillis, N., & Glineur, F. (2012). Accelerated multiplicative updates 
% and hierarchical ALS algorithms for nonnegative matrix factorization. 
% Neural computation, 24(4), 1085-1105.

function [V,UtU,UtM,e] = nnlsHALSupdt(M,U,V,alphaparam,delta)

if nargin <= 3
    alphaparam = 0.5; 
end
KX = sum( M(:) > 0 );
[m,n] = size(M);
[m,r] = size(U);
maxiter = floor( 1 + alphaparam*(KX+m*r)/(n*r+n) );
if nargin <= 4
    delta = 0.01; 
end
%% Precomputations 
UtU = U'*U;
UtM = U'*M; 
nM = norm(M,'fro'); 
%% Scaling initial iterate, not necessary if (U,V) is already well scaled
%scaling = sum(sum(UtM.*V))/sum(sum( UtU.*(V*V') )); 
%V = V*scaling; 
%% Coordinate descent 
eps0 = 0; cnt = 1; eps = 1; 
while eps >= (delta)^2*eps0 && cnt <= maxiter
    nodelta = 0; if cnt == 1, eit3 = cputime; end
    cputime1 = cputime; 
        for k = 1 : r
            deltaV = max((UtM(k,:)-UtU(k,:)*V)/UtU(k,k),-V(k,:));
            V(k,:) = V(k,:) + deltaV;
            nodelta = nodelta + deltaV*deltaV'; % used to compute norm(V0-V,'fro')^2;
            if V(k,:) == 0, V(k,:) = 1e-16*max(V(:)); end % safety procedure
        end
    if cnt == 1
        eps0 = nodelta; 
    end
    eps = nodelta; 
    if nargout >= 4
        e(cnt) = sqrt( nM^2 - 2*sum(sum( UtM.*V ) ) + sum(sum( UtU.*(V*V') )) ) / nM; 
    end
    cnt = cnt + 1; 
end