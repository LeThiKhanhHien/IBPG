% Extrapolaed gradient descent method from Algorithm 2 page 1774 of 
% 
% A Block Coordinate Descent Method for Regularized Multiconvex 
% Optimization with Applications to Nonnegative Tensor Factorization and 
% Completion, Yangyang Xu and Wotao Yinn, 
% SIAM J. IMAGING SCIENCES, Vol. 6, No. 3, pp. 1758–1789. 

function [Wn,Hn,e,t] = APGC(X,r,options) 

cputime0 = tic; 

%% Parameters of NMF algorithm
if nargin < 3
    options = [];
end
if ~isfield(options,'display')
    options.display = 1; 
end
if ~isfield(options,'init')
    W = rand(m,r); 
    H = rand(r,n); 
else
    W = options.init.W; 
    H = options.init.H; 
end
if ~isfield(options,'maxiter')
    options.maxiter = 200; 
end
if ~isfield(options,'timemax')
    options.timemax = Inf; 
end

%% Main loop
kdis=1;
[m,n] = size(X); 
nX = norm(X,'fro'); 
i = 1; 
paramt(i) = 1; 
% scale the innitial point 
HHt = H*H'; 
XHt = X*H'; 
scaling = sum(sum(XHt.*W))/sum(sum( (W'*W).*(HHt) )); 
W = W*scaling; 

Wn=W; % (Wn,Hn) is to save the current point 
Hn = H;  
Wold=W; % (Wold,Hold) is to save the previous point
Hold=H;

 % time axis
deltaw = 0.9999; % delta in Xu-Yin's paper. 

e(1)= nX^2 - 2*sum(sum( (W*H).*X ) ) + sum(sum( (W'*W).*(H*H') ) );
e(1)= sqrt(max(0,e(1)))/nX; % e is to save relative error
t(1) = toc(cputime0);
while i <= options.maxiter && t(i) < options.timemax  
    paramt(i+1) = 0.5 * ( 1+sqrt( 1 + 4*paramt(i)^2 ) ); 
    what(i) = (paramt(i)-1)/paramt(i+1); 
    %% Update Wn
    W=Wn;
    HHt = Hn*Hn'; 
    XHt = X*Hn'; 
    Lw(i) = norm(HHt); 
    if i == 1
        ww(i) = what(i); 
    else
        ww(i) = min( what(i), deltaw*sqrt( Lw(i-1)/Lw(i) ) ); 
    end
    Wy = Wn + ww(i)*(Wn - Wold); % extrapolation
    gradWy = Wy*HHt - XHt; 
    Wn = max( 0 , Wy - gradWy/Lw(i)  );
    Wold = W;
    %% Update Hn
    H=Hn;
    WtW = Wn'*Wn; 
    WntX = Wn'*X; 
    Lh(i) = norm(WtW); 
    if i == 1
        wh(i) = what(i); 
    else 
        wh(i) = min( what(i), deltaw*sqrt( Lh(i-1)/Lh(i) ) ); 
    end
    Hy = Hn + wh(i)*(Hn - Hold);
    gradHy = WtW*Hy - WntX;  
    Hold = H;
    Hn = max( 0 , Hy - gradHy/Lh(i)  ); 
    
    e(i+1)= nX^2 - 2*sum(sum( (Wn*Hn).*X ) ) + sum(sum( (Wn'*Wn).*(Hn*Hn') ) );
    e(i+1)=sqrt(max(0,e(i+1)))/nX;
    t(i+1) = toc(cputime0); 
    
    if e(i+1) > e(i) 
        % restart the scheme 
        Wn = W; 
        Hn = H; 
    end
    
    %% Display iteration number and error in percent
    if options.display == 1
        
        if i == 1
            % Display roughly every second, being a multiple of 10
            displayparam = max(  1, 10^round( log10( (t(1)+1e-2)^(-1)) )  ); 
        end
        if i/displayparam == floor(i/displayparam) 
            if e(i+1) < 1e-4
                fprintf('%2.0f: %2.1d...', i, 100*e(i+1)); 
            else
                fprintf('%2.0f: %2.2f...', i, 100*e(i+1)); 
            end
            kdis = kdis + 1; 
        end
        if kdis == 11
            kdis = 1; 
            fprintf('\n'); 
        end
    end
    i = i + 1; 
end
if options.display == 1, fprintf('\n'); end 
end