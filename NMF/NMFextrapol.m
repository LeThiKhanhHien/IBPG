% [W,H,e,t,etrue] = NMFextrapol(X,r,options) 
%
% Acceleration of NMF algorithms using Extrapolation. 
% 
% This code allows to solve the NMF problem
% 
%            min_{W >=0 and H >= 0} || X - WH ||_F^2 . 
% 
% It uses extrapolation on the standard NMF alternating scheme: 
% 0. Initialize (W,H)
% for k = 1, 2, ...
%   1. Update H
%   2. Update W
% end
% W and H are updated using an existing NNLS algorithm (see NNLSalgo.m)
% 
% *********
%   Input
% *********
% X : input m-by-n nonnegative matrix 
% r : factorization rank r 
%
% ---Options--- 
% display : = 1, displays the evolution of the relative error (default), 
%           = 0 otherwise. 
% 
% init    : init.W and init.H are the intial matrices. 
%          default: W = rand(m,r); H = rand(r,n)
% 
% maxiter : maimum number of iterations. 
%          default: 100
% 
% timemax : maximum allotated time
%          default: Inf
%
% NMFalgo : = 'ANLS', using the active set method from Kim and Park 
%           = 'HALS', using the accelerated HALS method from Gillis and
%                       Glineur (HALS from Cichocki et al.) (default)
%           = 'MUup', using the accelerated multiplicative updated of from 
%                       Gillis and Glineur (MU from Lee and Seung)
% 
% extrapolprojH : = 1, extrapolation of H is performed after the update of W
%                 = 2, extrapolation of H is performed directly
%                 = 3, extrapolation of H is performed directly and it is
%                 projected onto the nonnegative orthant. (default) 
%
% beta0 : initial value of the paramter beta un [0,1) 
%          default: 0.5
%
% eta : decrease of beta when error increases 
%          default: 1.5
%
% gammabeta : increase of beta when error decreases, it has to be  < eta
%          default: 1.1 for ANLS, 1.01 for A-HALS and MU
%
% gammabetabar : increase of the upper bound on beta when error decreases 
%                It has to be < gammabeta. 
%          default: 1.05 for ANLS, 1.005 for A-HALS and MU
%
% **********
%   Output
% **********
% (W,H) : rank-r NMF of X, that is, W is m-by-r, H is r-by-n, W>=0, H>=0, 
%          W*H approximates X. 
% e     : relative error of the iterates (W_y,H_n), that is,
%          ||X-W_yH_n||_F/||X||_F where W_y is the extrapolated sequence 
%          and H_n is the NNLS update of H_y. Note that W_y is not 
%          necessarily nonnegative (when options.extrapolprojH=2) but 
%          this value is cheap to compute. 
% t     : cputime --> plot(t,e) displays the evolution of the error e over
%           time
% etrue : if hp~=2, etrue = e; otherwise etrue is the error of (W_n,H_n)
%           which has to be recomputed from scratch hence the output etrue
%           should only be used for benchmarking. 
%
% See M.S. Ang and N. Gillis, Accelerating Nonnegative Matrix Factorization
% Algorithms using Extrapolation, 2018. 

function [W,H,e,t,etrue] = NMFextrapol(X,r,options) 

cputime0 = tic; 
[m,n] = size(X); 
addpath('./NMFalgo');  
if nargin < 3
    options = [];
end
if ~isfield(options,'display')
    options.display = 1; 
end
%% Parameters of NMF algorithm
if ~isfield(options,'init')
    W = rand(m,r); 
    H = rand(r,n); 
else
    W = options.init.W; 
    H = options.init.H; 
end
if ~isfield(options,'maxiter')
    options.maxiter = 100; 
end
if ~isfield(options,'NMFalgo')
    options.NMFalgo = 'HALS'; 
end
if ~isfield(options,'timemax')
    options.timemax = Inf; 
end
%% Parameters of extrapolation
if ~isfield(options,'extrapolprojH')
    options.extrapolprojH = 3; 
end
if ~isfield(options,'beta0')
    options.beta0 = 0.5; 
end
if ~isfield(options,'eta')
    options.eta = 1.5; 
end
if ~isfield(options,'gammabeta')
    if options.NMFalgo == 'ANLS'
        options.gammabeta = 1.1; 
    else
        options.gammabeta = 1.01; 
    end
end
if ~isfield(options,'gammabetabar')
    if options.NMFalgo == 'ANLS'
        options.gammabetabar = 1.05; 
    else
        options.gammabetabar = 1.005; 
    end
end
if options.eta < options.gammabeta || options.gammabeta < options.gammabetabar
    error('You should choose eta > gamma > gammabar.');
end
if options.beta0 >= 1 && options.beta0 <= 0
    error('beta0 must be in the interval [0,1].');
end
    
%% Scale initialization  
XHt = X*H'; HHt = H*H'; 
scaling = sum(sum(XHt.*W))/sum(sum( HHt.*(W'*W) )); 
W = W*scaling; 
%% Extrapolation 
Wy = W; 
Hy = H; 
% parameters
beta(1) = options.beta0; 
betamax = 1; 
nX = norm(X,'fro'); 
e(1) = sqrt( nX^2 - 2*sum(sum( XHt.*W ) ) + sum( sum( HHt.*(W'*W) ) ) ) / nX; 
etrue(1) = e(1); 
emin = e(1); 
t(1) = toc(cputime0); 
i = 1; 
if options.display == 1
    disp('Display of iteration number and relative error in precent:')
    kdis = 1; 
end
%% Main loop
while i <= options.maxiter && t(i) < options.timemax 
    %% NNLS for H 
    Hn = NNLSalgo(Wy, X, Hy, options.NMFalgo); 
    % Extrapolation
    if options.extrapolprojH >= 2 
        Hy =  Hn + beta(i)*(Hn-H) ;
    else
        Hy = Hn;
    end
    % Projection
    if options.extrapolprojH == 3
        Hy = max(0, Hy); 
    end
    %% NNLS for W 
    [Wn,HyHyT,XHyT] = NNLSalgo(Hy', X', Wy', options.NMFalgo); 
    Wn = Wn'; XHyT = XHyT'; HyHyT = HyHyT'; 
    % Extrapolation
    Wy = Wn + beta(i)*(Wn-W);  
    if options.extrapolprojH == 1
        Hy =  Hn + beta(i)*(Hn-H) ;
    end
    %% Relative error 
    e(i+1) = max(0,nX^2 - 2*sum(sum( XHyT.*Wn ) ) + sum( sum( HyHyT.*(Wn'*Wn) ) ));
    e(i+1) = sqrt(e(i+1))/nX; 
    t(i+1) = toc(cputime0); 
    %% Compute the 'true' error of the latest feasible solution (Wn,Hn)
    %% if options.extrapolprojH == 2, that is, if Hy is not feasible.
    %% This should be used (only) for benchmarking purposes.  
    nottakenintoaccount = tic; 
    if nargout >= 5
        % etrue(i+1) is not equal to e(i) only when H_y is not feasible
        if options.extrapolprojH == 2 && min(Hy(:)) < 0 
        	etrue(i+1) = max(0,nX^2 - 2*sum(sum( (X*Hn').*Wn ) ) + sum( sum( (Hn*Hn').*(Wn'*Wn) ) ));
            etrue(i+1) = sqrt(etrue(i+1))/nX; 
        else 
            etrue(i+1) = e(i+1); 
        end
    end
    % Do not take into account computation of etrue that are not needed
    t(i+1) = t(i+1) - max(0,toc(nottakenintoaccount)); 
    %% Check the evolution of the error 
    if e(i+1) > e(i) 
        % restart the scheme 
        Wy = W; 
        Hy = H; 
        betamax  = beta(i-1); 
        beta(i+1) = beta(i)/options.eta; 
    else
        % keep previous iterate in memory
        W = Wn; 
        H = Hn; 
        beta(i+1) = min(betamax, beta(i)*options.gammabeta); 
        betamax = min(1, betamax*options.gammabetabar); 
    end
    %% Keep best iterate in memory
    if e(i+1) <= emin
        Wbest = Wn; 
        Hbest = max(Hy,0);
        emin = e(i+1); 
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
    %% Increment
    i = i+1; 
end 
if options.display == 1
    fprintf('\n');
end
W = Wbest; 
H = Hbest; 