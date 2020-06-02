%%
% Inertial Block Proximal Method for NMF with the Frobenius 
% norm: 
%           min_{W,H} ||X-WH||_F^2 such that W>=0 & H>=0, 
% IBP allows repeating the cyclic update of columns of one matrix many 
% times before moving on to the other without restarting step; see the paper 
% Inertial Block Mirror Descent Method for non-convex non-smooth
% optimization, Le Thi Khanh Hien, Nicolas Gillis and Panos Patrinos, ICML 2020.
%
% Input 
%       X: the matrix to be factorized 
%       r: rank 
%       options.innertiter: number of times to repeat the cyclic update of columns; default =1 
%       options.init: initial point; default W = rand(m,r);   H = rand(r,n);
%       options.maxiter: number of max iterations, default = 100
%       options.timemax: max running time, default = inf
%       options.display: option for displaying the result, default=1 means
%                                                                       yes
%       beta: coefficient to find proximal point
%       gamma and alpha: coefficients to calculate extrapolation points
%       
% Output
%       (W,H): solution 
%        e: sequence of relative error ||X-WH||_F/||X|| with respect to main iterations
%        t: sequence of running time with respect to main iterations
%
% Written by Le Thi Khanh Hien
% Last update June-2020
%%
function [W,H,e,t] = IBP(X,r,beta,gamma,alpha,options) 

cputime0 = tic; 
[m,n] = size(X); 
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
    options.maxiter = 100; 
end
if ~isfield(options,'timemax')
    options.timemax = Inf; 
end

%% Main loop
kdis=1;
% scale the innitial point 
HHt = H*H'; 
XHt = X*H'; 
scaling = sum(sum(XHt.*W))/sum(sum( (W'*W).*(HHt) )); 
W = W*scaling;

nX=norm(X,'fro');
time1=tic;
e = sqrt( nX^2 - 2*sum(sum( XHt.*W ) ) + sum( sum( HHt.*(W'*W) ) ) ) / nX;
time_er=toc(time1);
t=toc(cputime0)-time_er;
tnext=t;
% first two initial points 
Hold=H; 
Wold=W;
i=1;

% number of inner iterations

[m,r] = size(W); 
alphaparam = 0.5;
KX = sum( X(:) > 0 );
maxiterH = floor( 1 + alphaparam*(KX+m*r)/(n*r+n) );
maxiterW=floor( 1 + alphaparam*(KX+n*r)/(m*r+m) );
 
betabar=1;
while i <= options.maxiter && tnext < options.timemax  
   % update H
   alpha=min(betabar,gamma*alpha);
   [H,Hold] = prox_col(X,W,H,Hold,r,maxiterH,alpha,beta); 
   
   % update W
   [W,Wold,eexi] = prox_col(X',H',W',Wold',r,maxiterW,alpha,beta); 
   W = W'; 
   Wold=Wold';
   e =[e  eexi]; 
   tnext = toc(cputime0); 
   t=[t tnext];
   
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
    
 i=i+1;  
    
end
if options.display == 1, fprintf('\n'); end 
end