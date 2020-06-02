%%
% Inertial Block Proximal Gradient Method for NMF with the Frobenius norm: 
%           min_{W,H} ||X-WH||_F^2 such that W>=0 & H>=0, 
% using 2 different extrapolated points to calculate gradient and  add 
% inertial force; see the paper 
% Inertial Block Proximal Method for non-convex non-smooth
% optimization, Le Thi Khanh Hien, Nicolas Gillis and Panos Patrinos, ICML 2020.
% 
% Input 
%       X: the matrix to be factorized 
%       r: rank 
%       options.innertiter: number of times to repeat updating W or H; default =1 
%       options.init: initial point; default W = rand(m,r);   H = rand(r,n);
%       options.maxiter: number of max iterations, default = 200
%       options.timemax: max running time, default = inf
%       options.display: option for displaying the result, default=1 means
%                                                                       yes
% Output
%       (Wn,Hn): solution 
%        e: sequence of relative error ||X-WH||_F/||X|| with respect to main iterations
%        t: sequence of running time with respect to main iterations
%
% Written by Le Thi Khanh Hien
% Last update June-2020
%%
function [Wn,Hn,e,t] = IBPG(X,r,options) 
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
    options.maxiter = 200; 
end
if ~isfield(options,'timemax')
    options.timemax = Inf; 
end
if ~isfield(options,'inneriter')
    options.inneriter = 1; 
end
%% Main loop
kdis=1;

% scale the innitial point 
HHt = H*H'; 
XHt = X*H'; 
scaling = sum(sum(XHt.*W))/sum(sum( (W'*W).*(HHt) )); 
W = W*scaling; 

nX = norm(X,'fro'); 
beta=1.01;

i = 1; 
paramt(i) = 1; 
Wn=W;
Hn = H; 
Wold=W;
Hold=H;
 

deltaw = 0.99; % 
time1=tic;
e(1) = sqrt( nX^2 - 2*sum(sum( XHt.*W ) ) + sum( sum( HHt.*(W'*W) ) ) ) / nX; 
time_er=toc(time1);
t(1) = toc(cputime0)-time_er; %remove time of computing the error
delta=0.01;
while i <= options.maxiter && t(i) < options.timemax  
    paramt(i+1) = 0.5 * ( 1+sqrt( 1 + 4*paramt(i)^2 ) ); 
    what(i) = (paramt(i)-1)/paramt(i+1); 
    %% Update Wn
    HHt = Hn*Hn'; 
    XHt = X*Hn'; 
    Lw(i) = norm(HHt); 
    if i == 1
        ww(i) = what(i); 
    else
        ww(i) = min( what(i), deltaw*sqrt( Lw(i-1)/Lw(i) ) ); 
    end
    j=1;eps0 = 0; eps = 1; 
    while j<=options.inneriter &&  eps >= (delta)^2*eps0
       W=Wn;
       Wnold=Wn - Wold;
       Wy = Wn + ww(i)*Wnold; 
       gradWy = Wy*HHt - XHt; 
       
       deltaW= max( -Wn ,  beta*ww(i)*Wnold - gradWy/Lw(i)  );
       %Wy2=Wn + beta*ww(i)*Wnold; % use another extrapolation point
       Wn = Wn+deltaW; % max( 0 , Wy2 - gradWy/Lw(i)  );
       
       if options.inneriter>1 
           eps= norm(deltaW,'fro');
       end
       Wold = W;
       
       if j==1 
           eps0=eps;
       end
       j=j+1;
    end
    %% Update Hn
    WtW = Wn'*Wn; 
    WntX = Wn'*X; 
    Lh(i) = norm(WtW); 
    if i == 1
        wh(i) = what(i); 
    else
        wh(i) = min( what(i), deltaw*sqrt( Lh(i-1)/Lh(i) ) ); 
    end
    j=1; eps0 = 0; eps = 1; 
    while j<=options.inneriter &&  eps >= (delta)^2*eps0
       H=Hn;
       Hnold=Hn - Hold;
       Hy = Hn + wh(i)*Hnold;
       gradHy = WtW*Hy - WntX;  
       deltaH=max( -Hn ,  beta*wh(i)*Hnold - gradHy/Lh(i)); 
       %Hy2= Hn + beta*wh(i)*Hnold;  % use another extrapolation point
       Hn =Hn+deltaH; % max( 0 , Hy2 - gradHy/Lh(i)  ); 
       if options.inneriter>1 
         eps=norm(deltaH,'fro');
       end
       Hold = H;
       if j==1
           eps0=eps;
       end
       j=j+1;
    end
    time1=tic;
    e(i+1)= nX^2 - 2*sum(sum( (Wn*Hn).*X ) ) + sum(sum( (Wn'*Wn).*(Hn*Hn') ) );
    e(i+1)=sqrt(max(0,e(i+1)))/nX;
    time_er=time_er+toc(time1);
    t(i+1) =toc(cputime0) - time_er; 
    
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