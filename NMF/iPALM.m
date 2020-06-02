function [Wn,Hn,e,t] = iPALM(X,r,options) 
cputime0 = tic; 
[m,n] = size(X); 
nX = norm(X,'fro'); 
%% Parameters of NMF algorithm
if nargin < 3
    options = [];
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
% scale the innitial point 
HHt = H*H'; 
XHt = X*H'; 
scaling = sum(sum(XHt.*W))/sum(sum( (W'*W).*(HHt) )); 
W = W*scaling; 

time1=tic;
e(1) = sqrt( nX^2 - 2*sum(sum( XHt.*W ) ) + sum( sum( HHt.*(W'*W) ) ) ) / nX; 
time_er=toc(time1);
t(1) = toc(cputime0)-time_er; %remove time of computing the error

i = 1; 
Wn=W;
Hn=H;
Wold=W; % (Wold,Hold) is to save the previous point
Hold=H;
while i <= options.maxiter && t(i) < options.timemax  
    alpha=(i-1)/(i+2);
    %% Update Wn
    W=Wn;
    HHt = Hn*Hn'; 
    XHt = X*Hn'; 
    Lw(i) = norm(HHt); 
    Wy = Wn + alpha*(Wn - Wold); % extrapolation
    gradWy = Wy*HHt - XHt; 
    Wn = max( 0 , Wy - gradWy/Lw(i)  );
    Wold = W;
    
    %% Update Hn
    H=Hn;
    WtW = Wn'*Wn; 
    WntX = Wn'*X; 
    Lh(i) = norm(WtW); 
    Hy = Hn + alpha*(Hn - Hold);
    gradHy = WtW*Hy - WntX;  
    Hold = H;
    Hn = max( 0 , Hy - gradHy/Lh(i)  ); 
    
    % calculate relative error
    time1=tic;
    e(i+1)= nX^2 - 2*sum(sum( (Wn*Hn).*X ) ) + sum(sum( (Wn'*Wn).*(Hn*Hn') ) );
    e(i+1)=sqrt(max(0,e(i+1)))/nX;
    time_er=time_er+toc(time1);
    t(i+1) = toc(cputime0)-time_er;
    i = i + 1; 
end

end