function [U,V,W,e,t] = APG(Y,r,options)
% Computes order-3 NCPD via accelerated alternating Projected Gradient
% Y. Xu and W. Yin, "A Block Coordinate Descent Method for Regularized 
% Multiconvex Optimization with Applications to Nonnegative Tensor
% Factorization and Completion", SIAM J. Imaging Sci., 6(3), 1758–1789
%
%
% Input:
% Y, r           : order-3 data tensor and factorization rank
% options (algorithm parameters)
%  .timemax      : max run time(sec), default 60
%  .itermax      : max num iterations of outer loop, default 100
%  init.U, init.V, init.W : initial mode-1,2,3 NN factors U,V,W 
% 
% Output
% U,V,W          : estimated mode-1,2,3 NN factors
%                  reconstruct data by tensorForm(dims,U,V,W)
%   e            : sequence of relative error ||X-U \circ V \circ W||_F/||X|| with respect to 
%                  iterations
%   t            : sequence of running time with respect to iterations
% 
time_start = tic; 
time_f     = 0;  
if (nargin < 3) 
    options = [];
end

[d1,d2,d3]= size(Y);  
dims=[d1,d2,d3];

if ~isfield(options,'init')
  U = rand(d1,r);
  V = rand(d2,r);
  W = rand(d3,r);
else
    U = options.init.U; 
    V = options.init.V; 
    W = options.init.W;
end
if ~isfield(options,'itermax')
    options.itermax = inf; 
end
if ~isfield(options,'timemax')
    options.timemax = 5; 
end
itermax=options.itermax ;
timemax=options.timemax;

 normY2=Y(:)'*Y(:); 
 Y_mode   = unfold3mode(Y,dims);
 [~,~,e] = objfun(1,normY2,W,U,V,Y_mode{3}); %compute initial f and time
 t = toc(time_start); % record the initialization time
if (t>=timemax) error('Initialization run longer then max runtime, set opt.timemax bigger'); end
t_now = t;
%% Main Loop
i = 1; % iteration counter for the main loop
% initialization for iteration 1
deltaw   = 0.9999; % value from the paper [Xu_13]
parat(1) = 1;
U_old = U;   V_old = V;  W_old = W; % old variable for extrapolation

while i <= itermax && t_now < timemax
 parat(i+1) = 0.5 * ( 1+sqrt( 1 + 4*parat(i)^2 ) ); 
 omegahat(i) = (parat(i) - 1)/parat(i+1);  
%% *** On U ***
Qu    = (V'*V).*(W'*W); 
Pu    =  Y_mode{1}*kr(V,W); % costly
Lu(i) = norm(Qu);    % Lipschitz constant 
    if  (i==1) wu(i) = omegahat(i);
    else       wu(i) = min( omegahat(i), deltaw * sqrt( Lu(i-1)/Lu(i) ) );   
    end
[U,U_old] = apg_innerloop(Qu,Pu,Lu(i),wu(i),U,U_old);
%% *** On V ***
Qv    = (U'*U).*(W'*W); 
Pv    = Y_mode{2}*kr(U,W); % costly
Lv(i) = norm(Qv);    % Lipschitz constant 
    if (i==1)  wv(i) = omegahat(i);
    else       wv(i) = min( omegahat(i), deltaw * sqrt( Lv(i-1)/Lv(i) ) );
    end
[V,V_old] = apg_innerloop(Qv,Pv,Lv(i),wv(i),V,V_old);
%% *** On W ***
Qw    = (U'*U).*(V'*V); 
Pw    = Y_mode{3}*kr(U,V); % costly
Lw(i) = norm(Qw);    % Lipschitz constant 
    if (i==1) ww(i) = omegahat(i);
    else      ww(i) = min( omegahat(i), deltaw * sqrt( Lw(i-1)/Lw(i) ) );
    end
[W,W_old] = apg_innerloop(Qw,Pw,Lw(i),ww(i),W,W_old);
%% Error computation and restart 
[~, ~, ei] = objfun(2,normY2,W,Pw, Qw);
if ((i>1)&&(ei>e(end))) 
    U=U_old; V=V_old; W=W_old; 
    e=[ e e(end)];
else
    e =[e ei];
end % restart
 % get time 
t_now = toc(time_start);  
t = [t t_now];% update t_now

i = i+1; % Iteration ++
end % end while

fprintf('APG done! \n'); 
end % EOF
%% The inner loop of APG
function [U,U_old] = apg_innerloop(Q,P,L,w,U,U_old)
wUold  = w*(U - U_old);
Uhat   = U + wUold;
Guhat  = Uhat*Q -P; % gradient matrix
U_old  = U;
deltaU = max(-U, wUold - Guhat / L);
U      = U + deltaU;
end%EOF