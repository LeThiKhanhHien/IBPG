%%
% Inertial Block Proximal Gradient Method IBPG for solving NCPD.
% see the paper "Inertial Block Proximal Method for non-convex non-smooth
% optimization", Le Thi Khanh Hien, Nicolas Gillis and Panos Patrinos, ICML 2020.
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
% Written by Andersen Ang and Le Thi Khanh Hien
% Last update June-2020
%%
function [U,V,W,e,t] = IBPG(Y,r,options)   
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
[~,t_f1,e] = objfun(1,normY2,W,U,V,Y_mode{3}); %compute initial f and time
time_f    = time_f + t_f1; % add recorded time to time_f for subtraction
t = toc(time_start) - time_f; % all the initialization time
if (t>=timemax) error('Initialization run longer then max runtime, set opt.timemax bigger'); end
t_now = t;
%% Main Loop
i = 1;
% initialization for iteration 1
parat(1) = 1; 
deltaw   = 0.99;  
beta     = 1.01;

inner_delta = 0.5; % inner loop stopping criterion for Hien et al. method
alphaparam = 0.02; 
% dimension of the tensor
% d1=size(U,1); d2=size(V,1); d3=size(W,1); % dimension of the tensor
if (max([d1,d2,d3]) <300) 
    alphaparam = 0.03; 
end
K = sum( Y_mode{1}(:) > 0 );
inneriterU = floor( 1 + alphaparam*((d2+d3)*r+K+d2*d3)/(d1*r+2*d1));
inneriterV = floor( 1 + alphaparam*((d1+d3)*r+K+d1*d3)/(d2*r+2*d2));
inneriterW = floor( 1 + alphaparam*((d1+d2)*r+K+d2*d1)/(d3*r+2*d3));
% the first 2 innitial points are the same
U_old=U; W_old=W; V_old=V; 
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
[U,U_old] = ibpg_innerloop(Qu,Pu,Lu(i),wu(i),U,U_old,beta,inneriterU,inner_delta);
%% *** On V ***
Qv    = (U'*U).*(W'*W); 
Pv    = Y_mode{2}*kr(U,W); % costly
Lv(i) = norm(Qv);    % Lipschitz constant 
    if (i==1)  wv(i) = omegahat(i);
    else       wv(i) = min( omegahat(i), deltaw * sqrt( Lv(i-1)/Lv(i) ) );
    end
[V,V_old] = ibpg_innerloop(Qv,Pv,Lv(i),wv(i),V,V_old,beta,inneriterV,inner_delta);
%% *** On W ***
Qw    = (U'*U).*(V'*V); 
Pw    = Y_mode{3}*kr(U,V); % costly
Lw(i) = norm(Qw);    % Lipschitz constant 
    if (i==1) ww(i) = omegahat(i);
    else      ww(i) = min( omegahat(i), deltaw * sqrt( Lw(i-1)/Lw(i) ) );
    end
[W,W_old] = ibpg_innerloop(Qw,Pw,Lw(i),ww(i),W,W_old,beta,inneriterW,inner_delta);
%% Error computation 
[~, t_ei, ei] = objfun(2,normY2,W,Pw, Qw);
time_f      = time_f + t_ei;            % add time on computing f to time_f
e=[e ei];
t_now  = toc(time_start) - time_f; % remove time_f, since no restart 
t   = [t t_now];                     % update t_now
i = i+1; % Iteration ++
end % end while
fprintf('IBPG done! \n'); 
end % EOF
%% iBPG update
function [U,U_old] = ibpg_innerloop(Q,P,L,w,U,U_old,beta,itermax,inner_delta)
eps  = 1;
eps0 = 0;
i = 1;
while i <= itermax && eps >= inner_delta*eps0 
  UUold  = U - U_old; % successive difference
  wUold  = w*UUold;
  Uhat   = U + wUold;
  Guhat  = Uhat*Q - P; % gradient matrix, using Uhat
  U_old  = U;
  % note Uhat2 = U + beta*w(i-1)*UUold;
  %      U = max(0, Uhat2 - Guhat / Lu(i-1)); % *** using Uhat2
  % calculate deltaU = U_new-U to use later for dynamic stopping
  deltaU = max( -U, beta*wUold- Guhat / L );
  U      = U + deltaU; % update U
  norm_diff = norm(deltaU,'fro');        
  if (i==1) 
      eps0 = norm_diff; 
  end
  eps = norm_diff;     
i = i + 1;
end 
end