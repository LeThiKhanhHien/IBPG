%% Computes order-3 NCPD via BCD using ADMM
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
% written by Andersen Ang
%%
function [U,V,W,e,t] = ADMM(Y,r,options)
            
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
 UtU=U'*U; VtV=V'*V; WtW=W'*W; Utild=U; Vtild=V; Wtild=W;
 
 t = toc(time_start) - time_f; % all the initialization time
if (t>=timemax) 
    error('Initialization run longer then max runtime, set opt.timemax bigger'); 
end
t_now = t;
%% Main Loop
i = 1; % iteration counter for the main loop
inner_delta = 1e-2; % inner loop early stopping critetion
inneritermax=50;
while i <= itermax && t_now < timemax
%% *** On U ***
% Compute parameters for update
Qu = (V'*V).*(W'*W); 
Pu = Y_mode{1}*kr(V,W); % 1 full size MTTKRP, costly 

% Perform the block update
[U,Utild,UtU] = nnlsadmm(Pu',U,Utild,Qu,UtU,inneritermax,inner_delta); 
%% *** On V ***
 % Compute parameters for update
 Qv = (U'*U).*(W'*W); 
 Pv = Y_mode{2}*kr(U,W); % 1 full size MTTKRP, costly 
 % Perform the block update
 [V,Vtild,VtV] = nnlsadmm(Pv',V,Vtild,Qv,VtV,inneritermax,inner_delta);
 %% *** On W ***
 % Compute parameters for update
 Qw = (U'*U).*(V'*V);
 Pw = Y_mode{3}*kr(U,V); % 1 full size MTTKRP, costly 
 % Perform the block update
 [W,Wtild,WtW] = nnlsadmm(Pw',W,Wtild,Qw,WtW,inneritermax,inner_delta);
 %% Error computation 
[~, t_ei, ei] = objfun(2,normY2,W,Pw, Qw);
time_f      = time_f + t_ei;            % add time on computing f to time_f
e = [e ei];
t_now        = toc(time_start) - time_f; % remove time_f, since no restart 
t       = [t t_now];                     % update t_now
i = i+1; % Iteration ++
end%endWhile
fprintf('ADMM done! \n'); 
end

function [U,Uhat,UtU] = nnlsadmm(Atb, U, Uhat, AtA, UtU, itermax, delta)
% solve min(1/2)||Y-U*V'||^2, U >= 0 using ADMM
%  U is main sequence
%  Uhat is Lagrangian multiplier
%  Uy is ADMM pairing variable
% Note. computing UtU inside, and take UtU as input is important
% as UtU = U'*U, a r-by-m times m-by-r, expensive if m big
r   = size(U,2);
rho = trace(AtA)/r; % heuristics rho selection
if rho==0 
    rho=1e-15;
end
L   = chol( AtA + rho*eye(r),'lower');% MOD
eps  = 1;
eps0 = 0;
i    = 1; % iteration counter
    while i <= itermax && eps >= delta*eps0   
     Uold = U; 
     Uy    = L' \ ( L\ ( Atb + rho*(U+Uhat)' ) ); 
     U     = max( 0, Uy'-Uhat);
     Uhat  = Uhat + U - Uy';
     norm_diff = norm(U-Uold,'fro'); % can be improved

     if (i == 1)  eps0 = norm_diff;  end
     eps = norm_diff;     
    i=i+1;
    end
UtU = U'*U;
end %EOF