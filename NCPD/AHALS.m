%% Computes order-3 NCPD via BCD with AHALS 
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
function [U,V,W,e,t] = AHALS(Y,r,options)

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
 if (t>=timemax) 
    error('Initialization run longer then max runtime, set opt.timemax bigger'); 
 end
 t_now = t;

 %% Main Loop
i = 1; % iteration counter for the main loop
inneritermax = 50; 
while i <= itermax && t_now < timemax
%% *** On U ***
% Compute parameters for update
Qu = (V'*V).*(W'*W); 
Pu = Y_mode{1}*kr(V,W); % 1 full size MTTKRP, costly 
% Perform the block update
U = BCD_blkUpdt( U, Qu, Pu, inneritermax );
%% *** On V ***
 % Compute parameters for update
 Qv = (U'*U).*(W'*W); 
 Pv = Y_mode{2}*kr(U,W); % 1 full size MTTKRP, costly 
  % Perform the block update
 V = BCD_blkUpdt( V, Qv, Pv, inneritermax ); 
%% *** On W ***
 % Compute parameters for update
 Qw = (U'*U).*(V'*V);
 Pw = Y_mode{3}*kr(U,V); % 1 full size MTTKRP, costly 
 % Perform the block update
 W = BCD_blkUpdt( W, Qw, Pw, inneritermax ); 
%% Error computation 
[~, t_ei, ei] = objfun(2,normY2,W,Pw, Qw);
time_f      = time_f + t_ei;            % add time on computing f to time_f
e = [e ei];
t_now        = toc(time_start) - time_f; % remove time_f, since no restart 
t       = [t t_now];                     % update t_now
i = i+1; % Iteration ++
end%endWhile
fprintf('A-HALS done! \n'); 
end

function A = BCD_blkUpdt( A, Q, P, itermax )
  A  = nnlsHALSupdt_jec(Q,P',A',itermax)';   
end
function V = nnlsHALSupdt_jec(UtU,UtM,V,maxiter)
% Computes an approximate sol of 
%           min_{V >= 0} ||M-UV||_F^2 
% with an exact block-coordinate descent scheme. 
% === Input ===============================================================
% M  : m-by-n matrix 
% U  : m-by-r matrix
% V  : r-by-n initialization matrix 
%      default: one non-zero entry per column corresponding to the 
%      clostest column of U of the corresponding column of M 
% maxiter: upper bound on the number of iterations (default=500).
%
% Remark. M, U and V are not required to be nonnegative. 
% === Output ==============================================================
% V  : an r-by-n nonnegative matrix \approx argmin_{V >= 0} ||M-UV||_F^2
% === Reference ===========================================================
% N. Gillis and F. Glineur, Accelerated Multiplicative Updates and 
% Hierarchical ALS Algorithms for Nonnegative Matrix Factorization, 
% Neural Computation 24 (4): 1085-1105, 2012.
if nargin <= 3  maxiter = 500; end % default 500 iterations

r = size(UtU,1); 

if nargin <= 2 || isempty(V) 
  V = pinv(UtU)*UtM; % Least Squares
  V = max(V,0); 
  alpha = sum(sum( (UtM).*V ))./sum(sum( (UtU).*(V*V'))); 
  V = alpha*V; 
end

% Stopping condition depending on evolution of the iterate V : 
% Stop if ||V^{k}-V^{k+1}||_F <= delta * ||V^{0}-V^{1}||_F 
% where V^{k} is the kth iterate. 
delta = 0.1; % inner loop stopping criterion for A-HALS
eps0  = 0; 
i     = 1; % iteration counter
eps   = 1; 
while eps >= delta*eps0 && i <= maxiter %Maximum number of iterations
nodelta = 0; 
  for k = 1 : r
    deltaV = max((UtM(k,:)-UtU(k,:)*V)/UtU(k,k),-V(k,:));
    V(k,:) = V(k,:) + deltaV;
    nodelta = nodelta + deltaV*deltaV';%used to compute norm(V0-V,'fro')^2;
    if V(k,:) == 0, V(k,:) = 1e-16*max(V(:)); end % safety procedure
  end
  if (i==1)   eps0 = nodelta; end
  eps = nodelta; 
i = i + 1; 
end
end%EOF