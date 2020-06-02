clear all; 
close all; 
clc; 
%% Experimental set up
dims = [100,100,100]; % size of order 3 tensor 
r = 20;            % factorization rank
options.timemax = 5;  % run time maximum in second (suggested : 5,10,30,60,...)
options.itermax = inf; %


% generate data
I = dims(1); 
J = dims(2); 
K = dims(3);

Utrue = rand(I,r);
Vtrue = rand(J,r);
Wtrue = rand(K,r);

Ttrue = tensorForm(dims,Utrue,Vtrue,Wtrue);

options.init.U=rand(I,r);
options.init.V=rand(J,r);
options.init.W=rand(K,r);

% run IBPG
disp('Running IBPG-A...')
[U,V,W,e,t] = IBPG(Ttrue,r,options);

 e_IBPG=e;
 t_IBPG=t;
 l_IBPG=length(e);

% run APG
disp('Running APGC...')
[U,V,W,e,t] = APG(Ttrue,r,options);
e_APG=e;
t_APG=t;
l_APG=length(e);

 

% run ADMM
disp('Running ADMM...')
[U,V,W,e,t] = ADMM(Ttrue,r,options);
 e_ADMM=e;
 t_ADMM=t;
 l_ADMM=length(e);


% run AHALS 
disp('Running A-HALS...')
[U,V,W,e,t] = AHALS(Ttrue,r,options);
e_AHALS=e;
t_AHALS=t;
l_AHALS=length(e);

% 
% Remove smallest value among relative errors obtained
vmin=min([min(e_IBPG),min(e_ADMM),min(e_AHALS),min(e_APG)]);
e_IBPG=e_IBPG-vmin;
e_ADMM=e_ADMM-vmin;
e_AHALS=e_AHALS-vmin;
e_APG=e_APG-vmin;

semilogy(t_IBPG,e_IBPG,'r','linewidth',3),hold on
semilogy(t_APG,e_APG,'c','linewidth',2.2),hold on;
semilogy(t_ADMM,e_ADMM,'k','linewidth',2.2),hold on;
semilogy(t_AHALS,e_AHALS,'b','linewidth',2.2),hold on;

    
 legend('IBPG-A','APGC','ADMM','A-HALS');
 ylabel('||X-U.V.W||_F / ||X||_F -e_{min}');
 xlabel('Time (s.)')

