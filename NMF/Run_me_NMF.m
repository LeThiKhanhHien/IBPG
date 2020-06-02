% This files allows to run a comparison of the NMF algorithms on synthetic
% data sets, as discussed in
% Inertial Block Mirror Descent Method for non-convex non-smooth
% optimization, Le Thi Khanh Hien, Nicolas Gillis and Panos Patrinos, ICML 2020.
clear all; close all; clc;
% Options and parameters
options.maxiter =inf;
options.timemax =3;
options.display=0;

% Dimensions of input matrix and rank
m = 200;
n = 200;
r = 20;
% Generation of synthetic data set
Vtrue = rand(r,n);
Utrue = rand(m,r);
X = Utrue*Vtrue;
% Initialization
U0 = rand(m,r);
V0 = rand(r,n);
% initial
options.init.W = U0;
options.init.H = V0;
% Run NMF Algorithms
% A-HALS
disp('Running A-HALS (Gillis & Glineur)...')
options.beta0 = 0;
options.NMFalgo = 'HALS';
[W,H,e0,t,e] = NMFextrapol(X,r,options);
fprintf('Final relative error of A-HALS   = %2.2d\n', e(end));
vAHALS=e;
tAHALS=t;
lAHALS=length(e);
% Andersen E-A-HALS (hp=3)
disp('Running E-A-HALS (Ang & Gillis)...');
options.beta0 = 0.5;
options.gammabeta = 1.01;
options.gammabetabar = 1.005;
options.extrapolprojH = 3;
options.NMFalgo = 'HALS';
[W,H,e0,t,e] = NMFextrapol(X,r,options);
fprintf('Final relative error of E-A-HALS = %2.2d\n', e(end));
vAnd=e;
tAnd=t;
% XuYin
disp('Running APGC (Xu & Yin, 2013)...');
[W,H,e,t] = APGC(X,r,options);
fprintf('Final relative error of APGC    = %2.2d\n', e(end));
vXuYin=e;
tXuYin=t;
% Inertial matrix proximal gradient method with repeating update
disp('Running IBPG_A ...');
[W,H,e,t] = IBPG_A(X,r,options);
fprintf('Final relative error of IBPG-A     = %2.2d\n', e(end));
vIBPG_A=e;
tIBPG_A=t;
% Inertial column wise proximal method with repeating update
disp('Running IBP...');
[W,H,e,t] = IBP(X,r,0.001,1.01,0.6,options);
fprintf('Final relative error of IBP      = %2.2d\n', e(end));
vIBP=e;
tIBP=t;
% iMPG cyclic
disp('Running IBPG cyclic...');
options.inneriter=1;
[W,H,e,t] = IBPG(X,r,options);
fprintf('Final relative error of IBPG = %2.2d\n', e(end));
vIBPG_cyclic=e;
tIBPG_cyclic=t;
% iPALM
disp('Running iPALM ...');
[W,H,e,t] = iPALM(X,r,options);
fprintf('Final relative error of iPALM = %2.2d\n', e(end));
viPALM=e;
tiPALM=t;

% Remove smallest value among relative errors obtained
vmin=min([min(viPALM),min(vAHALS),min(vAnd),min(vIBPG_A),min(vIBP),min(vXuYin),min(vIBPG_cyclic)]);
vAHALS=vAHALS-vmin;
vAnd=vAnd-vmin;
vIBPG_A=vIBPG_A-vmin;
vIBP=vIBP-vmin;
vXuYin=vXuYin-vmin;
vIBPG_cyclic=vIBPG_cyclic-vmin;
viPALM=viPALM-vmin;
% Display results
set(0, 'DefaultAxesFontSize', 18);
set(0, 'DefaultLineLineWidth', 2);
figure;
surf(peaks);
grid('on');
semilogy(tAHALS,vAHALS ,'g-.');hold on; %A-HALS
semilogy(tAnd,vAnd,'b-.');hold on; %E-A-HALS
semilogy(tXuYin,vXuYin,'c-.');hold on; %XuYin
semilogy(tIBPG_A,vIBPG_A,'r--');hold on; %iMPG
semilogy(tIBP,vIBP,'k--');hold on;   %iCP
semilogy(tIBPG_cyclic,vIBPG_cyclic,'m--');hold on; %iMPG-cyclic
semilogy(tiPALM,viPALM,'y--');hold on; 
legend('A-HALS','E-A-HALS','APGC','IBPG-A','IBP','IBPG','iPALM');
ylabel('||X-WH||_F / ||X||_F -e_{min}');
xlabel('Time (s.)'); 