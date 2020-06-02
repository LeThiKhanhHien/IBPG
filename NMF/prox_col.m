function [V,Vold,e] = prox_col(M,U,V,Vold1,r,maxiter,alpha,beta)
 % solving problem min_V ||M-UV||_F^2 by innertial block proximal method
 % which updates V row by row with extrapolation.  
 
 if nargin <= 2 || isempty(Vold1) 
    Vold1 = zeros(r,n); 
 end


 UtU = U'*U; %
 UtM = U'*M;
 delta = 0.01; % Stopping condition depending on evolution of the iterate V: 
              % Stop if ||V^{k}-V^{k+1}||_F <= delta * ||V^{0}-V^{1}||_F 
              % where V^{k} is the kth iterate. 
 eps0 = 0; cnt = 1; eps = 1; 
 Vold = Vold1; 
 while  cnt <= maxiter  && eps >= (delta)^2*eps0 % Maximum number of iterations
    
    nodelta = 0; 
    VexV = alpha*(V-Vold);%  VexV=Vex-V, Vex = V + alpha*(V-Vold); 
    Voldold = V;
    for k = 1 : r
    
     deltaV= max(( UtM(k,:)-UtU(k,:)*V + beta*VexV(k,:))/(UtU(k,k) + beta), -V(k,:) ); %Vk-Vold
   
     V(k,:) = V(k,:)+  deltaV;
     if V(k,:) == 0, 
         V(k,:) = 1e-16*max(V(:));
     end % safety procedure 
     nodelta = nodelta + deltaV*deltaV';  % used to compute norm(V0-V,'fro')^2;
    end 
    Vold=Voldold;
    if cnt == 1
        eps0 = nodelta; 
    end
    eps = nodelta; 
    cnt = cnt + 1; 
 end
    if nargout >=3
        normM = norm(M,'fro'); 
       e = sqrt(normM^2 - 2*sum(sum( UtM.*V ) ) + sum(sum( UtU.*(V*V'))))/normM; 
    end
    
end