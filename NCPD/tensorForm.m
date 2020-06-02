function [T,T3] = tensorForm(dims,U,V,W,G)
% Create order 3 tensor from given U,V,W,G as 
% T = (U*V*W)G
% === Inputs ==============================================================
% dims  - dimensions [I,J,K]
% U     - mode 1 matrix, size I X r
% V     - mode 2 matrix, size J x r
% W     - mode 3 matrix, size K x r
% G     - core tensor, size r x r x r
% If no input for G then it assumes G = I
% === Output ==============================================================
% T     - tensor with size I x J x K 
% T3    - T matricized along mode 3
%
% External function requirement : kr
%% Input handling
if nargin == 5
 hasG = 1;
else
 hasG = 0;   
end
%% main
switch hasG   
 case 0
  % Create tensor matricized along mode-3 by matrix multiplication
    T3 = W*kr(U,V)';
 case 1
  % Get dimension of core G
    dimsG = size(G);
  % reshape G for multiplcation
    G = reshape(permute(G,[3,2,1]),dimsG(3),dimsG(1)*dimsG(2));
  % Create tensor matricized along mode-3 by matrix multiplication
    T3 = W*G*kron(U,V)';
 otherwise error('Error');
end
% Fold T3 (matricized tensor) back to 3-way array
T = permute(reshape(T3,dims(3),dims(2),dims(1)),[3,2,1]);
end%EOF