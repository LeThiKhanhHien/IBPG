function X = kr(U,varargin) % Khatri-Rao product of two matrices A and B, 
% of dimensions I-by-K and J-by-K resp. The result is an I*J-by-K matrix 
% formed by the matching columnwise Kronecker products, 
% i.e. the k-th col of the Khatri-Rao product is defined as
%   kron(A(:,k),B(:,k)).
%
%   kr(A,B,C,...) and kr({A B C ...}) compute a string of Khatri-Rao 
%   products A o B o C o ..., where o denotes the Khatri-Rao product.
%   Version: 21/10/10 , Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
if ~iscell(U), U = [U varargin]; end

K = size(U{1},2);
if any(cellfun('size',U,2)-K) error('kr:ColumnMismatch','Input matrices must have same num. of col');
end

J = size(U{end},1);
X = reshape(U{end},[J 1 K]);
for n = length(U)-1:-1:1
    I = size(U{n},1);
    A = reshape(U{n},[1 I K]);
    X = reshape(bsxfun(@times,A,X),[I*J 1 K]);
    J = I*J;
end
X = reshape(X,[size(X,1) K]);
end %EOF