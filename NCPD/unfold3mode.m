function [Y_mode, J] = unfold3mode(Y,dims)
% 3 different unfoldings of order-3 tensor Y
if (nargin<2) dims = size(Y); end

J         =  prod(dims)./dims; % Index for n-mode matricization
Y_mode    =  cell(3);          % assign memory space
Y_mode{1} =  reshape(permute(Y,[1,3,2]),dims(1),J(1)); % mode1 unfolding
Y_mode{2} =  reshape(permute(Y,[2,3,1]),dims(2),J(2)); % mode2 unfolding
Y_mode{3} =  reshape(permute(Y,[3,2,1]),dims(3),J(3)); % mode3 unfolding
end