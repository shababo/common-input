function [A,C,Q,R,initx,initV] = enforce_constraint(A,C,Q,R,initx,initV,M,x_sigma,obs_matrix)

A(1:M,:) = 0;
Q(1:M,1:M) = diag(ones(M,1)*x_sigma);
C = obs_matrix;
R = zeros(size(R));