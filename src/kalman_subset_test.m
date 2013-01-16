
% add Kalman toolbox
if ispc
    addpath(genpath('C:\Users\Shababo\Projects\CommonInput\common-input\ext\Kalman'))
else
    addpath(genpath('../ext/Kalman/'))
end


% num neurons
N = 100;

% time
T = 10000;

% stim
M = 20; % dimension of stimulus
x_sigma = 1;
% X = [normrnd(0,x_sigma,T,M) ones(T,1)];
% X = [rand(M,T); ones(1,T)];
X = rand(M,T+1);


% create filters
theta_sigma = .2;
theta = normrnd(0,theta_sigma,N,size(X,1));

% create connectivity - use a spike and slab
% sparsity = .1;
% w_sigma = 1;
% W = normrnd(0,w_sigma,N,N) .* (rand(N,N) < sparsity); 

% create connectivity so that it has only eigenvalues less than 1
eigenvals = (0.5*rand(N,1)+0.5) .* sign(randn(N,1));
D = diag(eigenvals);
V = orth(randn(N));
W = V*D*V';

% create connectivy with all eigvals < 1 AND sparse
% sparsity = .1;
% eigenvals = rand(N,1);
% W = triu(sprand(N,N,sparsity),1) + spdiags(eigenvals,0,N,N);
% for i = 1:N
%     W = rjr(W);
% end


% noise std dev for each neuron
n_sigma = .5*rand(N,1)+.5;

% generate neural responses
S = zeros(N,T+1); % NOTE, FIRST INDEX is T=0
for t = 1:T

   S(:,t+1) = theta * X(:,t) + W * S(:,t) + mvnrnd(zeros(N,1),diag(n_sigma))';

end


% CASE 1: CONSTANT SUBSET
% K = ceil(N/10);
K = N;
obs_neurons = 1:K;
S_obs = S(obs_neurons,2:end); % don't forget to chop off t=0 (S(t=0) = zeros)

data = [X; S];

% [A, C, Q, R, INITX, INITV, LL] = LEARN_KALMAN(DATA, A0, C0, Q0, R0, INITX0, INITV0, MAX_ITER, DIAGQ, DIAGR, ARmode) fits
% the parameters which are defined as follows
%   x(t+1) = A*x(t) + w(t),  w ~ N(0, Q),  x(0) ~ N(init_x, init_V)
%   y(t)   = C*x(t) + v(t),  v ~ N(0, R)
% A0 is the initial value, A is the final value, etc.
ARmode = 1; % set to be AR linear-gaussian model
diagQ = 1;
diagR = 1;
max_iter = 10;
kalman_dim = K + size(X,1);
contraint_func = @enforce_constraint;
[A, C, Q, R, initx, initV, LL] = ...
    learn_kalman(data, randn(kalman_dim), eye(kalman_dim), diag([theta_sigma.*ones(1,size(X,1)) n_sigma']), zeros(kalman_dim), zeros(kalman_dim,1), diag([theta_sigma.*ones(1,size(X,1)) n_sigma']), max_iter, diagQ, diagR, ARmode, @enforce_constraint, size(X,1), x_sigma);
%                data  A0                 C0               Q0
%                                                                                                          R0                   initX0           initV0 
%% plot diagnostic plots
A_theta_part = A(size(X,1)+1:end,1:size(X,1));
A_w_part = A(size(X,1)+1:end,size(X,1)+1:end);

figure(1)
scatter(1:numel(W),W(:))

figure(2)
scatter(A_w_part(:),W(:))

figure(3)
scatter(A_theta_part(:),theta(:))




