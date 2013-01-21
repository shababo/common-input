close all;
clear all;
clc;

stream = RandStream('mt19937ar','Seed',12345);
RandStream.setDefaultStream(stream);

% add Kalman toolbox
if ispc
    addpath(genpath('C:\Users\Shababo\Projects\CommonInput\common-input\ext\Kalman'))
else
    addpath(genpath('../ext/Kalman/'))
end

%% CREATE AND RUN SYSTEM
% num neurons total
N = 1000;
% num observed neurons
K = 200;

% time
T = 100000;

% stim
stim_dim = 20; % dimension of stimulus
x_sigma = 1; % sd of stimulus
% X = [normrnd(0,x_sigma,stim_dim,T+1) ones(1,T+1)];
X = normrnd(0,x_sigma,stim_dim,T+1);
M = stim_dim + 1;



% create filters
theta_sigma = .2;
theta = normrnd(0,theta_sigma,N,M);

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
% eigenvals = (.5*rand(N,1)+.5).*sign(randn(N,1));
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

%% RUN TEST ON DIFFERENT OBSERVATION PARADIGMS

% CASE 1: CONSTANT SUBSET - OBSERVE FIRST K NEURONS
obs_matrix_one_sub = eye(M+K, M+K);
obs_matrix_one_sub = [obs_matrix_one_sub zeros(M+K, N-K)];
model = ones(1,T+1);

data_one_sub = [X; S(1:K,:)];

% [A, C, Q, R, INITX, INITV, LL] = LEARN_KALMAN(DATA, A0, C0, Q0, R0, INITX0, INITV0, MAX_ITER, DIAGQ, DIAGR, ARmode) fits
% the parameters which are defined as follows
%   x(t+1) = A*x(t) + w(t),  w ~ N(0, Q),  x(0) ~ N(init_x, init_V)
%   y(t)   = C*x(t) + v(t),  v ~ N(0, R)
% A0 is the initial value, A is the final value, etc.
ARmode = 0; % set to be AR linear-gaussian model
diagQ = 1;
diagR = 1;
max_iter = 10;
dim = N + M;
[A_onesub, C_onesub, Q_onesub, R_onesub, initx_onesub, initV_onesub, LL_onesub] = ...
    learn_kalman_subset(data, randn(dim), obs_matrix, diag([x_sigma.*ones(1,M) n_sigma']), zeros(N+M), [X(:,1); zeros(N,1)], zeros(M+N), max_iter, diagQ, diagR, ARmode, model, @enforce_constraint, M, x_sigma, obs_matrix);
%                data  A0          C0           Q0
%                                                                                   R0        initX0                initV0 
%% plot diagnostic plots
A_theta_part = A_onesub(M+1:end,1:M);
A_w_part_obs = A_onesub(M+1:M+K,M+1:M+K);
% A_w_part_unobs = A_onesub(M+1+K+1:end,M+1+K+1:end);

W_true_obs = W(1:K,1:K);
theta_true_obs = theta(1:K,1:K);

figure(1)
% scatter(A_w_part_unobs(:),'b');
scatter(A_w_part_obs(:),W_true_obs(:),'r')
xlabel('Inferred W values')
ylabel('Actual W values')
R_w = corrcoef(A_w_part_obs(:),W_true_obs(:));
title(['Connectivity Inference - 1 Subset - R_{obs} = ' num2str(R_w(2))]);

figure(2)
scatter(A_theta_part(:),theta_true_obs(:))
xlabel('Inferred \theta values')
ylabel('Actual \theta values')
R_theta = corrcoef(A_theta_part(:),theta_true_obs(:));
title(['Filter Inference - 1 Subset - R_{obs} = ' num2str(R_theta(2))]);

%% RUN TEST ON DIFFERENT OBSERVATION PARADIGMS

% CASE 2: VARYING SUBSET - OBSERVE SETS K NEURONS
num_subsets = 10;
T_per_subset = T/num_subsets;

%for now I'm going to hard code this, but we should make it more flexible
subsets = [1:20;21:40;41:60;61:80;81:100;11:30;31:50;51:70;71:90;1:10 91:100];
S_obs_idx = zeros(size(S));

for i = 1:size(subsets,1)
    
    S_obs_idx(subsets(i,:),(T_per_subset*(i-1)+1):i*T_per_subset) = 1;
    
end

% S_obs = zeros(size(S_obs));
% S_obs(logical(S_obs_idx)) = S(logical(S_obs_idx));
% 
% S_obs = [zeros(size(X)); S_obs];

data = [X; S];

% [A, C, Q, R, INITX, INITV, LL] = LEARN_KALMAN(DATA, A0, C0, Q0, R0, INITX0, INITV0, MAX_ITER, DIAGQ, DIAGR, ARmode) fits
% the parameters which are defined as follows
%   x(t+1) = A*x(t) + w(t),  w ~ N(0, Q),  x(0) ~ N(init_x, init_V)
%   y(t)   = C*x(t) + v(t),  v ~ N(0, R)
% A0 is the initial value, A is the final value, etc.
ARmode = 0; % set to be AR linear-gaussian model
diagQ = 1;
diagR = 1;
max_iter = 10;
dim = N + M;
contraint_func = @enforce_constraint;
[A_multisub, C_multisub, Q_multisub, R_multisub, initx_multisub, initV_multisub, LL_multisub] = ...
    learn_kalman(data, randn(dim), eye(dim), diag([x_sigma.*ones(1,M) n_sigma']), zeros(dim), zeros(dim,1), diag([theta_sigma.*ones(1,M) n_sigma']), max_iter, diagQ, diagR, ARmode, @enforce_constraint, M, x_sigma, S_obs_idx);
%                data  A0                 C0               Q0
% 


%% plot diagnostic plots
A_theta_part_obs = A_multisub(M+1:M+K,1:M);
A_theta_part_unobs = A_multisub(M+K+1:end,1:M);
A_w_part_obs = A_multisub(M+1:M+K,M+1:M+K);
A_w_part_unobs = A_multisub(M+K+1:end,M+K+1:end);
W_rest = W(K+1:end,K+1:end);
theta_rest = theta(K+1:end,:);

figure(3)
scatter(A_w_part_unobs(:),W_rest(:),'b');
hold on;
scatter(A_w_part_obs(:),W_true_obs(:),'r')
xlabel('Inferred W values')
ylabel('Actual W values')
R_w = corrcoef(A_w_part_obs(:),W_true_obs(:));
title(['Connectivity Inference - Multi-Subset - R_{obs} = ' num2str(R_w(2))]);

figure(4)
scatter(A_theta_part_unobs(:),theta_rest(:),'b');
hold on;
scatter(A_theta_part_obs(:),theta_true_obs(:),'r')
xlabel('Inferred \theta values')
ylabel('Actual \theta values')
R_theta = corrcoef(A_theta_part_obs(:),theta_true_obs(:));
title(['Filter Inference - Multi-Subset - R_{obs} = ' num2str(R_theta(2))]);




