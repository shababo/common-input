close all;
clear all;
clc;

disp('Running...')

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
num_neurons_total = 100; % DON'T CHANGE FOR NOW
% num observed neurons at any given time (FOR NOW ONLY CHOOSE 20, 50, 80,
% OR 100)
num_neurons_obs = 80; 
% for now please ensure obs/total evenly
% if mod(num_neurons_total,num_neurons_obs) ~= 0
%     error('num_neurons_obs does not evenly divide num_neurons_total... sorry, but it is simplier to enforce this for now...')
% end
% timesteps (not including t = 0 which has all neural responses at 0)
T = 10000;
% do we have a baseline parameter for each neuron
have_baseline = 0;

% build stimuli % FOR NOW KEEP STIM_DIM = 0!! Will add this functionality
% in next day or two
stim_dim = 0; % dimension of stimulus
stim_sigma = 1; % sd of stimulus
% sample stimuli
state_stimuli = normrnd(0,stim_sigma,stim_dim,T+1); % we use T+1 to accont for t=0

% create stimulus filters for each neuron
stim_filter_sigma = .2;
stim_filter = normrnd(0,stim_filter_sigma,num_neurons_total,stim_dim);

% BELOW WE HAVE THREE METHODS FOR CREATING THE WEIGHT MATRICES

% 1) create connectivity - use a spike and slab - NOTE THAT I'M NOT SURE IF
% THIS METHOD ENSURES GOOD SYSTEM DYNAMICS
% sparsity = .1;
% w_sigma = 1;
% W = normrnd(0,w_sigma,num_neurons_total,num_neurons_total) .* (rand(num_neurons_total,num_neurons_total) < sparsity); 

% 2) create connectivity so that it has only eigenvalues less than 1 to
% ensure good system dynamics, but not sparse
eigenvals = (0.5*rand(num_neurons_total,1)+0.5) .* sign(randn(num_neurons_total,1));
D = diag(eigenvals);
V = orth(randn(num_neurons_total));
connectivty_matrix = V*D*V';

% create connectivy with all eigvals < 1 AND sparse - this method should
% both give us sparsity and good system dynamics
% sparsity = .1;
% eigenvals = (.5*rand(num_neurons_total,1)+.5).*sign(randn(num_neurons_total,1));
% W = triu(sprand(num_neurons_total,num_neurons_total,sparsity),1) + spdiags(eigenvals,0,num_neurons_total,num_neurons_total);
% for i = 1:num_neurons_total
%     W = rjr(W);
% end

% add baseline
% use_baseline = 0;
% if use_baseline
%     baseline_sigma = .25;
%     baseline = normrnd(0,baseline_sigma,num_neurons_total,1);
% else
%     baseline = [];
% end

% noise std dev for each neuron's state
state_neurons_sigma = .5*rand(num_neurons_total,1)+.5;

% generate neural responses
state_neurons = zeros(num_neurons_total,T+1); % NOTE, FIRST INDEX is T=0
for t = 1:T

   state_neurons(:,t+1) = stim_filter * state_stimuli(:,t) + connectivty_matrix * state_neurons(:,t) + mvnrnd(zeros(num_neurons_total,1),diag(state_neurons_sigma))';

end

state = [state_stimuli; state_neurons];
state_dim = size(state,1);
    
%% RUN TEST ON DIFFERENT OBSERVATION PARADIGMS

% CASE 1: CONSTANT SUBSET - OBSERVE FIRST SUBSET OF NEURONS FOR WHOLE
% EXPERIMENT

% build observation matrix and generate observations
observation_dim = stim_dim+num_neurons_obs;
obs_matrix_one_sub = [eye(observation_dim) zeros(observation_dim, num_neurons_total-num_neurons_obs)];
subset_index = ones(1,T+1); % we only have one obeservation matrix so this is just all ones
observation_sigma = 1e-4;
observations_one_sub = obs_matrix_one_sub * state + normrnd(0,observation_sigma,observation_dim,T+1);

%% RUN KALMAN LEARNING

% below is a section of comments from learn_kalman and kalman_filter so you
% can see how they work and how we set up parameters for a changing
% observation matrix (this will only come into play for the next part where
% we change our observed subset

% for more info on these function see m-files

% [A, C, Q, R, INITX, INITV, LL] = LEARN_KALMAN_SUBSET(DATA, A0, C0, Q0, R0, INITX0, INITV0, MAX_ITER, DIAGQ, DIAGR, ARmode) fits
% the parameters which are defined as follows
%   x(t+1) = A*x(t) + w(t),  w ~ num_neurons_total(0, Q),  x(0) ~ num_neurons_total(init_x, init_V)
%   y(t)   = C*x(t) + v(t),  v ~ num_neurons_total(0, R)
% A0 is the initial value, A is the final value, etc.

% OPTIONAL INPUTS (string/value pairs [default in brackets])
% 'model' - model(t)=m means use params from model m at time t [ones(1,T) ]
%     In this case, all the above matrices take an additional final dimension,
%     i.e., A(:,:,m), C(:,:,m), Q(:,:,m), R(:,:,m).
%     However, init_x and init_V are independent of model(1).


ARmode = 0; % set to be AR linear-gaussian model
diagQ = 1;
diagR = 1;
max_iter = 5;

state_matrix_0 = randn(state_dim);
state_cov_0 = diag([stim_sigma.*ones(stim_dim,1) state_neurons_sigma']);
obs_matrix_0 = obs_matrix_one_sub;
obs_cov_0 = observation_sigma*eye(observation_dim);
posterior_state_mean_0 = state(:,1);
posterior_state_cov_0 = observation_sigma*eye(state_dim); 

[state_matrix_learned_one_sub, obs_matrix_learned_one_sub, state_cov_learned_one_sub, obs_cov_learned_one_sub, init_state_one_sub, init_state_cov_one_sub, LL_one_sub] = ...
    learn_kalman_subset(observations_one_sub, state_matrix_0, obs_matrix_0, state_cov_0, obs_cov_0,...
            posterior_state_mean_0, posterior_state_cov_0,...
            max_iter, diagQ, diagR, ARmode, subset_index, @enforce_learning_constraint, obs_matrix_one_sub);


%%

% CASE 2: VARYING SUBSET - OBSERVE SETS num_neurons_obs NEURONS

% below I've once again pasted the comments for using a model that varies
% with time - in our case only the observation matrix has this property

% OPTIONAL INPUTS (string/value pairs [default in brackets])
% 'model' - model(t)=m means use params from model m at time t [ones(1,T) ]
%     In this case, all the above matrices take an additional final dimension,
%     i.e., A(:,:,m), C(:,:,m), Q(:,:,m), R(:,:,m).
%     However, init_x and init_V are independent of model(1).

num_subsets = 10;
%again, we want to make sure we divide evenly into T to keep things simple
%for now
if mod(T,num_subsets) ~= 0
    error('Please choose a subset size that divides evenly into T')
end
T_per_subset = T/num_subsets;

%for now I'm going to hard code this, but we should make it more flexible
switch num_neurons_obs
    case 20
        subsets = [1:20;21:40;41:60;61:80;81:100;11:30;31:50;51:70;71:90;1:10 91:100];
    case 50
        subsets = [1:50;11:60;21:70;31:80;41:90;51:100;1:10 61:100;1:20 71:100;1:30 81:100;1:40 91:100];
    case 80
        subsets = [1:80;21:100;1:20 41:100;1:40 61:100;1:60 81:100;11:90;1:10 31:100;1:30 51:100;1:50 71:100;1:70 91:100];
    case 100
        subsets = repmat(1:100,10,1);
end

obs_matrix_multi_sub = zeros(observation_dim,state_dim,num_subsets);
observations_multi_sub = zeros(observation_dim,T+1);
subset_index_multi_sub = zeros(T,1);

for i = 1:size(subsets,1)
    
    obs_matrix_multi_sub(1:stim_dim,1:stim_dim,i) = eye(stim_dim);
    obs_matrix_multi_sub(stim_dim + 1:end,stim_dim + subsets(i,:),i) = eye(num_neurons_obs);
    subset_index_multi_sub((i-1)*T_per_subset+1:i*T_per_subset) = i;
    observations_multi_sub(:,(i-1)*T_per_subset+1:i*T_per_subset) = obs_matrix_multi_sub(:,:,i) * state(:,(i-1)*T_per_subset+1:i*T_per_subset) + normrnd(0,observation_sigma,observation_dim,T_per_subset);
    
end

%%
% setup parameters for learning
ARmode = 0;
diagQ = 1;
diagR = 1;
max_iter = 5;

% setup initial conditions
state_matrix_0 = repmat(randn(state_dim),[1 1 num_subsets]);
state_cov_0 = repmat(diag([stim_sigma.*ones(stim_dim,1) state_neurons_sigma']),[1 1 num_subsets]);
obs_matrix_0 = obs_matrix_multi_sub;
obs_cov_0 = repmat(observation_sigma*eye(observation_dim),[1 1 num_subsets]);
posterior_state_mean_0 = state(:,1);
posterior_state_cov_0 = observation_sigma*eye(state_dim); 


[state_matrix_learned_multi_sub, obs_matrix_learned_multi_sub, state_cov_learned_multi_sub, obs_cov_learned_multi_sub, init_state_multi_sub, init_state_cov_multi_sub, LL_multi_sub] = ...
    learn_kalman_subset(observations_multi_sub, state_matrix_0, obs_matrix_0, state_cov_0, obs_cov_0,...
            posterior_state_mean_0, posterior_state_cov_0,...
            max_iter, diagQ, diagR, ARmode, subset_index, @enforce_learning_constraint, obs_matrix_multi_sub);







