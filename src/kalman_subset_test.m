% num neurons
N = 400;

% time
T = 1000;

% stim
x_dim = 100;
x_sigma = 1;
X = [normrnd(0,x_sigma,T,x_dim) ones(T,1)];

% create filters
theta_sigma = 1;
theta = normrnd(0,theta_sigma,N,x_dim + 1);

% create connectivity - use a spike and slab
% sparsity = .1;
% w_sigma = 1;
% W = normrnd(0,w_sigma,N,N) .* (rand(N,N) < sparsity); 

% create connectivity so that it has only eigenvalues less than 1
eigenvals = rand(N,1);
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
n_sigma = rand(N,1)+.5;

% generate neural responses
S = zeros(N,T+1); % NOTE, FIRST INDEX is T=0
for t = 1:T

   S(:,t+1) = theta * X(t,:)' + W * S(:,t) + mvnrnd(zeros(N,1),diag(n_sigma))';

end


% CASE 1: CONSTANT SUBSET
K = ceil(N/10);
obs_neurons = 1:K;
S_obs = S(obs_neurons,2:end); % don't forget to chop off t=0 (S(t=0) = zeros)

data = [X'; S_obs];


ARmode = 1; % set to be AR linear-gaussian model
diagQ = 1;
diagR = 1;
[A, C, Q, R, initx, initV, LL] = ...
    learn_kalman(data, zeros(size([theta(1:K,:) W(1:K,1:K)])), eye(K + size(X,2)), eye(K + size(X,2)), zeros(K), zeros(K+size(X,2),1), eye(K), max_iter, diagQ, diagR, ARmode);


% NOTE TO SELF FOR PICKING BACK UP... WE ARE HAVING ISSUES WITH MAKING SURE
% ALL OF OUR PARAMS ARE THE RIGHT SIZE... ALSO, IT LOOKS WE HAVE TO FIGURE
% OUT HOW TO DEAL WITH THE FACT THAT WE KNOW THE STIMULUS AND THE STIMULUS
% IS NOT PART OF THE LINEAR DYNAMICAL SYSTEM THAT WE ARE TRYING TO INFER,
% THAT IS, THE TOP PART OF A IS ALL ZEROS - THERE IS NO INFLUENCE FROM ONE
% TIME TO THE NEXT...







