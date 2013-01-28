function [state_matrix,true_obs_matrix,state_cov,obs_cov,posterior_state_mean_0,posterior_state_cov_0] = ...
        enforce_learning_constraint(state_matrix,obs_matrix,state_cov,obs_cov,posterior_state_mean_0,posterior_state_cov_0,true_obs_matrix)

% this function enforces any contraints on the learned parameters, in other
% words it allows us to set parameters we know the values of... this is 
% done by replacing the arguments at the end on the input arguments 
% into the appropriate output arguments - look closely at inputs vs.
% outputs to see how this works