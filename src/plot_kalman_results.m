function plot_kalman_results(varargin)

% load a passed file name/path or if empty just load in the current
% workspace
if ~isempty(varargin)
    load(vargin{1})
else
    evalin('base','save temp_current_workspace.mat')
    load('temp_current_workspace.mat')
end

%%

connectivity_learned_one_sub = state_matrix_learned_one_sub(stim_dim+1:end,stim_dim+1:end);
connectivity_obs_learned_one_sub = connectivity_learned_one_sub(1:num_neurons_obs,1:num_neurons_obs);
connectivity_obs_true_one_sub = connectivity_matrix(1:num_neurons_obs,1:num_neurons_obs);

figure(1)
scatter(connectivity_obs_learned_one_sub(:),connectivity_obs_true_one_sub(:),'r')
xlabel('Inferred Connection Weights')
ylabel('Actual Connection Weights')
R_w = corrcoef(connectivity_obs_learned_one_sub(:),connectivity_obs_true_one_sub(:));
title(['Connectivity Inference - 1 Subset (size = ' num2str(num_neurons_obs) ') - R_{obs} = ' num2str(R_w(2))]);

if stim_dim ~= 0
    
    stim_filters_obs_learned_one_sub = state_matrix_learned_one_sub(stim_dim+1:stim_dim+1+num_neurons_obs,1:stim_dim);
    stim_filters_obs_true_one_sub = stim_filter(1:num_neurons_obs,:);
    figure(2)
    scatter(stim_filters_obs_learned_one_sub(:),stim_filters_obs_true_one_sub(:))
    xlabel('Inferred Filter Values')
    ylabel('Actual Filter Values')
    corr_coef_filters = corrcoef(stim_filters_obs_learned_one_sub(:),stim_filters_obs_true_one_sub(:));
    title(['Filter Inference - 1 Subset (size = ' num2str(num_neurons_obs) ') - R = ' num2str(corr_coef_filters(2))]);
    
end


connectivity_learned_multi_sub = state_matrix_learned_multi_sub(stim_dim+1:end,stim_dim+1:end,1);
connectivity_obs_learned_multi_sub = connectivity_learned_multi_sub(1:num_neurons_obs,1:num_neurons_obs); % this is the same group we observed in the first part of the experiment

figure(3)
scatter(connectivity_learned_multi_sub(:),connectivty_matrix(:),'b')
hold on
scatter(connectivity_obs_learned_multi_sub(:), connectivity_obs_true_one_sub(:),'r')
xlabel('Inferred Connection Weights')
ylabel('Actual Connection Weights')
R_w = corrcoef(connectivity_obs_learned_multi_sub(:),connectivity_obs_true_one_sub(:));
title(['Connectivity Inference - Multi Subset (size = ' num2str(num_neurons_obs) ') - R_{obs} = ' num2str(R_w(2))]);

if stim_dim ~= 0
    
    stim_filters_obs_learned_multi_sub = state_matrix_learned_multi_sub(stim_dim+1:stim_dim+1+num_neurons_obs,1:stim_dim);
    stim_filters_obs_true_one_sub = stim_filter(1:num_neurons_obs,:);
    figure(5)
    scatter(stim_filters_obs_learned_one_sub(:),stim_filters_obs_true_one_sub(:))
    xlabel('Inferred Filter Values')
    ylabel('Actual Filter Values')
    corr_coef_filters = corrcoef(stim_filters_obs_learned_one_sub(:),stim_filters_obs_true_one_sub(:));
    title(['Filter Inference - Multi Subset (size = ' num2str(num_neurons_obs) ') - R = ' num2str(corr_coef_filters(2))]);
    
end

%%

delete('temp_current_workspace.mat')