function f_x = multivar_least_squares_error(connectivity_vec, X)

num_neurons = sqrt(length(connectivity_vec));
connectivity_matrix = reshape(connectivity_vec,num_neurons,num_neurons);
f_x = trace((X(:,2:end) - connectivity_matrix*X(:,1:end-1))'*(X(:,2:end) - connectivity_matrix*X(:,1:end-1)));
