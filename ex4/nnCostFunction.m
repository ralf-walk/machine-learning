function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % 25 x 401
Theta2_grad = zeros(size(Theta2)); % 10 x 26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ######## Calculate Calculate h_of_x (the hypothesis) with forward propagation ##########
a_1 = X;
a_1_w_bias = [ones(m, 1) X];
z_2 = a_1_w_bias * Theta1';
a_2 = sigmoid(z_2);

a_2_w_bias = [ones(size(a_2, 1), 1) a_2];
z_3 = a_2_w_bias * Theta2';
a_3 = sigmoid(z_3); % = h_of_x, is a matrix with size 5000 x 10

% ######## Calculate y #########
y_recoded = zeros(m,num_labels);
for i = 1:m
  y_recoded(i, y(i)) = 1;
end

% ######## Calculate the Costs ########
pos = -y_recoded .* log(a_3);
neg = -(1 - y_recoded) .* log(1 - a_3);
all = pos + neg;

J = sum(all(:)) / m;

% ######## add regularization to the costs #######
sum_theta_1 = sum(Theta1(:,[2:end])(:).^2);
sum_theta_2 = sum(Theta2(:,[2:end])(:).^2);
J = J + (lambda / (2 * m)) * (sum_theta_1 + sum_theta_2);

% ######## Calculate the gradients with backpropagation ######

% calculate the deltas for the output layer
delta_3 = a_3 - y_recoded; % 5000 x 10

% calculate the deltas for the hidden layer
weighted_deltas = (delta_3 * Theta2); % 5000 x 26
weighted_deltas = weighted_deltas(:,[2:end]); % 5000 x 25
delta_2 =  weighted_deltas .* sigmoidGradient(z_2); % 5000 x 25

% loop over every training example and update big_delta
big_delta_1 = zeros(size(Theta1_grad)); % 25 x 401
big_delta_2 = zeros(size(Theta2_grad)); % 10 x 26

big_delta_1 = big_delta_1 + (delta_2' * a_1_w_bias); % = 25 x 5000 * 5000 x 401 = 25 x 401
big_delta_2 = big_delta_2 + (delta_3' * a_2_w_bias); % = 5000 x 10 * 5000 x 26 = 10 x 26

% compute the gradients with regularization
Theta1_grad = big_delta_1 / m;
Theta1_grad(:,[2:end]) = Theta1_grad(:,[2:end]) + ((lambda * Theta1(:,[2:end])) / m);

Theta2_grad = big_delta_2 / m;
Theta2_grad(:,[2:end]) = Theta2_grad(:,[2:end]) + ((lambda * Theta2(:,[2:end])) / m);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
