function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	
	%below is an easy to understand way to do it
    %tmp1 = X * theta - y; % 97*2 mult 2*1 = 97 * 1 (predicted by current theta values) less actual values
	%tmp2 = alpha / m;
	%theta(1) = theta(1) - tmp2 * (sum (tmp1 .* X(:,1)) );
	%theta(2) = theta(2) - tmp2 * (sum (tmp1 .* X(:,2)) );
	
	%below is a compact form of doing it
	theta = theta - alpha / m *  ( sum( X .* (X * theta - y) ) )';





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	%disp(sprintf('cost is %0.5f', J_history(iter)));

end

end
