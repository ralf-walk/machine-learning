function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

min_dists = realmax(size(X,1),1);

for i  = 1:K
  current_dists = sqrt( sum( (X - centroids(i,:)) .^ 2, 2) ); % calculate the eucledien distance for all x with one k
  min_dists = min(min_dists, current_dists); % set the new minimal distances
  min_positions = min_dists - current_dists == 0; % calculate the positions where this loop iterations had the lowest distances
  idx(min_positions > 0) = i; % set the idx to the centroid i where we had a lowest distance in this run
end

% =============================================================

end

