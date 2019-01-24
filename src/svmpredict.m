%  Parameters
%  training_set     - Training inputs
%  training_labels  - Training targets
%  test_set         - Test inputs
%  kernel           - kernel function
%  beta             - Lagrange Multipliers
%  b0               - bias

function predictions = svmpredict(training_set,training_labels,test_set,kernel,beta,b0)

n = size(training_set,1);
m = size(test_set,1);
H = zeros(m,n);  
    
for i=1:m
	for j=1:n
        H(i,j) = training_labels(j)*svmkernel(kernel,test_set(i,:),training_set(j,:));
	end
end

% Maximum (Hard) Margin
predictions = sign(H*beta + b0);

end
