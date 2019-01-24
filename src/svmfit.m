%  Parameters
%  training_set      - Training inputs
%  training_labels   - Training targets
%  kernel            - kernel function
%  C                 - regularization parameter


function [num_sv, beta, b0] = svmfit(training_set, training_labels, kernel, C)
    % checking if the # of arguments is correct  
	if (nargin<2 || nargin>4) 
        help svmfit 
    else
        fprintf('Classification using SVM\n')
        n = size(training_set,1);
         
	if (nargin<4) 
        C=Inf;
	end
       
	if (nargin<3) 
        kernel='linear';
	end
        
	% Constructing the kernel matrix
	H = zeros(n,n);
	for i=1:n
        for j=1:n
            H(i,j) = training_labels(i)*training_labels(j)*svmkernel(kernel,training_set(i,:),training_set(j,:));
        end
	end
	c = -ones(n,1);
        
	% Add a very small number in order to avoid problems when Hessian is badly conditioned. 
    % H = H+1e-10*eye(size(H));
        
	% Setting up the parameters for the Optimization problem
	% Set the upper and lower bounds
	lbound = zeros(n,1);  % betas >= 0
	ubound = C*ones(n,1); % betas <= C
       
    % Setting up the constraint Ax = b
    A = training_labels';
	b = 0; 
        
	% Solving the Optimization Objective via Quadratic Programming
	[beta, lambda, exitflag] = quadprog(H,c,[],[],A,b,lbound,ubound);

	fprintf('Status : %s\n', exitflag);
            
	w_length2 = beta'*H*beta;
            
	fprintf('|w0|^2 : %f\n',w_length2);
	fprintf('Margin : %f\n',2/sqrt(w_length2)); 
    fprintf('Sum beta : %f\n',sum(beta));
            
            
	% Computing the # of support vectors
	epsilon = svmtol(beta);
    svi = find(beta > epsilon);
	num_sv = length(svi);
	fprintf('Support Vectors : %d (%3.1f%%)\n',num_sv,100*num_sv/n);

        
	svii = find(beta > epsilon & beta < (C - epsilon)); 
            
	if length(svii) > 0
        b0 = (1/length(svii))*sum(training_labels(svii) - H(svii,svi)*beta(svi).*training_labels(svii));
    else
        fprintf('There are no SVs on margin. Cannot compute b0.\n');
	end
            
	end
end
   
          
        
        
        
        
        
        