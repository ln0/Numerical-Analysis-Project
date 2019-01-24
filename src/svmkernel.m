%  Parameters 
%  kernel       - kernel type ('linear', 'poly', 'gauss', 'sigmoid')
%  xi, xj       - kernel vector arguments

function k = svmkernel(kernel,xi,xj)
	global p;
	switch lower(num2str(kernel))
        case 'linear'
            k = xi*xj';
        case 'poly'
            k = (xi*xj'+1)^p;
        case 'gauss'
            k = exp(-(xi-xj)*(xi-xj)'/(2*p^2));
        otherwise
            k = xi*xj'; % the same linear kernel if no valid kernel is specified
	end
end
