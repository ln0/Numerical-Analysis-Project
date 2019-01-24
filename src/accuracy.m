%  Parameters
%  test_labels   - Test targets
%  predictions   - Our predictions on test set

function acc = accuracy(test_labels,predictions)
acc=0;
n=length(test_labels);
for i=1:n
    if test_labels(i)==predictions(i)
        acc=acc+1;
    end
end
acc=acc/n*100;
end