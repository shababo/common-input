X = rand(10);

XtX_all = X'*X;

XtX_iter = zeros(10);

for i = 1:10
    
    XtX_iter = XtX_iter + X(i,:)'*X(i,:);
    
end

XtX_iter

XtX_all - XtX_iter