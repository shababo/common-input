function Ainv = inversePD(A)

M=size(A,1);
[R b] = chol(A);
b
if b ~= 0
    return
end
R_ = R \ eye(M);
Ainv = R_ * R_';
end