function [u, v] = LucasKanade(im1, im2, wsize)

[fx, fy, ft] = ComputeDerivatives(im1, im2);
u = zeros(size(im1));
v = zeros(size(im2));
halfwin = floor(wsize/2);

for i = halfwin+1:size(fx,1)-halfwin
    for j = halfwin+1:size(fx,2)-halfwin
        cfx = fx(i-halfwin:i+halfwin,j-halfwin:j+halfwin);
        cfy = fy(i-halfwin:i+halfwin,j-halfwin:j+halfwin);
        cft = ft(i-halfwin:i+halfwin,j-halfwin:j+halfwin);
        
        cfx = cfx';
        cfy = cfy';
        cft = cft';
        
        cfx = cfx(:);
        cfy = cfy(:);
        cft = -cft(:);
        
        A = [cfx cfy];
        U = pinv(A'*A)*A'*cft;
        
        u(i,j) = U(1);
        v(i,j) = U(2);
    end;
end;

u(isnan(u)) = 0;
v(isnan(v)) = 0;


