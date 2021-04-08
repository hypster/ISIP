%% read image and blur
close all; clear;
f= imread('expo_building_gray.jpg');

figure;
imshow(f);
title("original");
[m,n]=size(f);

h=fspecial('gaussian',5,2); %gaussian spatial filter
g=imfilter(f,h,'circular'); %filter on spatial domain
figure;
imshow(g);
title("with blur");
%% add gaussian noise
% noise = uint8(randn(m,n));
g_only_blur = g;
g= imnoise(g, 'gaussian');
figure;
imshow(g); 
title("with blur and noise");

%% power spectrum

G = fftshift(fft2(g)); 
figure;
P = abs(G).^2;
imagesc(log(1+P));
title("power spectrum");

[x,y] = meshgrid(1:n, 1:m);
figure;
mesh(x,y,log(1+P));
title("3d power");
view(10,20);
axis off;

H = fftshift(fft2(h,size(G,1), size(G,2)));
%% spectrum in 1d

figure;
plot(1: size(P,2), log(1+abs(P(size(P,1)/2+1, :))));
title("spectrum along u = m/2+1 axis");

%%
imagesc(log(1+abs(H)));
title("spectrum of frequency response");



%% ideal inverse filter, best value 150
% R=100:10:200 %the cutoff value
R = 150;
for i = 1:numel(R)
    F = zeros(size(f)); %pixels far away are 0 by default
    r = R(i);
    for v=1:size(f,1)
        for u=1:size(f,2)
            du = u - size(f,2)/2;
            dv = v - size(f,1)/2;
            if du^2 + dv^2 <= r^2
                F(v,u) = G(v,u)./H(v,u);
            end
        end
    end
figure;
imshow(abs(ifft2(F)),[]);
title(sprintf('restored image with cutoff = %d', r));
figure;imshow(log(abs(F)+1),[]);
end
% figure;
% imshow(log(abs(F)),[]);
% f_h = abs(ifft2(F));
%% butterworth inverse filter, best value 140
% R=100:10:200 %the cutoff value
R = 140;
for i = 1:numel(R)
    F = zeros(size(f)); %pixels far away are 0 by default
    r = R(i);
    for v=1:size(f,1)
        for u=1:size(f,2)
            du = u - size(f,2)/2;
            dv = v - size(f,1)/2;
            d = sqrt(du^2+dv^2);
            F(v,u) = G(v,u)./H(v,u);
            F(v,u) = F(v,u) /(1 + (d/r)^(2*10));
        end
    end
    figure;
    imshow(abs(ifft2(F)),[]);
    title(sprintf('restored image with cutoff = %d', r));
    figure;imshow(log(abs(F)+1),[]);
end

%%
bar(g,n,m);
bar(f,n,m);
function[] = bar(g,n,m)
G = fftshift(fft2(g)); 
figure;
P = abs(G).^2;
imagesc(log(1+P));
title("power spectrum");

[x,y] = meshgrid(1:n, 1:m);
figure;
mesh(x,y,log(1+P));
title("3d power");
view(10,20);
axis off;

% H = fftshift(fft2(h,size(G,1), size(G,2)));

end

