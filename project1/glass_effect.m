close all;clear all;
[im,map] = imread('c_akai.png');
im = ind2rgb(im,map);
imshow(im);
[m,n, ~] = size(im);
imo = zeros(size(im));

% glass effect
offset = 10; %the scale of the offset
for i = 1:m
    for j = 1:n
        delta = randi([-offset,offset], 1); %the actual offset
        imo(i,j,:) = im(max(1,mod(i+delta,m)), max(1,mod(j+delta,n)),:); %use max function to prevent 0 index
    end
end
figure;
imshow(imo);



% alternative warping effect
% s = min(m,n)/2;
% [rho,theta] = meshgrid(linspace(0,s-1,s), linspace(0,2*pi));
% [x,y] = pol2cart(theta, rho);
% z = zeros(size(x));
% figure;
% warp(x, y, z, imo), view(2), axis square tight off


        
