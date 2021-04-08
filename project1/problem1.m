close all; clear all; clc
im_building = imread('expo_building_gray.jpg'); %for
% im_building = rgb2gray(imread('lolipop.jpg'));

% simplicity for analyzing results, I use gray image
% imwrite(rgb2gray(im_building), 'expo_building_gray.jpg'); 
imd = im2double(im_building);


% imd = im2double(rgb2gray(im_building));

% imd = imnoise(imd, 'gaussian', 0, 0.01); %apply gaussian noise of 0.01 
% imd = imnoise(imd, 'salt & pepper', 0.05); 
% imshow(imd)

% average_filter = ones(3)/(3^2); %apply average filter
% imd = imfilter(imd, average_filter);

% apply median filter
% imd = medfilt2(imd);



% for i = 1:size(imd,3)
%     I = imd(:,:,i);
%     edge_detection(I);
% end
% function [] = edge_detection(imd)

[n1 n2]= size(imd);
%standard filter
y1 = edge(imd,'prewitt');
y2 = edge(imd,'sobel');
y3 = edge(imd,'canny');
y4 = edge(imd, 'log');
figure; 
subplot(3,2,1); 
imshow(imd); 
title('original');
subplot(3,2,2);
imshow(y1); title('prewitt')
subplot(3,2,3); 
imshow(y2); 
title('sobel'); 
subplot(3,2,4); 
imshow(y3);
title('canny');
subplot(3,2,5);
imshow(y4);
title('log');

%custom kernel
sober_horizontal = [-1 -2 -1;0 0 0; 1 2 1];
sober_vertical = [-1 0 1;-2 0 2; -1 0 1];
sober_diagonal_1 = [0 1 2;-1 0 1;-2 -1 0];
sober_diagonal_2 = [-2 -1 0;-1 0 1;0 1 2];

figure;
subplot(221)
sh = imfilter(imd, sober_horizontal);
imshow(abs(sh));
title('sober horizontal');

subplot(222);
sv = imfilter(imd, sober_vertical);
imshow(abs(sv));
title('sober vertical');

subplot(223);
sd1 = imfilter(imd, sober_diagonal_1);
imshow(abs(sd1));
title('sober diagonal 1');

subplot(224);
sd2 = imfilter(imd, sober_diagonal_2);
imshow(abs(sd2));
title('sober diagonal 2');

figure;
% subplot(211);
y_m = abs(sv) + abs(sh);
imshow(y_m);
title('sober gradient(x-y)');
% 
% figure;
% imshow(im2bw(y_m, 0.35)); %binary image

% second order filter
laplacian = [1 1 1;1 -8 1; 1 1 1];
y_l = imfilter(imd, laplacian);
figure;
imshow(mat2gray(y_l));

% 
% 
% 
% 
% 
