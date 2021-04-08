close all; clear;
img = imread('migong.jpg');
img = rgb2gray(img);
[T,SM] = graythresh(img);
img_b = imbinarize(img,T);
figure;
imshow(img_b);
title('maze');

%% identify the top left corner point in the maze
B1 = [0 0 0;
      0 1 1;
      0 1 0];
  
%   initially the B2 is defined as this
% B2 = [1 1 1;
%       1 0 0;
%       1 0 0];
% the modified B2 matrix
B2 = [1 1 1;
      1 0 0;
      0 0 0];
hm = imerode(img_b, B1) & imerode(~img_b, B2);
% alternative way to define the interval matrix instead
% E1 = [-1 -1 -1;-1 1 1;-1 1 0];
% hm = bwhitmiss(img_b,E1);
mat_di = strel('square',5);
figure;
hm_shifted = circshift(hm, [14 14]);
hm_shifted = imdilate(hm_shifted,mat_di);
imshow(hm_shifted);
figure;
imshow(~img_b + hm_shifted);


%% identify all the joint point in the maze
w = 30;
% E = strel('square',w);
% E1 = zeros(30,30);
% E1(5:25,:) = 1;
% E2 = zeros(30,30);
% E2(:,5:25) = 1;

E1 = zeros(40,40);
E1(10:30,:) = 1;
E2 = zeros(40,40);
E2(:,10:30) = 1;

% E1 = [0 0 0;
%       1 1 1;
%       0 0 0];
% E1 = kron(E1, ones(20,20));
% E2 = [0 1 0;
%       0 1 0;
%       0 1 0];
% E2 = kron(E2, ones(20,20));

hm = bwhitmiss(img_b,E1);
figure;
imshow(hm);
hm2 = bwhitmiss(img_b,E2);

% E3 = ones(6);
% E3 = ones(14);
E3 = strel('disk',6);
hm = imdilate(hm, E3);
hm2 = imdilate(hm2,E3);

figure;
imshow(hm);
figure;
imshow(hm2);
% hm = circshift(hm,[14 14]);
% hm2 = circshift(hm2,[14 14]);
hm3 = hm & hm2;
figure;
imshow(hm3);


% img_d = imdilate(hm2,disk);
% figure;
% imshow(img_d);
figure;
imshow(~img_b + hm3);
