%%
close all;clear;
img = imread('./homework_scanned.png');
img = rgb2gray(img);
figure;
imshow(img);
title("original image");
figure;
imhist(img);
title("histogram without filtering");

% for k=80:10:130
%     imgB = imbinarize(img,k/255);
%     figure;
%     imshow(imgB);
%     title(sprintf("threshold value %2f", k));
% end

[T,SM] = graythresh(img);
fprintf("threshold value with statistical method: %.2f\n",T*255);
figure;
imshow(imbinarize(img,T));
title("binary image without filtering");

img_f = imgaussfilt(img,1);
figure;
imshow(img_f);
title("image after filtering");
imhist(img_f);
title("histogram after filtering");

[T,SM] = graythresh(img_f);
fprintf("threshold value by statistical method(with filtering): %.2f\n",T*255);
imgB = imbinarize(img_f,T);
figure;
imshow(imgB);
title("binary image with gaussian filtering");


%%
E = strel('square',3);
img_d = imgB;
img_d = imclose(img_d, E);
figure;
imshow(img_d);
title("after close operation");

% for i = 1:5
%     img_d = imdilate(img_d, E);
%     
% end

