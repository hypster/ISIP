close all; clear all;
[im,map] = imread('c_akai.png');
img = ind2rgb(im,map);
figure;
imshow(img)


img=im2double(img);
[h,w,c]=size(img);

% ratio=600/(h+w);
% img=imresize(img,ratio);
[h,w,c]=size(img);

wave=[20,200];  %amplitude, period
% calculate the new width and height due to the wave
newh=h+2*wave(1);
neww=w+2*wave(1);
rot=100;

img2=zeros(newh,neww,3);

for y=1:newh
    for x=1:neww
        yy=round((y-wave(1))+(wave(1)*cos(2*pi/wave(2)*x+rot )));
        xx=round((x-wave(1))+(wave(1)*cos(2*pi/wave(2)*y+rot )));

        if yy>=1 && yy<=h && xx>=1 &&xx<=w
            img2(y,x,:)=img(yy,xx,:);
        end
    end
end
    
figure;
imshow(img2,[]);
