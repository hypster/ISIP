close all;clear all;clc

im1 = rgb2gray(imread('test1.png'));
im2 = rgb2gray(imread('test10.png'));

wsize = 20;
[u, v] = LucasKanade(im1, im2, wsize);

q(:,:,1) = u;
q(:,:,2) = v;
