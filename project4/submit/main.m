%% default setting
clear; close all;
% get video
video = VideoReader('./jump/shahar_jump.avi');
% video = VideoReader('./bend/shahar_bend.avi');
% video = VideoReader('./skip/shahar_skip.avi');
w = video.Width;
h = video.Height;
m = int16(video.Duration * video.FrameRate);
% calling with default method which is interframe difference
%default window is 5 frames, default threshold is 0.1

window = 5;
% window = 10;
% 
% mei = get_mei(video, "window", window, "threshold", 10, "method", "lk");


frames = get_frames(video);



mei = get_mei(video, "window", window, "threshold", 0.05);
figure;
% montage(mei,'Indices', 15:15+window-1, "size",[2 nan]);
% montage(mei,'Indices', 22:22+window-1, "size",[1 nan])
% figure;
montage(mei,'Indices', 20:20+window-1, "size",[2 nan])
figure;
montage(frames, 'Indices', 20:20+window-1, "size",[2 nan]);
% montage(frames, 'Indices', 15:15+window-1, "size",[2 nan]);
% montage(mei,'Indices', 1:10+window-1, "size",[1 nan])
implay(mei);


%% with filtering
filter = fspecial('gaussian', [3 3], 5); 
mei = get_mei(video, "window", window, "threshold", 0.05, "filter", filter);
% montage(mei,'Indices', 22:22+window-1, "size",[1 nan])
figure;
montage(mei,'Indices', 22:22+window-1, "size",[2 nan]);
implay(mei);

%% noised performance
video = VideoReader('./jump/shahar_jump.avi');


frames_n = get_frames(video);
for i = 1: size(frames_n,3)
    f = frames_n(:,:,i);
    frames_n(:,:,i) = imnoise(f, 'gaussian', 0.01);
end

mei_n = get_mei(frames_n, "threshold", 0.2);
implay(mei_n);

%% denoised performance
% store the filtered frame
frames_f = zeros(size(frames_n));
% gaussian filter of size 5*5 and std 10
filter = fspecial('gaussian', [5 5], 10); 
for i = 1:size(frames_f,3)
    f = frames_n(:,:,i);
    frames_f(:,:,i) = imfilter(f, filter, 'replicate');
end

mei_de = get_mei(frames_f, "threshold", 0.14);
montage(mei_de,'Indices', 20:20+5-1, "size",[1 nan]);
implay(mei_de);

%% morphological transformation
mei_m = zeros(h,w,m);
for i = 1:size(mei,3)
    f = mei(:,:,i);
    f = imclose(f,strel('disk',3));
    mei_m(:,:,i) = f;
end
figure;
montage(mei_m,'Indices', 20:20+5-1, "size",[1 nan]);
implay(mei_m);

%% outline using morphology operation
outline = get_outline(mei);
implay(outline);

%% shape descriptor
% matrix where each row corresponds to hu's moments in one frame
a = get_hu(VideoReader('./jump/shahar_jump.avi'));
b = get_hu(VideoReader('./bend/shahar_bend.avi'));
c = get_hu(VideoReader('./skip/shahar_skip.avi'));

% matrix M holds the averaged result for the outline sequence
M = [mean(a);mean(b);mean(c)]
% the comparison using euclidean distance
D = squareform(pdist(M))



% get hu's moment from each sequence
function [a] = get_hu(video)
    w = 5;
    mei = get_mei(video, "window", w, "threshold", 0.05);
    outline = get_outline(mei);
    a = zeros(size(outline,3), 7);
    for i = w:size(outline,3)
        a(i,:) = get_hu_vec(outline(:,:,i));
    end
end



