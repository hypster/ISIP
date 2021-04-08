% get grayscale frames from video
function [frames] = get_frames(video)
    if ~isa(video, 'VideoReader')
        if length(size(video)) == 3
            frames = video; 
            return;
        else
            error("error: must be video type");
        end
    end

    w = video.Width;
    h = video.Height;
    m = int16(video.Duration * video.FrameRate);
    frames = zeros(h,w,m);
    for i = 1:m
        frames(:,:,i) = im2double(rgb2gray(read(video, i)));
    end
end