function [frames] = filter_frames(frames, filter)
    for i = 1:size(frames,3)
        f = frames(:,:,i);
        frames(:,:,i) = imfilter(f, filter, 'replicate');
    end
end