%%%%%%%%
% for motion detection, default method is the interframe difference
function [mei] = get_mei(video, varargin)
    frames = get_frames(video);
    [h,w,m] = size(frames);
    
    %threshold to reduce the background noise 
    t = 0.05;
    % size of window
    window = 5;
    %default filter
    
    p = inputParser;
    addParameter(p, "method", "difference");
    addParameter(p, "threshold", t);
    addParameter(p, "window", window);
    addParameter(p, "filter", false);
    parse(p,varargin{:});
    method = p.Results.method;
    t = p.Results.threshold;
    window = p.Results.window;
    filter = p.Results.filter;
    
    % apply filter if specified     
    if filter
        frames = filter_frames(frames,filter);
    end
    
    % difference: boolean matrix indicates whether there's motion between
    % frames
    diff = zeros(h,w,m);

    
    if method == "difference"
        for i = 2:m
            curr = abs(frames(:,:,i) - frames(:,:,i-1));
            % if absolute difference greater than threshold set to 1     
            curr = (curr > t);
            diff(:,:,i) = curr;
        end
    elseif method == "lk"
        for i = 2:m
            curr = LucasKanade(frames(:,:,i), frames(:,:,i-1), 20);
            % if absolute difference greater than threshold set to 1     
            curr = (abs(curr - t) > t);
            diff(:,:,i) = curr;
        end
    end
    

    mei = zeros(h,w,m);
    for i = window:m
        curr = zeros(h,w);     
        for j = 0:window-1
            % union of boolean matrix from current to the current - window
            curr = curr | diff(:,:,i-j);
        end
        mei(:,:,i) = curr;
    end

    
end
