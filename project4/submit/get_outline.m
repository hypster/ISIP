function [outline] = get_outline(mei)
    outline = zeros(size(mei));
    for i = 1:size(mei,3)
        f = mei(:,:,i);
        f2 = imerode(f,strel('disk',3));
        outline(:,:,i) = f - f2;
    end
% implay(outline);
end