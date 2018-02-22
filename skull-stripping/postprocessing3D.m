function [ postprocessed ] = postprocessing3D( predictions )
%POSTPROCESSING3D Implements postprocessing of a 3D volume containing 
%predicted segmentation masks. First, it extracts the largest 3D conneted
%component and then adds a convex hull for each slice.

predictions = round(predictions);

% Extract the largest connected 3D component

CC = bwconncomp(predictions);

PixelIdxList = CC.PixelIdxList;

if numel(PixelIdxList) < 1
    
    postprocessed = pred;
    return
    
end

maxval = numel(PixelIdxList{1});
index = 1;

for i = 1:length(PixelIdxList)
    if numel(PixelIdxList{i}) > maxval
        maxval = numel(PixelIdxList{i});
        index = i;
    end
end

[y, x, z] = ind2sub(size(pred), PixelIdxList{index});
postprocessed = zeros(size(pred));
for i = 1:numel(PixelIdxList{index})
    postprocessed(y(i), x(i), z(i)) = 1;
end

% 2D (per slice) union convex hull

for k = 1:size(postprocessed, 3)
    slice = postprocessed(:, :, k);
    slice = bwconvhull(slice);
    postprocessed(:, :, k) = slice;
end
