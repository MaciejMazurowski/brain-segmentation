function [ slices, mask] = preprocessing3D( slices, mask, destination_path, prefix )
%PREPROCESSING3D Implements preprocessing of a 3D volume containing slices 
%of a FLAIR modality together with its segmentation mask. Needs a path to 
%the folder (string) where you want to save the result and a filename 
%prefix (string) to which a slice number is appended. The images will be 
%saved in tiff format.
%
%Examples:
%
%   Basic usecase:
%
%       [slices, mask] = preprocessing3D(slices, mask, '/media/username/data/train/', 'patient_001');
%   
%   If you don't have a segmentation mask and want to preprocess test 
%   images for inference, pass a zeros matrix instead:
%
%       [slices, mask] = preprocessing3D(slices, zeros(size(slices)), '/media/username/data/train/', 'patient_001');


    mask(mask ~= 0) = 1;

    % resize to have smaller dimension equal 256 pixels
    if min(size(slices(:, :, 1))) ~= 256

        scale = 256 / min(size(slice));
        % resize images to 256 with bicubic interpolation
        slices = imresize(slices, scale);
        % and mask with NN interpolation
        mask = imresize(mask, scale, 'method', 'nearest');

    end

    % center crop to 256x256 square
    slices = center_crop(slices, [256 256]);
    mask = center_crop(mask, [256 256]);

    % fill holes in segmentation mask
    for s = 1:size(mask, 3)
        mask(:, :, s) = imfill(mask(:, :, s), 'holes');
    end

    % fix the rage of pixel values after bicubic interpolation
    slices(slices < 0) = 0;

    % get histogram of an image volume
    [N, edges] = histcounts(slices(:), 'BinWidth', 2);

    % rescale the intensity peak to be at value 100
    minimum = edges(find(edges > prctile(slices(:), 2), 1));

    diffN = zeros(size(N));
    for nn = 2:numel(N)
        diffN(nn) = N(nn) / N(nn - 1);
    end
    s = find(edges >= prctile(slices(:), 50), 1);
    f = find(diffN(s:end) > 1.0, 5);
    start = s + f(5);

    [~, ind] = max(N(start:end));
    peak_val = edges(ind + start - 1);
    maximum = minimum + ((peak_val - minimum) * 2.55);

    slices(slices < minimum) = minimum;
    slices(slices > maximum) = maximum;
    slices = (slices - minimum) ./ (maximum - minimum);

    % save preprocessed images
    slices = im2uint8(slices);
    mask = im2uint8(mask);

    slicesPerImage = 1;

    for s = size(slices, 3):-slicesPerImage:1
        startSlice = max([1, (s - slicesPerImage + 1)]);
        imageSlices = slices(:, :, startSlice:(startSlice + slicesPerImage - 1));
        maskSlices = mask(:, :, startSlice:(startSlice + slicesPerImage - 1));

        saveastiff(imageSlices, [destination_path prefix '_' num2str(startSlice) '.tif']);
        saveastiff(maskSlices, [destination_path prefix '_' num2str(startSlice) '_mask.tif']);
    end

end


function [ image ] = center_crop( image, cropSize )
%CENTER_CROP Center crop of given size

    [p3, p4, ~] = size(image);

    i3_start = max(1, floor((p3 - cropSize(1)) / 2));
    i3_stop = i3_start + cropSize(1) - 1;

    i4_start = max(1, floor((p4 - cropSize(2)) / 2));
    i4_stop = i4_start + cropSize(2) - 1;

    image = image(i3_start:i3_stop, i4_start:i4_stop, :);

end
