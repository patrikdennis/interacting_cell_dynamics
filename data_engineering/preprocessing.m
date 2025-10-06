% --- Image Processing Script ---
% This script processes all images in a specified input folder by adjusting
% their contrast and applying a top-hat filter, then saves the results
% to a new output folder.

clear; clc; close all;

%% 1. Define File Paths
% Set the path to your folder of original images
inputFolder = '/Users/patrik/cell_diffusion_modelling/558/experiment_Stabilized/A1';

% Set the path for the new folder where processed images will be saved
outputFolder = '/Users/patrik/cell_diffusion_modelling/558/A1_transformed';

% Check if the output folder exists. If not, create it
if ~exist(outputFolder, 'dir')
   mkdir(outputFolder);
   disp(['Created output folder: ' outputFolder]);
end

%% 2. Get Image Files
% Get a list of all image files in the input folder.
imageFiles = dir(fullfile(inputFolder, '*.tif'));

if isempty(imageFiles)
    error('No image files found in the specified input folder. Check the path and file extension.');
end

%% 3. Define Processing Parameters
% Create the disk-shaped structuring element for the top-hat filter
% This is done once before the loop for better performance
se = strel('disk', 4);

%% 4. Process Each Image in a Loop
fprintf('Starting image processing...\n');

for i = 1:length(imageFiles)
    % Prepare file paths 
    baseFileName = imageFiles(i).name;
    fullInputFileName = fullfile(inputFolder, baseFileName);
    fullOutputFileName = fullfile(outputFolder, baseFileName);
    
    % Display progress to the command window.
    fprintf('Processing image %d of %d: %s\n', i, length(imageFiles), baseFileName);
    
    % Read the original image 
    originalImage = imread(fullInputFileName);
    
    % Adjust contrast 
    % Using imadjust with settings: low_in=0.3, high_in=0.7, low_out=0, high_out=1
    contrastAdjustedImage = imadjust(originalImage, [0.3 0.7], [0 1]);
    
    %  Apply top-hat filter 
    % Using imtophat on the contrast-adjusted image with the predefined structuring element.
    finalImage = imtophat(contrastAdjustedImage, se);
    
    % Save the final processed image 
    imwrite(finalImage, fullOutputFileName);
end

fprintf('Processing complete! All %d images have been saved to:\n%s\n', length(imageFiles), outputFolder);