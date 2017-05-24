classdef ParImageStack
% Wrapper around a 4-D tensor of image data:
% img1 x img 2 x channels x frames
% 2015 Dan O'Shea
%
% Properties
%   .data - underlying data [Y by X by channels by frames]
%   .name - string
%   .frameRate - used for tvec, play, and saveVideo
% 
% Computed properties
%   .imageSize - [nX nY]
%   .nChannels
%   .nFrames
%
% Loading:
%   s = ParImageStack.fromAllInDirectory(dir, ...)
%   s = ParImageStack.fromTif(file, ...)
% 
% Indexing into data
%   s(maskX, maskY, channels, frames): same as s.data(...)
%   s{frame} : grab single frame as matrix
%   s = s.getFrames(select) - same as s.data(:, :, :, select);
%   [meanVsTime, tvec] = s.vsTime - mean over time, precede with s.crop
%
% Return a new ParImageStack:
%   s = s.frames(select) - get ParImageStack with selected frames
%   s = s.withEveryNthFrame(skip) - get 1:skip:end frames
%   s = s.channels(select) - get ParImageStack with selected frames
%   s = s.crop(selectX, selectY)
%   s = s.minProject, meanProject, maxProject, stdProject
% Stats:
%   s.minOverTime, s.maxOverTime, s.meanOverTime
%   s.globalMin, s.globalMax, s.globalMinMax
%
% Many stats functions are supported directly and act on s.data
%   min, max, nanmin, nanmax
%   mean, nanmean, var, nanvar
%
% Standard arithmetic operators are overloaded to apply to s.data and are
% automatically passed through bsxfun, e.g.
%   dfoverf = s ./ mean(s, 4);
%   change = s - s{1};
%
% Transformations (return image stack)
%   sAlign = alignToReferenceTranslation(s, reference)
%   sAlign = alignToMeanTranslation(s)
%   sNorm = normalize(s)
%
% Visualization - uses 
%   s.show - imagesc on first frame
%   s.play - implay on s.data
%   s.imcat - output to iTerm2 terminal as an image
%   s.montage
%
% Chaining: any method that returns an image stack can be chained, e.g.
%   s.withEveryNthFrame(50).normalize.montage

    properties
        data
        
        frameRate = 10;
        
        name = ''
        
        metaByFrame % nFrames x 1 struct which will automatically be sliced with the data
    end
    
    properties(Dependent)
        imageSize
        
        nChannels
        
        nFrames
    end
    
    methods
        function s = ParImageStack(data, varargin)
            if ~isfloat(data)
                s.data = single(data);
            else
                s.data = data;
            end
            
            p = inputParser();
            p.addParameter('name', '', @ischar);
            p.addParameter('frameRate', 10, @isscalar);
            p.addParameter('metaByFrame', [], @(x) isempty(x) || isstruct(x));
            p.parse(varargin{:});
            
            s.frameRate = s.frameRate;
            s.name = p.Results.name;
            if ~isempty(p.Results.metaByFrame)
                assert(numel(p.Results.metaByFrame) == s.nFrames, 'Meta by frame length must match nFrames');
                s.metaByFrame = makecol(p.Results.metaByFrame);
            else
                s.metaByFrame = emptyStructArray(s.nFrames);
            end
        end
    end
    
    methods % Statistics
        function tvec = tvec(s)
            tvec = (0 : (1/s.frameRate) : ((s.nFrames-1)/s.frameRate))';
        end
            
        function img = meanOverTime(s)
            img = mean(s.data, 4);
        end
        
        function r = globalMinMax(s)
            r = [min(s.data(:)), max(s.data(:))];
        end
        
        function m = globalMin(s)
            m = nanmin(s.data(:));
        end
        
        function m = globalMax(s)
            m = nanmax(s.data(:));
        end
        
        function m = min(s, varargin)
            m = min(s.data, varargin{:});
        end
        
        function m = max(s, varargin)
            m = max(s.data, varargin{:});
        end
        
        function m = mean(s, varargin)
            m = mean(s.data, varargin{:});
        end

        function m = nanmean(s, varargin)
            m = nanmean(s.data, varargin{:});
        end
        
        function m = var(s, varargin)
            m = var(s.data, varargin{:});
        end
        
        function m = nanvar(s, varargin)
            m = nanvar(s.data, varargin{:});
        end
        
        function m = nanmin(s, varargin)
            m = nanmin(s.data, varargin{:});
        end
        
        function m = nanmax(s, varargin)
            m = nanmax(s.data, varargin{:});
        end
        
        function img = minOverTime(s)
            img = min(s.data, [], 4);
        end
        
        function img = maxOverTime(s) 
            img = max(s.data, [], 4);
        end
        
        function [v, tvec] = vsTime(s)
            % v = roiVsTime(s)
            % v is nFrames x nChannels
            data = reshape(s.data, [prod(s.imageSize) s.nChannels s.nFrames]); %#ok<*PROPLC>
            v = TensorUtils.squeezeDims(mean(data, 1), 1)';
            tvec = s.tvec;
        end
    end
    
    methods % Access to data
        function d = getFrames(s, idx)
            d = s.data(:, :, :, idx);
        end
        
        function d = getChannels(s, idx)
            d = s.data(:, :, idx, :);
        end
        
        % as new ParImageStack
        function s = frames(s, idx)
            s.data = s.data(:, :, :, idx);
            if ~isempty(s.metaByFrame)
                s.metaByFrame = s.metaByFrame(idx);
            end
        end
        
        function s = withEveryNthFrame(s, skip)
            s = s.frames(1:skip:s.nFrames);
        end
        
        function s = channels(s, idx)
            s.data = s.data(:, :, idx, :);
        end
            
        function s = crop(s, cropX, cropY)
            s.data = s.data(cropX, cropY, :, :);
        end
    end
    
    methods % Transformations
        function [s, shifts] = alignToReferenceTranslation(s, reference)
            [optimizer, metric] = imregconfig('monomodal');
            prog = ProgressBar(s.nFrames, 'Aligning images...');
            prog.enableParallel();
            data = s.data;
            nFrames = size(data, 4);
           % ffRef = fft2(reference);
            shifts = nan(nFrames, 2);
            parfor i = 1:nFrames
                %[out, freg] = ParImageStackTools.dftregistration(fft2(data(:, :, :, i)), ffRef, 4);
                %shifts(i, :) = out(3:4);
                %data(:, :, :, i) = abs(ifft2(freg));
                data(:, :, :, i) = imregister(data(:, :, :, i), reference, 'translation', optimizer, metric);
                prog.update(i); %#ok<PFBNS>
            end
            s.data = data;
            prog.finish();
        end
        
        function s = alignToMeanTranslation(s)
            s = s.alignToReferenceTranslation(s.meanOverTime);
        end
        
        function [s, minmax] = normalize(s)
            minmax = s.globalMinMax;
            s.data = (s.data - minmax(1)) / (minmax(2) - minmax(1)); %#ok<*PROP>
        end
        
        function s = dfof(s)
            s.data = s.data ./ nanmean(s.data, 4);
        end
        
        function s = meanProject(s)
            s.data = nanmean(s.data, 4);
        end
        
        function s = maxProject(s)
            s.data = nanmax(s.data, 4);
        end
        
        function s = minProject(s)
            s.data = nanmin(s.data, 4);
        end
        
        function s = stdProject(s)
            s.data = nanstd(s.data, [], 4);
        end
        
        function s = applyBinaryFn(s, fn, o)
            if isnumeric(o)
                s.data = bsxfun(fn, s.data, o);
            elseif isa(o, 'ParImageStack')
                s.data = bsxfun(fn, s.data, o.data);
            else
                error('Unknown argument type %s', class(o));
            end
        end
        
        function s = globalLinearDetrend(s)
%             nX = size(s.data, 1);
%             nY = size(s.data, 2);
%             nC = size(s.data, 3);

            gm = s.vsTime();
            gmDetrend = detrend(squeeze(gm));
            residual = gm - gmDetrend;
            s = s - shiftdim(residual, -3);
        end
    end
    
    
    methods % Filtering
        
        function s = smoothGaussian2(s, sigmaXY)
            for c = 1:s.nChannels
                for f = 1:s.nFrames
                    s.data(:, :, c, f) = imgaussfilt(s.data(:, :, c, f), sigmaXY);
                end
            end
        end
        
        function s = smoothGaussian3(s, sigmaXYZ)
            for c = 1:s.nChannels
                cdata = squeeze(s.data(:, :, c, :));
            	cdata = imgaussfilt3(cdata, sigmaXYZ);
                s.data(:, :, c, :) = reshape(cdata, size(s.data(:, :, c, :)));
            end
        end
        
    end
    
    methods(Static)
%         function varargout = alignWholeStackUsingMeansTranslation(varargin)
%             meanCell = cellfun(@(x) mean(x, 4), varargin{:}, 'UniformOutput', false);
%             meanStack = cat(4, meanCell{:});
%             
%             sMean = ParImageStack(meanStack);
%             sMean.alignToReference
%         end
    end
    
    methods(Hidden) % Internal Utilities
        function assertMonoOrRGB(s)
            assert(s.nChannels == 1 || s.nChannels == 3, 'Operation supported only for 1 or 3 channel image stacks');
        end
    end
    
    methods % operator and builtin function overloading 
        function sz = size(s, varargin)
            sz = size(s.data, varargin{:});
        end
        
        function ind = end(s,k,n)
           szd = size(s.data);
           if k < n
              ind = szd(k);
           else
              ind = prod(szd(k:end));
           end
        end

        function s = plus(s, o)
            s = s.applyBinaryFn(@plus, o);
        end
        
        function s = minus(s, o)
            s = s.applyBinaryFn(@minus, o);
        end
        
        function s = rdivide(s, o)
            s = s.applyBinaryFn(@rdivide, o);
        end
        
        function s = times(s, o)
            s = s.applyBinaryFn(@times, o);
        end
        
        function s = logStack(s)
            s.data = log(s.data);
        end
        
        function s = cat(dim, varargin)
            dataCell = cellfun(@(x) x.data, varargin, 'UniformOutput', false);
            s = varargin{1};
            s.data = cat(dim, dataCell{:});
        end
        
        function s = horzcat(varargin)
            s = cat(2, varargin{:});
        end
        
        function s = vertcat(varargin)
            s = cat(1, varargin{:});
        end
        
        function varargout = subsref(s, subs)
            % direct pass thru to .data
            if strcmp(subs(1).type, '()')
                if length(subs) > 1
                    [varargout{1:nargout}] = builtin('subsref', s.data, subs(2:end));
                else
                    [varargout{1:nargout}] = s.data(subs(1).subs{:});
                end
            elseif strcmp(subs(1).type, '{}')
                % index into frames
                if numel(subs(1).subs) == 1 && numel(numel(subs(1).subs{1})) == 1
                    % index into channels, then frames
                    data = s.data(:, :, :, subs(1).subs{1});
                else
                    error('Only one frame index accepted with {} indexing');
                end
                
                if length(subs) > 1
                    [varargout{1:nargout}] = builtin('subsref', data);
                else
                    [varargout{1:nargout}] = data;
                end
            else
                [varargout{1:nargout}] = builtin('subsref', s, subs);
            end
        end
    end
    
    methods % Visualization methods
        function setColormap(s) %#ok<MANU>
            colormap(ParImageStack.getDefaultColormap());
        end
        
        function imcat(s)
           m = s.getFrames(1);
           C = ParImageStack.getDefaultColormap;
            
            m = squeeze(double(m));

            % make large enough to see (min dimension should be 500 px)
            maxPixelSize = 10;
            [r, c] = size(m);

            if ~ismatrix(m)
                warning('Showing slice (:, :, 1) of multidimensional matrix');
                m = m(:, :, 1);
            end

            resizeBy = min(maxPixelSize, round(min(800 / r, 800 / c)));
            if resizeBy > 1
                m = kron(m, ones(resizeBy));
            end

            if nargin < 2
                % Now make an RGB image that matches display from IMAGESC:
                C = get(0, 'DefaultFigureColormap');  % Get the figure's colormap.
            end
            L = size(C,1);

            % Scale the matrix to the range of the map.
            maxM = nanmax(m(:));
            minM = nanmin(m(:));
            if ~isnan(maxM) && ~isnan(minM)
                if maxM - minM < eps
                    mc = ones(size(m));
                else
                    mc = round(interp1(linspace(minM,maxM,L),1:L,m));
                end
            else
                mc = m;
            end
            mc(isnan(mc)) = L+1; % specify nan's index into colormap
            C = cat(1, C, [0 0 0]); % make white the nan color
            mc = reshape(C(mc,:),[size(mc) 3]); % Make RGB image from scaled.

            f = tempname;
            imwrite(mc, f, 'png');
            imgcat(f);
            
            function imgcat(fname)
                % execute imgcat
                stack = dbstack('-completenames');
                d = fileparts(stack(1).file);
                d = strrep(d, ' ', '\ ');
                imgcatPath = fullfile(d, 'imgcat');
                system(['chmod u+x ' imgcatPath]);
                system([imgcatPath ' ' fname]);
            end
        end
        
        function show(s, idx)
            if nargin < 2
                idx = 1;
            end
            imagesc(s.getFrames(idx));
            axis equal; axis tight;
            h = title(s.name);
            h.Interpreter = 'none';
        end
        
        function h = roiBox(s, rlims, clims, varargin)
            rlims = [min(rlims(:)), max(rlims(:))];
            clims = [min(clims(:)), max(clims(:))];
            
            w = diff(clims);
            h = diff(rlims);
           
            h = rectangle('Position', [clims(1) rlims(1) w h], 'EdgeColor', 'r', 'LineWidth', 4, varargin{:});
        end
        
        function play(s)
            implay(s.normalize.data, s.frameRate);
            s.setColormap();
        end
        
        function montage(s, varargin)
            montage(s.normalize.data);
            s.setColormap();
        end
    end
    
    methods % Dependent properties
        function sz =  get.imageSize(s)
            sz = [size(s.data, 1), size(s.data, 2)];
        end
        
        function n = get.nChannels(s)
            n = size(s.data, 3);
        end
        
        function n = get.nFrames(s)
            n = size(s.data, 4);
        end
    end
    
    methods % Persistence to disk
        function saveVideo(s, file, varargin)
            s.assertMonoOrRGB();
            
            [path, leaf, ext] = fileparts(file);
            if isempty(ext)
                folder = file;
                file = fullfile(folder, 'video.avi');
            else
                folder = path;
            end
            
            mkdirRecursive(folder);
            
            vObj = VideoWriter(GetFullPath(file));
            vObj.FrameRate = s.frameRate;
            
            prog = ProgressBar(s.nFrames, 'Writing video to %s', vObj.Filename);
            open(vObj);
            for iT = 1:s.nFrames
                prog.update(iT);
                vObj.writeVideo(s.data(:, :, :, iT));
            end
            prog.finish();
            close(vObj);
        end

        function saveToDirectory(s, imgDir, varargin)
            p = inputParser();
            p.addParameter('prefix', 'image_', @ischar);
            p.addParameter('compression', 'lzw', @ischar);
            p.parse(varargin{:});

            mkdirRecursive(imgDir);
            nFrames = s.nFrames;
            [s, minmax] = s.normalize();
            debug('Saving info in info.mat\n');
            info.minmax = minmax; %#ok<STRNU>
            save('info.mat', 'info');

            data = s.data;
            prog = ProgressBar(s.nFrames, 'Writing tif images to %s', imgDir);
            compression = p.Results.compression;
            prefix = p.Results.prefix;
            prog.enableParallel();
            parfor i = 1:nFrames
                fname = fullfile(imgDir, sprintf('%s%05d.tif', prefix, i));
                scaled = uint16(data(:, :, :, i) * (2^16-1));
                imwrite(scaled, fname, 'Compression', compression);
                prog.update(i); %#ok<PFBNS>
            end

        end
    end
        
    methods(Static)
        function s = fromFileFullPathList(fileList, varargin)
            % load images with fully specified paths
            p = inputParser();
            p.addParameter('name', '', @ischar);
            p.addParameter('cropX', [], @isvector);
            p.addParameter('cropY', [], @isvector);
            p.KeepUnmatched = true;
            p.parse(varargin{:});
            
            nFiles = numel(fileList);
            
            % figure out cropping masks
            img = imread(fileList{1});
            [useRegion, cropX, cropY, szImg] = ParImageStack.determinePixelRegion(p.Results.cropX, p.Results.cropY, size(img));
            
            debug('Allocating memory for images...this may take some time\n');
            stack = zeros([szImg, 1, nFiles], 'like', img);
                
            prog = ProgressBar(nFiles, 'Loading %d images', nFiles);
            %prog.enableParallel();
            for i = 1:nFiles
                file = fileList{i};
                if useRegion
                    img = imread(file, 'PixelRegion', {cropX, cropY});
                else
                    img = imread(file);
                end
%                 if ~isfloat(img)
%                 	img = single(img);
%                 end
                stack(:, :, :, i) = img;
                prog.update(i);
            end
            prog.finish();
            
            if isempty(p.Results.name)
                % use leaf of directory name
                [~, name] = fileparts(fileList{1});
            else
                name = p.Results.name;
            end
            
            s = ParImageStack(stack, 'name', name, p.Unmatched);
        end
        
        function s = fromFileListInDirectory(fileList, directory, varargin)
            directory = GetFullPath(directory);
            for iF = 1:numel(fileList)
                fileList{iF} = fullfile(directory, fileList{iF});
            end
            
            [~, name] = fileparts(directory);
            s = ParImageStack.fromFileFullPathList(fileList, 'name', name, varargin{:});
        end
        
        function s = fromAllInDirectoryMatching(imgDir, match, varargin)
            % imgDir is rootpath, match is like '*.tif';
            match = fullfile(imgDir, match);
            matches = dir(match);
            files = c
            
            s = ParImageStack.fromFileListInDirectory(files, imgDir);
        end
        
        function s = fromAllInDirectoryRecursive(imgDir, varargin)
            % loads all images within directory and returns a nX x nY x nImg single
            % tensor of image data

            p.parse(varargin{:});

            imgDir = GetFullPath(imgDir);
            if ~exist(imgDir, 'dir')
                error('Directory %s not found', imgDir);
            end
            
            % load the images using data store
            debug('Searching for file names\n')
            ds = datastore(imgDir, 'IncludeSubfolders', true, 'FileExtensions', '.tif','Type', 'image');
            
            [~, name] = fileparts(imgDir);
            
            s = ParImageStack.fromFileFullPathList(ds.Files, 'name', name, p.Unmatched);
        end
        
        function s = fromTif(imgFile, varargin)
            p = inputParser();
            p.addParameter('name', '', @ischar);
            p.addParameter('cropX', [], @isvector);
            p.addParameter('cropY', [], @isvector);
            p.KeepUnmatched = true;
            p.parse(varargin{:});

            imgFile = GetFullPath(imgFile);
            if ~exist(imgFile, 'file')
                error('Image file %s not found', imgFile);
            end

            info = imfinfo(imgFile);
            nFiles = length(info);

            % figure out cropping masks
            img = imread(imgFile, 1, 'Info', info);
            [useRegion, cropX, cropY, szImg] = ParImageStack.determinePixelRegion(p.Results.cropX, p.Results.cropY, size(img));
            stack = nan([szImg, 1, nFiles]);

            prog =  ProgressBar(nFiles, 'Loading %d images in %s...', nFiles, imgFile);
            for i = 1:nFiles
                prog.update(i);
                if useRegion
                    img = imread(imgFile, i, 'Info', info, 'PixelRegion', {cropX, cropY});
                else
                    img = imread(imgFile, i, 'Info', info);
                end
                if ~isfloat(img)
                    img = single(img);
                end
                stack(:, :, :, i) = img;
            end
            prog.finish();
            
            if isempty(p.Results.name)
                % use leaf of directory name
                [~, name] = fileparts(imgFile);
            else
                name = p.Results.name;
            end
            
            s = ParImageStack(stack, 'name', name, p.Unmatched);
        end
    end

    methods(Static, Hidden)
        function c = getDefaultColormap()
            c = pmkmp(255, 'CubicL');
        end
        function [useRegion, cropX, cropY, szImg] = determinePixelRegion(cropX, cropY, szImg)
            % determine the argument 'PixelRegion' sent to imread
            useRegion = false;
            if ~isempty(cropX)
                useRegion = true;
                if numel(cropX) == 3
                    szImg(1) = numel(cropX(1):cropX(2):cropX(3));
                elseif numel(cropX) == 2
                    szImg(1) = numel(cropX(1):cropX(2));
                else
                    error('cropX must be 2 or 3 element vector as used with : notation');
                end
            else
                cropX = [1 szImg(1)];
            end

            if ~isempty(cropY)
                useRegion = true;
                if numel(cropY) == 3
                    szImg(2) = numel(cropY(1):cropY(2):cropY(3));
                elseif numel(cropY) == 2
                    szImg(2) = numel(cropY(1):cropY(2));
                else
                    error('cropY must be 2 or 3 element vector as used with : notation');
                end
            else
                cropY = [1 szImg(2)];
            end
        end
    end

end
