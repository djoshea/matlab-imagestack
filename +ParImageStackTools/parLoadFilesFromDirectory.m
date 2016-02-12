function stack = parLoadFilesFromDirectory(files, useRegion, cropX, cropY)

    img = imread(files{1});
    nFiles = length(files);

    debug('Allocating memory for images...this may take some time\n');
    stack = zeros([size(img), 1, nFiles], 'like', img);

    prog = ProgressBar(nFiles, 'Loading %d images in directory...', nFiles);
    prog.enableParallel();
    parfor i = 1:nFiles
        prog.update(i);
        file = files{i};
        if useRegion
            img = imread(file, 'PixelRegion', {cropX, cropY});
        else
            img = imread(file);
        end
        if ~isfloat(data)
    %                 	img = single(img);
        end
        stack(:, :, :, i) = img;
    end
    prog.finish();

end