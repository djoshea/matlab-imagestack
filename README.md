# matlab-imagestack
Lightweight Matlas class which wraps multichannel ImageStacks

# Wrapper around a 4-D tensor of image data:

Image data tensor is X by Y by channels by frames.

## Properties

```matlab
  .data      % stores underlying data
  .name      % descriptive string
  .frameRate % used for tvec, play, and saveVideo
```

## Computed properties
```matlab
  .imageSize - [nX nY]
  .nChannels
  .nFrames
```

## Loading:
```matlab
  s = ImageStack.fromAllInDirectory(dir, ...)
  s = ImageStack.fromTif(file, ...)
```

## Indexing into data
```matlab
  s(maskX, maskY, channels, frames) % same as s.data(...)
  s{frame} % grab single frame as matrix
  s = s.getFrames(select) % same as s.data(:, :, :, select);
  [meanVsTime, tvec] = s.vsTime() % mean over time, precede with s.crop
```

## Return a new ImageStack:
```matlab
  s = s.frames(select) % get ImageStack with selected frames
  s = s.everyNthFrame(skip) % get 1:skip:end frames
  s = s.channels(select) % get ImageStack with selected frames
  s = s.crop(selectX, selectY) % get ImageStack with selected image region
```

## Stats:
```matlab
  s.minProj
  s.maxProj
  s.meanProj
  s.globalMin
  s.globalMax
  s.globalMinMax % [min max]
```

Many stats functions are supported directly and act on s.data
```matlab
  min(s, ...)
  max(s, ...)
  nanmin(s, ...)
  nanmax(s, ...)
  mean(s, ...)
  nanmean(s, ...)
  var(s, ...)
  nanvar(s, ...)
```

Standard arithmetic operators are overloaded to apply to s.data and are
automatically passed through bsxfun, e.g.
```matlab
  dfoverf = s ./ mean(s, 4);
  change = s - s{1};
```

## Transformations (return image stack)
```matlab
  sAlign = alignToReferenceTranslation(s, reference)
  sAlign = alignToMeanTranslation(s) % shifts each image to align with s.meanProj
  sNorm = normalize(s) % normalizes all values to [0 1] range
```

## Visualization - uses 
```matlab
  s.show % imagesc on first frame
  s.play % implay on s.data
  s.imcat % output to iTerm2 terminal as an image
  s.montage(...) % montage on s.data
```

## Chaining: any method that returns an image stack can be chained, e.g.
```matlab
  s.everyNthFrame(50).normalize.montage
```
