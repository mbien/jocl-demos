/**
 * For a description of this algorithm please refer to
 * http://en.wikipedia.org/wiki/Mandelbrot_set
 * @author Michael Bien
 */
kernel void mandelbrot(
        global uint *output,
        const int width, const int height,
        const float x0, const float y0,
        const float rangeX, const float rangeY,
        global uint *colorMap, const int colorMapSize, const int maxIterations) {

    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);

    float r = x0 + ix * rangeX / width;
    float i = y0 + iy * rangeY / height;

    float x = 0;
    float y = 0;

    float magnitudeSquared = 0;
    int iteration = 0;

    while (magnitudeSquared < 4 && iteration < maxIterations) {
        float x2 = x*x;
        float y2 = y*y;
        y = 2 * x * y + i;
        x = x2 - y2 + r;
        magnitudeSquared = x2+y2;
        iteration++;
    }

    if (iteration == maxIterations)  {
        output[iy * width + ix] = 0;
    }else {
        float alpha = (float)iteration / maxIterations;
        int colorIndex = (int)(alpha * colorMapSize);
        output[iy * width + ix] = colorMap[colorIndex];
      // monochrom
      //  output[iy * width + ix] = 255*iteration/maxIterations;
    }

}