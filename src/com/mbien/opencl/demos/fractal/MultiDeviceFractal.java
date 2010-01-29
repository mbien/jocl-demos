package com.mbien.opencl.demos.fractal;

import com.mbien.opencl.CLBuffer;
import com.mbien.opencl.CLCommandQueue;
import com.mbien.opencl.CLDevice;
import com.mbien.opencl.CLEvent;
import com.mbien.opencl.CLEventList;
import com.mbien.opencl.CLException;
import com.mbien.opencl.CLGLBuffer;
import com.mbien.opencl.CLGLContext;
import com.mbien.opencl.CLKernel;
import com.mbien.opencl.CLProgram;
import com.mbien.opencl.CLProgram.CompilerOptions;
import com.sun.opengl.util.awt.TextRenderer;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Point;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.io.IOException;
import java.nio.IntBuffer;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.media.opengl.DebugGL2;
import javax.media.opengl.GL;
import javax.media.opengl.GL2;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLContext;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.GLProfile;
import javax.media.opengl.awt.GLCanvas;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import static com.sun.gluegen.runtime.BufferFactory.*;
import static javax.media.opengl.GL2.*;
import static com.mbien.opencl.CLMemory.Mem.*;
import static com.mbien.opencl.CLEvent.ProfilingCommand.*;
import static com.mbien.opencl.CLCommandQueue.Mode.*;
import static com.mbien.opencl.CLDevice.Type.*;
import static java.lang.Math.*;

/**
 * Computes the Mandelbrot set with OpenCL using multiple GPUs and renders the result with OpenGL.
 * A shared PBO is used as storage for the fractal image.
 * http://en.wikipedia.org/wiki/Mandelbrot_set
 * @author Michael Bien
 */
public class MultiDeviceFractal implements GLEventListener {

    // max number of used GPUs
    private static final int MAX_PARRALLELISM_LEVEL = 8;

    // max per pixel iterations to compute the fractal
    private static final int MAX_ITERATIONS         = 500;

    private GLCanvas canvas;

    private CLGLContext clContext;
    private CLCommandQueue[] queues;
    private CLKernel[] kernels;
    private CLEventList probes;
    private CLGLBuffer<IntBuffer>[] pboBuffers;

    private int width  = 0;
    private int height = 0;

    private float minX = -2f;
    private float minY = -1.2f;
    private float maxX  = 0.6f;
    private float maxY  = 1.3f;

    private int slices;

    private boolean drawSeperator = false;
    private boolean initialized = false;

    private final TextRenderer textRenderer;

    public MultiDeviceFractal(int width, int height) {

        this.width = width;
        this.height = height;

        canvas = new GLCanvas(new GLCapabilities(GLProfile.get(GLProfile.GL2)));
        canvas.addGLEventListener(this);
        initSceneInteraction();

        JFrame frame = new JFrame("JOCL Multi GPU Mandelbrot Set");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        canvas.setPreferredSize(new Dimension(width, height));
        frame.add(canvas);
        frame.pack();

        frame.setVisible(true);

        textRenderer = new TextRenderer(frame.getFont().deriveFont(Font.BOLD, 14), true, true, null, false);
    }

    public void init(GLAutoDrawable drawable) {

        // enable GL error checking using the composable pipeline
        drawable.setGL(new DebugGL2(drawable.getGL().getGL2()));

        initCL(drawable.getContext());

        GL2 gl = drawable.getGL().getGL2();

        gl.setSwapInterval(0);
        gl.glDisable(GL_DEPTH_TEST);
        gl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        initView(gl, drawable.getWidth(), drawable.getHeight());

        initPBO(gl);

    }

    private void initCL(GLContext glCtx){
        try {
            // create context managing all available GPUs
            clContext = CLGLContext.create(glCtx, GPU);

            // load and build program
            CLProgram program = clContext.createProgram(getClass().getResourceAsStream("Mandelbrot.cl"))
                                         .build(CompilerOptions.FAST_RELAXED_MATH);

            CLDevice[] devices = clContext.getCLDevices();

            slices = min(devices.length, MAX_PARRALLELISM_LEVEL);

            // create command queues for every GPU, setup colormap and init kernels
            queues = new CLCommandQueue[slices];
            kernels = new CLKernel[slices];
            probes = new CLEventList(slices);

            for (int i = 0; i < slices; i++) {

                CLBuffer<IntBuffer> colorMap = clContext.createIntBuffer(32*2, READ_ONLY);
                initColorMap(colorMap.getBuffer(), 32, Color.BLUE, Color.GREEN, Color.RED);

                // create command queue and upload color map buffer on each used device
                queues[i] = devices[i].createCommandQueue(PROFILING_MODE).putWriteBuffer(colorMap, true); // blocking upload

                // init kernel with constants
                kernels[i] = program.createCLKernel("mandelbrot")
                  .setArg(7, colorMap)
                  .setArg(8, colorMap.getBuffer().capacity())
                  .setArg(9, MAX_ITERATIONS);

            }

        } catch (IOException ex) {
            Logger.getLogger(getClass().getName()).log(Level.SEVERE, "can not find 'Mandelbrot.cl' in classpath.", ex);
        } catch (CLException ex) {
            Logger.getLogger(getClass().getName()).log(Level.SEVERE, "something went wrong, hopefully no one got hurt", ex);
        }

    }

    private void initColorMap(IntBuffer colorMap, int stepSize, Color... colors) {
        
        for (int n = 0; n < colors.length - 1; n++) {

            Color color = colors[n];
            int r0 = color.getRed();
            int g0 = color.getGreen();
            int b0 = color.getBlue();

            color = colors[n + 1];
            int r1 = color.getRed();
            int g1 = color.getGreen();
            int b1 = color.getBlue();

            int deltaR = r1 - r0;
            int deltaG = g1 - g0;
            int deltaB = b1 - b0;

            for (int step = 0; step < stepSize; step++) {
                float alpha = (float) step / (stepSize - 1);
                int r = (int) (r0 + alpha * deltaR);
                int g = (int) (g0 + alpha * deltaG);
                int b = (int) (b0 + alpha * deltaB);
                colorMap.put((r << 16) | (g << 8) | (b << 0));
            }
        }
        colorMap.rewind();

    }

    private void initView(GL2 gl, int width, int height) {

        gl.glViewport(0, 0, width, height);

        gl.glMatrixMode(GL_MODELVIEW);
        gl.glLoadIdentity();

        gl.glMatrixMode(GL_PROJECTION);
        gl.glLoadIdentity();
        gl.glOrtho(0.0, width, 0.0, height, 0.0, 1.0);
    }

    @SuppressWarnings("unchecked")
    private void initPBO(GL gl) {

        pboBuffers = new CLGLBuffer[kernels.length];

        int[] pbo = new int[pboBuffers.length];
        gl.glGenBuffers(pboBuffers.length, pbo, 0);

        // setup one empty PBO per slice
        for (int i = 0; i < slices; i++) {

            gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[i]);
            gl.glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height * SIZEOF_INT / slices, null, GL_STREAM_DRAW);
            gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            pboBuffers[i] = clContext.createFromGLBuffer(null, pbo[i], WRITE_ONLY);
        }
        
        initialized = true;
    }

    public void display(GLAutoDrawable drawable) {
        if(!initialized) {
            initPBO(drawable.getGL());
        }
        compute();
        render(drawable.getGL().getGL2());
    }

    // OpenCL
    private void compute() {

        int sliceWidth = width / slices;
        float rangeX   = (maxX - minX) / slices;
        float rangeY   = (maxY - minY);

        // release all old events, you can't reuse events in OpenCL
        probes.release();

        for (int i = 0; i < slices; i++) {

            kernels[i].putArg(pboBuffers[i])
                      .putArg(sliceWidth).putArg(height)
                      .putArg(minX + rangeX*i).putArg(minY)
                      .putArg(       rangeX  ).putArg(rangeY)
                      .rewind();

            // aquire GL objects, and enqueue a kernel with a probe from the list
            queues[i].putAcquireGLObject(pboBuffers[i].ID)
                     .put2DRangeKernel(kernels[i], 0, 0, sliceWidth, height, 0, 0, probes)
                     .putReleaseGLObject(pboBuffers[i].ID);

        }

    }

    // OpenGL
    private void render(GL2 gl) {

        gl.glClear(GL_COLOR_BUFFER_BIT);

        //draw slices
        int sliceWidth = width / slices;

        for (int i = 0; i < slices; i++) {

            int seperatorOffset = drawSeperator?i:0;

            gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboBuffers[i].GLID);
            gl.glRasterPos2i(sliceWidth*i + seperatorOffset, 0);

            gl.glDrawPixels(sliceWidth, height, GL_BGRA, GL_UNSIGNED_BYTE, 0);

        }
        gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        //draw info text
        textRenderer.beginRendering(width, height, false);

            for (int i = 0; i < slices; i++) {
                CLEvent event = probes.getEvent(i);
                long start = event.getProfilingInfo(START);
                long end = event.getProfilingInfo(END);
                textRenderer.draw("GPU"+i +" "+(int)((end-start)/1000000.0f)+"ms", 10, 10+16*(slices-i));
            }

        textRenderer.endRendering();
    }

    public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {

        if(this.width == width && this.height == height)
            return;

        this.width = width;
        this.height = height;

        for (CLGLBuffer<IntBuffer> buffer : pboBuffers) {
            buffer.release();
        }

        initPBO(drawable.getGL());
        initView(drawable.getGL().getGL2(), drawable.getWidth(), drawable.getHeight());
    }

    private void initSceneInteraction() {

        MouseAdapter mouseAdapter = new MouseAdapter() {

            Point lastpos = new Point();

            @Override
            public void mouseDragged(MouseEvent e) {
                
                float offsetX = (lastpos.x - e.getX()) * (maxX - minX) / width;
                float offsetY = (lastpos.y - e.getY()) * (maxY - minY) / height;

                minX += offsetX;
                minY -= offsetY;

                maxX += offsetX;
                maxY -= offsetY;

                lastpos = e.getPoint();

                canvas.display();

            }

            @Override
            public void mouseMoved(MouseEvent e) {
                lastpos = e.getPoint();
            }
            
            @Override
            public void mouseWheelMoved(MouseWheelEvent e) {
                float rotation = e.getWheelRotation() / 25.0f;

                float deltaX = rotation * (maxX - minX);
                float deltaY = rotation * (maxY - minY);

                // offset for "zoom to cursor"
                float offsetX = (e.getX() / (float)width - 0.5f) * deltaX * 2;
                float offsetY = (e.getY() / (float)height- 0.5f) * deltaY * 2;

                minX += deltaX+offsetX;
                minY += deltaY-offsetY;

                maxX +=-deltaX+offsetX;
                maxY +=-deltaY-offsetY;

                canvas.display();
            }
        };

        KeyAdapter keyAdapter = new KeyAdapter() {

            @Override
            public void keyPressed(KeyEvent e) {
                if(e.getKeyCode() == KeyEvent.VK_SPACE) {
                    drawSeperator = !drawSeperator;
                }else if(e.getKeyChar() > '0' && e.getKeyChar() < '9') {
                    int number = e.getKeyChar()-'0';
                    slices = min(number, min(queues.length, MAX_PARRALLELISM_LEVEL));
                    initialized = false;
                }
                canvas.display();
            }

        };

        canvas.addMouseMotionListener(mouseAdapter);
        canvas.addMouseWheelListener(mouseAdapter);
        canvas.addKeyListener(keyAdapter);
    }

    public void dispose(GLAutoDrawable drawable) {
    }

    public static void main(String args[]) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                new MultiDeviceFractal(512, 512);
            }
        });
    }

}
