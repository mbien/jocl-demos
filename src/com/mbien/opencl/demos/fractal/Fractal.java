package com.mbien.opencl.demos.fractal;

import com.mbien.opencl.CLBuffer;
import com.mbien.opencl.CLCommandQueue;
import com.mbien.opencl.CLException;
import com.mbien.opencl.CLGLBuffer;
import com.mbien.opencl.CLGLContext;
import com.mbien.opencl.CLKernel;
import com.mbien.opencl.CLProgram;
import com.mbien.opencl.CLProgram.CompilerOptions;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Point;
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
import static com.mbien.opencl.CLDevice.Type.*;

/**
 * Computes the Mandelbrot set with OpenCL and renders the result with OpenGL.
 * A shared PBO is used as storage for the fractal image.
 * http://en.wikipedia.org/wiki/Mandelbrot_set
 * @author Michael Bien
 */
public class Fractal implements GLEventListener {

    private GLCanvas canvas;

    private CLGLContext clContext;
    private CLCommandQueue commandQueue;
    private CLKernel kernel;

    private CLGLBuffer<IntBuffer> pboBuffer;

    private int width  = 0;
    private int height = 0;

    private float minX = -2f;
    private float minY = -1.2f;
    private float maxX  = 0.6f;
    private float maxY  = 1.3f;

    public Fractal(int width, int height) {

        this.width = width;
        this.height = height;

        canvas = new GLCanvas(new GLCapabilities(GLProfile.get(GLProfile.GL2)));
        canvas.addGLEventListener(this);
        initSceneInteraction();

        JFrame frame = new JFrame("JOCL Mandelbrot Set");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        canvas.setPreferredSize(new Dimension(width, height));
        frame.add(canvas);
        frame.pack();

        frame.setVisible(true);
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
            clContext = CLGLContext.create(glCtx);

            // load and build program
            CLProgram program = clContext.createProgram(getClass().getResourceAsStream("Mandelbrot.cl"))
                                         .build(CompilerOptions.FAST_RELAXED_MATH);

            // setup colormap and create command queue on fastest GPU
            CLBuffer<IntBuffer> colorMap = clContext.createIntBuffer(32*2, READ_ONLY);
            initColorMap(colorMap.getBuffer(), 32, Color.BLUE, Color.GREEN, Color.RED);

            commandQueue = clContext.getMaxFlopsDevice(GPU).createCommandQueue()
              .putWriteBuffer(colorMap, true); // blocking upload

            // init kernel with constants
            kernel = program.getCLKernel("mandelbrot")
              .setArg(7, colorMap)
              .setArg(8, colorMap.getBuffer().capacity())
              .setArg(9, 200);  // maxIterations

        } catch (IOException ex) {
            Logger.getLogger(Fractal.class.getName()).log(Level.SEVERE, "can not find OpenCL program.", ex);
        } catch (CLException ex) {
            Logger.getLogger(Fractal.class.getName()).log(Level.SEVERE, "something went wrong, hopefully no one got hurt", ex);
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
        gl.glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    }

    private void initPBO(GL gl) {

        int[] pbo = new int[1];
        gl.glGenBuffers(1, pbo, 0);

        // setup empty PBO
        gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[0]);
        gl.glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * SIZEOF_INT, null, GL_STREAM_DRAW);
        gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        pboBuffer = clContext.createFromGLBuffer(null, pbo[0], WRITE_ONLY);

    }

    public void display(GLAutoDrawable drawable) {
        compute();
        render(drawable.getGL().getGL2());
    }

    private void compute() {

        kernel.putArg(pboBuffer)
              .putArg(width).putArg(height)
              .putArg(minX).putArg(minY)
              .putArg(maxX).putArg(maxY)
              .rewind();

        commandQueue.putAcquireGLObject(pboBuffer.ID)
                    .put2DRangeKernel(kernel, 0, 0, width, height, 0, 0)
                    .putReleaseGLObject(pboBuffer.ID)
                    .putBarrier();
    }

    private void render(GL2 gl) {

        gl.glClear(GL_COLOR_BUFFER_BIT);

        gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboBuffer.getGLObjectID());
            gl.glRasterPos2i(0, 0);
            gl.glDrawPixels(width, height, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    }

    public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {

        if(this.width == width && this.height == height)
            return;

        this.width = width;
        this.height = height;

        pboBuffer.release();

        initPBO(drawable.getGL());
        initView(drawable.getGL().getGL2(), drawable.getWidth(), drawable.getHeight());
    }

    private void initSceneInteraction() {

        MouseAdapter adapter = new MouseAdapter() {

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

                maxX    +=-deltaX+offsetX;
                maxY    +=-deltaY-offsetY;

                canvas.display();
            }
        };

        canvas.addMouseMotionListener(adapter);
        canvas.addMouseWheelListener(adapter);
    }

    public void dispose(GLAutoDrawable drawable) {
    }

    public static void main(String args[]) {
        SwingUtilities.invokeLater(new Runnable() {

            public void run() {
                new Fractal(500, 500);
            }
        });
    }

}
