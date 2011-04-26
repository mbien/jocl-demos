package com.jogamp.opencl.demos.fractal;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLEvent;
import com.jogamp.opencl.CLEventList;
import com.jogamp.opencl.CLException;
import com.jogamp.opencl.gl.CLGLBuffer;
import com.jogamp.opencl.gl.CLGLContext;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLProgram.CompilerOptions;
import com.jogamp.opencl.util.CLProgramConfiguration;
import com.jogamp.opengl.util.awt.TextRenderer;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Frame;
import java.awt.Point;
import java.awt.Window;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
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
import javax.swing.SwingUtilities;

import static com.jogamp.common.nio.Buffers.*;
import static javax.media.opengl.GL2.*;
import static com.jogamp.opencl.CLMemory.Mem.*;
import static com.jogamp.opencl.CLDevice.Type.*;
import static com.jogamp.opencl.CLEvent.ProfilingCommand.*;
import static com.jogamp.opencl.CLCommandQueue.Mode.*;
import static java.lang.Math.*;

/**
 * Computes the Mandelbrot set with OpenCL using multiple GPUs and renders the result with OpenGL.
 * A shared PBO is used as storage for the fractal image.<br/>
 * http://en.wikipedia.org/wiki/Mandelbrot_set
 * <p>
 * controls:<br/>
 * keys 1-9 control parallelism level<br/>
 * space enables/disables slice seperator<br/>
 * 'd' toggles between 32/64bit floatingpoint precision<br/>
 * mouse/mousewheel to drag and zoom<br/>
 * </p>
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
    private CLProgram[] programs;
    private CLEventList probes;
    private CLGLBuffer<?>[] pboBuffers;
    private CLBuffer<IntBuffer>[] colorMap;

    private int width  = 0;
    private int height = 0;

    private double minX = -2f;
    private double minY = -1.2f;
    private double maxX  = 0.6f;
    private double maxY  = 1.3f;

    private int slices;

    private boolean drawSeperator;
    private boolean doublePrecision;
    private boolean buffersInitialized;
    private boolean rebuild;

    private final TextRenderer textRenderer;

    public MultiDeviceFractal(int width, int height) {

        this.width = width;
        this.height = height;

        canvas = new GLCanvas(new GLCapabilities(GLProfile.get(GLProfile.GL2)));
        canvas.addGLEventListener(this);
        initSceneInteraction();

        Frame frame = new Frame("JOCL Multi Device Mandelbrot Set");
        frame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                MultiDeviceFractal.this.release(e.getWindow());
            }
        }); 
        canvas.setPreferredSize(new Dimension(width, height));
        frame.add(canvas);
        frame.pack();

        frame.setVisible(true);

        textRenderer = new TextRenderer(frame.getFont().deriveFont(Font.BOLD, 14), true, true, null, false);
    }

    @Override
    public void init(GLAutoDrawable drawable) {

        if(clContext == null) {
            // enable GL error checking using the composable pipeline
            drawable.setGL(new DebugGL2(drawable.getGL().getGL2()));

            drawable.getGL().glFinish();
            initCL(drawable.getContext());

            GL2 gl = drawable.getGL().getGL2();

            gl.setSwapInterval(0);
            gl.glDisable(GL_DEPTH_TEST);
            gl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

            initView(gl, drawable.getWidth(), drawable.getHeight());

            initPBO(gl);
            drawable.getGL().glFinish();

            setKernelConstants();
        }
    }

    private void initCL(GLContext glCtx){
        try {
            CLPlatform platform = CLPlatform.getDefault();
            // SLI on NV platform wasn't very fast (or did not work at all -> CL_INVALID_OPERATION)
            if(platform.getICDSuffix().equals("NV")) {
                clContext = CLGLContext.create(glCtx, platform.getMaxFlopsDevice(GPU));
            }else{
                clContext = CLGLContext.create(glCtx, platform, ALL);
            }
            CLDevice[] devices = clContext.getDevices();

            slices = min(devices.length, MAX_PARRALLELISM_LEVEL);

            // create command queues for every GPU, setup colormap and init kernels
            queues = new CLCommandQueue[slices];
            kernels = new CLKernel[slices];
            probes = new CLEventList(slices);
            colorMap = new CLBuffer[slices];

            for (int i = 0; i < slices; i++) {

                colorMap[i] = clContext.createIntBuffer(32*2, READ_ONLY);
                initColorMap(colorMap[i].getBuffer(), 32, Color.BLUE, Color.GREEN, Color.RED);

                // create command queue and upload color map buffer on each used device
                queues[i] = devices[i].createCommandQueue(PROFILING_MODE).putWriteBuffer(colorMap[i], true); // blocking upload

            }

            // check if we have 64bit FP support on all devices
            // if yes we can use only one program for all devices + one kernel per device.
            // if not we will have to create (at least) one program for 32 and one for 64bit devices.
            // since there are different vendor extensions for double FP we use one program per device.
            // (OpenCL spec is not very clear about this usecases)
            boolean all64bit = true;
            for (CLDevice device : devices) {
                if(!isDoubleFPAvailable(device)) {
                    all64bit = false;
                    break;
                }
            }

            // load program(s)
            if(all64bit) {
                programs = new CLProgram[] {
                    clContext.createProgram(getClass().getResourceAsStream("Mandelbrot.cl"))
                };
            }else{
                programs = new CLProgram[slices];
                for (int i = 0; i < slices; i++) {
                    programs[i] = clContext.createProgram(getClass().getResourceAsStream("Mandelbrot.cl"));
                }
            }

            buildProgram();

        } catch (IOException ex) {
            Logger.getLogger(getClass().getName()).log(Level.SEVERE, "can not find 'Mandelbrot.cl' in classpath.", ex);
            if(clContext != null) {
                clContext.release();
            }
        } catch (CLException ex) {
            Logger.getLogger(getClass().getName()).log(Level.SEVERE, "something went wrong, hopefully nobody got hurt", ex);
            if(clContext != null) {
                clContext.release();
            }
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

        if(pboBuffers != null) {
            int[] oldPbos = new int[pboBuffers.length];
            for (int i = 0; i < pboBuffers.length; i++) {
                CLGLBuffer<?> buffer = pboBuffers[i];
                oldPbos[i] = buffer.GLID;
                buffer.release();
            }
            gl.glDeleteBuffers(oldPbos.length, oldPbos, 0);
        }

        pboBuffers = new CLGLBuffer[slices];

        int[] pbo = new int[slices];
        gl.glGenBuffers(slices, pbo, 0);

        // setup one empty PBO per slice
        for (int i = 0; i < slices; i++) {

            int length = width*height * SIZEOF_INT / slices;
            gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[i]);
            gl.glBufferData(GL_PIXEL_UNPACK_BUFFER, length, null, GL_STREAM_DRAW);
            gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            pboBuffers[i] = clContext.createFromGLBuffer(pbo[i], length, WRITE_ONLY);

        }

        buffersInitialized = true;
    }

    private void buildProgram() {

        /*
         * workaround: The driver keeps using the old binaries for some reason.
         * to solve this we simple create a new program and release the old.
         * however rebuilding programs should be possible -> remove when drivers are fixed.
         * (again: the spec is not very clear about this kind of usages)
         */
        if(programs[0] != null && rebuild) {
            for(int i = 0; i < programs.length; i++) {
                String source = programs[i].getSource();
                programs[i].release();
                programs[i] = clContext.createProgram(source);
            }
        }

        // disable 64bit floating point math if not available
        for(int i = 0; i < programs.length; i++) {
            CLDevice device = queues[i].getDevice();

            CLProgramConfiguration configure = programs[i].prepare();
            if(doublePrecision && isDoubleFPAvailable(device)) {
                //cl_khr_fp64
                configure.withDefine("DOUBLE_FP");

                //amd's verson of double precision floating point math
                if(!device.isDoubleFPAvailable() && device.isExtensionAvailable("cl_amd_fp64")) {
                    configure.withDefine("AMD_FP");
                }
            }
            if(programs.length > 1) {
                configure.forDevice(device);
            }
            System.out.println(configure);
            configure.withOption(CompilerOptions.FAST_RELAXED_MATH).build();
         }

        rebuild = false;

        for (int i = 0; i < kernels.length; i++) {
            // init kernel with constants
            kernels[i] = programs[min(i, programs.length)].createCLKernel("mandelbrot");
        }

    }

    // init kernels with constants
    private void setKernelConstants() {
        for (int i = 0; i < slices; i++) {
            kernels[i].setForce32BitArgs(!doublePrecision || !isDoubleFPAvailable(queues[i].getDevice()))
                      .setArg(6, pboBuffers[i])
                      .setArg(7, colorMap[i])
                      .setArg(8, colorMap[i].getBuffer().capacity())
                      .setArg(9, MAX_ITERATIONS);
        }
    }

    // rendering cycle
    @Override
    public void display(GLAutoDrawable drawable) {
        GL gl = drawable.getGL();

        // make sure GL does not use our objects before we start computeing
        gl.glFinish();
        if(!buffersInitialized) {
            initPBO(gl);
            setKernelConstants();
        }
        if(rebuild) {
            buildProgram();
            setKernelConstants();
        }
        compute();

        render(gl.getGL2());
    }

    // OpenCL
    private void compute() {

        int sliceWidth = (int)(width / (float)slices);
        double rangeX  = (maxX - minX) / slices;
        double rangeY  = (maxY - minY);

        // release all old events, you can't reuse events in OpenCL
        probes.release();

        // start computation
        for (int i = 0; i < slices; i++) {

            kernels[i].putArg(     sliceWidth).putArg(height)
                      .putArg(minX + rangeX*i).putArg(  minY)
                      .putArg(       rangeX  ).putArg(rangeY)
                      .rewind();

            // aquire GL objects, and enqueue a kernel with a probe from the list
            queues[i].putAcquireGLObject(pboBuffers[i])
                     .put2DRangeKernel(kernels[i], 0, 0, sliceWidth, height, 0, 0, probes)
                     .putReleaseGLObject(pboBuffers[i]);

        }

        // block until done (important: finish before doing further gl work)
        for (int i = 0; i < slices; i++) {
            queues[i].finish();
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

            textRenderer.draw("device/time/precision", 10, height-15);

            for (int i = 0; i < slices; i++) {
                CLDevice device = queues[i].getDevice();
                boolean doubleFP = doublePrecision && isDoubleFPAvailable(device);
                CLEvent event = probes.getEvent(i);
                long start = event.getProfilingInfo(START);
                long end = event.getProfilingInfo(END);
                textRenderer.draw(device.getType().toString()+i +" "
                               + (int)((end-start)/1000000.0f)+"ms @"
                               + (doubleFP?"64bit":"32bit"), 10, height-(20+16*(slices-i)));
            }

        textRenderer.endRendering();
    }

    @Override
    public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {

        if(this.width == width && this.height == height)
            return;

        this.width = width;
        this.height = height;

        initPBO(drawable.getGL());
        setKernelConstants();

        initView(drawable.getGL().getGL2(), width, height);
        
    }

    private void initSceneInteraction() {

        MouseAdapter mouseAdapter = new MouseAdapter() {

            Point lastpos = new Point();

            @Override
            public void mouseDragged(MouseEvent e) {
                
                double offsetX = (lastpos.x - e.getX()) * (maxX - minX) / width;
                double offsetY = (lastpos.y - e.getY()) * (maxY - minY) / height;

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

                double deltaX = rotation * (maxX - minX);
                double deltaY = rotation * (maxY - minY);

                // offset for "zoom to cursor"
                double offsetX = (e.getX() / (float)width - 0.5f) * deltaX * 2;
                double offsetY = (e.getY() / (float)height- 0.5f) * deltaY * 2;

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
                    buffersInitialized = false;
                }else if(e.getKeyCode() == KeyEvent.VK_D) {
                    doublePrecision = !doublePrecision;
                    rebuild = true;
                }
                canvas.display();
            }

        };

        canvas.addMouseMotionListener(mouseAdapter);
        canvas.addMouseWheelListener(mouseAdapter);
        canvas.addKeyListener(keyAdapter);
    }


    private boolean isDoubleFPAvailable(CLDevice device) {
        return device.isDoubleFPAvailable() || device.isExtensionAvailable("cl_amd_fp64");
    }

    private void release(Window win) {
        if(clContext != null) {
            // releases all resources
            clContext.release();
        }
        win.dispose();
    }

    @Override
    public void dispose(GLAutoDrawable drawable) {
    }

    public static void main(String args[]) {
        
        //false for webstart compatibility
        GLProfile.initSingleton(false);
        
        SwingUtilities.invokeLater(new Runnable() {
            @Override public void run() {
                new MultiDeviceFractal(512, 512);
            }
        });
    }

}
