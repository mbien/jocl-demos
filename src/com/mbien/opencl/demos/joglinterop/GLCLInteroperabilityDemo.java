package com.mbien.opencl.demos.joglinterop;

import com.mbien.opencl.CLCommandQueue;
import com.mbien.opencl.CLContext;
import com.mbien.opencl.CLDevice;
import com.mbien.opencl.CLException;
import com.mbien.opencl.CLKernel;
import com.mbien.opencl.CLPlatform;
import com.mbien.opencl.CLProgram;
import com.sun.opengl.util.Animator;
import com.sun.opengl.util.BufferUtil;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import javax.media.opengl.DebugGL2;
import javax.media.opengl.GL2;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.GLProfile;
import javax.media.opengl.awt.GLCanvas;
import javax.media.opengl.glu.gl2.GLUgl2;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import static com.sun.opengl.util.BufferUtil.*;

/**
 * Sample for interoperability between JOCL and JOGL.
 * @author Michael Bien
 */
public class GLCLInteroperabilityDemo implements GLEventListener {

    private final GLUgl2 glu = new GLUgl2();

    private final int GRID_SIZE = 100;
    
    private int width;
    private int height;

    private final FloatBuffer vb;
    private final IntBuffer ib;

    private final int[] buffer = new int[2];
    private final int INDICES  = 0;
    private final int VERTICES = 1;

    private final UserSceneInteraction usi;

    private CLContext clContext;
    private CLKernel kernel;
    private CLCommandQueue commandQueue;
    private final CLProgram program;

    public GLCLInteroperabilityDemo() throws IOException {

        this.usi = new UserSceneInteraction();

        vb = newFloatBuffer(GRID_SIZE * GRID_SIZE * 4);
        ib = newIntBuffer((GRID_SIZE - 1) * (GRID_SIZE - 1) * 2 * 3);

        // build indices
        //    0---3
        //    | \ |
        //    1---2
        for (int h = 0; h < GRID_SIZE - 1; h++) {
            for (int w = 0; w < GRID_SIZE - 1; w++) {

                // 0 - 3 - 2
                ib.put(w * 6 + h * (GRID_SIZE - 1) * 6,      w + (h) * (GRID_SIZE)        );
                ib.put(w * 6 + h * (GRID_SIZE - 1) * 6 + 1,  w + (h) * (GRID_SIZE) + 1    );
                ib.put(w * 6 + h * (GRID_SIZE - 1) * 6 + 2,  w + (h + 1) * (GRID_SIZE) + 1);

                // 0 - 2 - 1
                ib.put(w * 6 + h * (GRID_SIZE - 1) * 6 + 3,  w + (h) * (GRID_SIZE)        );
                ib.put(w * 6 + h * (GRID_SIZE - 1) * 6 + 4,  w + (h + 1) * (GRID_SIZE) + 1);
                ib.put(w * 6 + h * (GRID_SIZE - 1) * 6 + 5,  w + (h + 1) * (GRID_SIZE)    );

            }
        }
        ib.rewind();

        // build grid
        for (int w = 0; w < GRID_SIZE; w++) {
            for (int h = GRID_SIZE; h > 0; h--) {
                vb.put(w - GRID_SIZE / 2).put(h - GRID_SIZE / 2).put(0).put(1);
            }
        }
        vb.rewind();

        try {
            clContext = CLContext.create();
            program = clContext.createProgram(getClass().getResourceAsStream("JoglInterop.cl"));
//            System.out.println(program.getSource());
//            program.build();
        } catch (IOException ex) {
            throw new CLException("can not handle exception", ex);
        }

        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                initUI();
            }
        });

    }

    private void initUI() {

        this.width  = 600;
        this.height = 400;

        GLCapabilities config = new GLCapabilities(GLProfile.get(GLProfile.GL2));
        config.setSampleBuffers(true);
        config.setNumSamples(4);

        GLCanvas canvas = new GLCanvas(config);
        canvas.addGLEventListener(this);
        usi.init(canvas);

        JFrame frame = new JFrame("JOGL-JOCL Interoperability Example");
        frame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosed(WindowEvent e) {
                deinit();
            }

        });
        frame.add(canvas);
        frame.setSize(width, height);

        frame.setVisible(true);

    }


    public void init(GLAutoDrawable drawable) {

        // enable GL error checking using the composable pipeline
        drawable.setGL(new DebugGL2(drawable.getGL().getGL2()));

        GL2 gl = drawable.getGL().getGL2();

        gl.glPolygonMode(GL2.GL_FRONT_AND_BACK, GL2.GL_LINE);
        
        gl.glGenBuffers(buffer.length, buffer, 0);
        
        gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, buffer[INDICES]);
        gl.glBufferData(GL2.GL_ARRAY_BUFFER, ib.capacity() * SIZEOF_FLOAT, ib, GL2.GL_STATIC_DRAW);
        gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, 0);

        gl.glBindBuffer(GL2.GL_ELEMENT_ARRAY_BUFFER, buffer[VERTICES]);
        gl.glBufferData(GL2.GL_ELEMENT_ARRAY_BUFFER, vb.capacity() * SIZEOF_FLOAT, vb, GL2.GL_DYNAMIC_DRAW);
        gl.glBindBuffer(GL2.GL_ELEMENT_ARRAY_BUFFER, 0);

        // OpenCL
//        commandQueue = clContext.getMaxFlopsDevice().createCommandQueue();

//            kernel = program.getCLKernel("sineWave");
//        CLBuffer<FloatBuffer> clBuffer = clContext.createFromGLBuffer(vb, buffer[VERTICES], CLBuffer.Mem.WRITE_ONLY);
//        kernel.setArg(0, clBuffer);
//        kernel.setArg(1, GRID_SIZE);
//        kernel.setArg(2, GRID_SIZE);
        

        pushPerspectiveView(gl);

        // start rendering thread
        Animator animator = new Animator(drawable);
        animator.start();
    }

    public void display(GLAutoDrawable drawable) {

        GL2 gl = drawable.getGL().getGL2();
        gl.glClear(GL2.GL_COLOR_BUFFER_BIT | GL2.GL_DEPTH_BUFFER_BIT);
        gl.glLoadIdentity();

        usi.interact(gl);

        gl.glEnableClientState(GL2.GL_VERTEX_ARRAY);

            gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, buffer[VERTICES]);
            gl.glVertexPointer(4, GL2.GL_FLOAT, 0, 0);

            gl.glBindBuffer(GL2.GL_ELEMENT_ARRAY_BUFFER, buffer[INDICES]);
            gl.glDrawElements(GL2.GL_TRIANGLES, ib.capacity(), GL2.GL_UNSIGNED_INT, 0);

        gl.glDisableClientState(GL2.GL_VERTEX_ARRAY);

        gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, 0);

    }

    private void pushPerspectiveView(GL2 gl) {

        gl.glMatrixMode(GL2.GL_PROJECTION);
        gl.glPushMatrix();

            gl.glLoadIdentity();

            glu.gluPerspective(60, width / (float)height, 1, 1000);
            gl.glMatrixMode(GL2.GL_MODELVIEW);

            gl.glPushMatrix();
                gl.glLoadIdentity();

    }

    private void popView(GL2 gl) {

                gl.glMatrixMode(GL2.GL_PROJECTION);
            gl.glPopMatrix();

            gl.glMatrixMode(GL2.GL_MODELVIEW);
        gl.glPopMatrix();

    }


    public void reshape(GLAutoDrawable drawable, int arg1, int arg2, int width, int height) {
        this.width = width;
        this.height = height;
        GL2 gl = drawable.getGL().getGL2();
        popView(gl);
        pushPerspectiveView(gl);
    }

    public void dispose(GLAutoDrawable drawable) {  }
    
    private void deinit() {
        clContext.release();
        System.exit(0);
    }

    public static void main(String[] args) throws IOException {
        new GLCLInteroperabilityDemo();
    }

}
