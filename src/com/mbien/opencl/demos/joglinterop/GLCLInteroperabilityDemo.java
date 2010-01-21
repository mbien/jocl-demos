package com.mbien.opencl.demos.joglinterop;

import com.mbien.opencl.CLCommandQueue;
import com.mbien.opencl.CLGLBuffer;
import com.mbien.opencl.CLGLContext;
import com.mbien.opencl.CLKernel;
import com.mbien.opencl.CLProgram;
import com.sun.opengl.util.Animator;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;
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
 * JOCL - JOGL interoperability example.
 * @author Michael Bien
 */
public class GLCLInteroperabilityDemo implements GLEventListener {

    private final GLUgl2 glu = new GLUgl2();

    private final int MESH_SIZE = 256;

    private int width;
    private int height;

//    private final FloatBuffer vb;
//    private final IntBuffer ib;

    private final int[] glObjects = new int[2];
    private final int VERTICES = 0;
//    private final int INDICES  = 1;

    private final UserSceneInteraction usi;

    private CLGLContext clContext;
    private CLKernel kernel;
    private CLCommandQueue commandQueue;
    private CLGLBuffer<?> clBuffer;

    private float step = 0;

    public GLCLInteroperabilityDemo() {

        this.usi = new UserSceneInteraction();

        // create direct memory buffers
//        vb = newFloatBuffer(MESH_SIZE * MESH_SIZE * 4);
//        ib = newIntBuffer((MESH_SIZE - 1) * (MESH_SIZE - 1) * 2 * 3);
//
//        // build indices
//        //    0---3
//        //    | \ |
//        //    1---2
//        for (int h = 0; h < MESH_SIZE - 1; h++) {
//            for (int w = 0; w < MESH_SIZE - 1; w++) {
//
//                // 0 - 3 - 2
//                ib.put(w * 6 + h * (MESH_SIZE - 1) * 6,      w + (h    ) * (MESH_SIZE)    );
//                ib.put(w * 6 + h * (MESH_SIZE - 1) * 6 + 1,  w + (h    ) * (MESH_SIZE) + 1);
//                ib.put(w * 6 + h * (MESH_SIZE - 1) * 6 + 2,  w + (h + 1) * (MESH_SIZE) + 1);
//
//                // 0 - 2 - 1
//                ib.put(w * 6 + h * (MESH_SIZE - 1) * 6 + 3,  w + (h    ) * (MESH_SIZE)    );
//                ib.put(w * 6 + h * (MESH_SIZE - 1) * 6 + 4,  w + (h + 1) * (MESH_SIZE) + 1);
//                ib.put(w * 6 + h * (MESH_SIZE - 1) * 6 + 5,  w + (h + 1) * (MESH_SIZE)    );
//
//            }
//        }
//        ib.rewind();

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

        // create OpenCL context before creating any OpenGL objects
        // you want to share with OpenCL (AMD driver requirement)
        clContext = CLGLContext.create(drawable.getContext());

        // enable GL error checking using the composable pipeline
        drawable.setGL(new DebugGL2(drawable.getGL().getGL2()));

        // OpenGL initialization
        GL2 gl = drawable.getGL().getGL2();

        gl.setSwapInterval(1);

        gl.glPolygonMode(GL2.GL_FRONT_AND_BACK, GL2.GL_LINE);

        gl.glGenBuffers(glObjects.length, glObjects, 0);

//        gl.glBindBuffer(GL2.GL_ELEMENT_ARRAY_BUFFER, glObjects[INDICES]);
//        gl.glBufferData(GL2.GL_ELEMENT_ARRAY_BUFFER, ib.capacity() * SIZEOF_INT, ib, GL2.GL_STATIC_DRAW);
//        gl.glBindBuffer(GL2.GL_ELEMENT_ARRAY_BUFFER, 0);

        gl.glEnableClientState(GL2.GL_VERTEX_ARRAY);
            gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, glObjects[VERTICES]);
            gl.glBufferData(GL2.GL_ARRAY_BUFFER, MESH_SIZE * MESH_SIZE * 4 * SIZEOF_FLOAT, null, GL2.GL_DYNAMIC_DRAW);
            gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, 0);
        gl.glDisableClientState(GL2.GL_VERTEX_ARRAY);

        gl.glFinish();

        // init OpenCL
        initCL();

        pushPerspectiveView(gl);

        // start rendering thread
        Animator animator = new Animator(drawable);
        animator.start();
    }

    private void initCL() {

        CLProgram program;
        try {
            program = clContext.createProgram(getClass().getResourceAsStream("JoglInterop.cl"));
            program.build();
            System.out.println(program.getBuildLog());
            System.out.println(program.getBuildStatus());
        } catch (IOException ex) {
            throw new RuntimeException("can not handle exception", ex);
        }

        commandQueue = clContext.getMaxFlopsDevice().createCommandQueue();

        clBuffer = clContext.createFromGLBuffer(null, glObjects[VERTICES], CLGLBuffer.Mem.WRITE_ONLY);

        System.out.println("cl buffer type: " + clBuffer.getGLObjectType());
        System.out.println("shared with gl buffer: " + clBuffer.getGLObjectID());

        kernel = program.getCLKernel("sineWave")
                        .putArg(clBuffer)
                        .putArg(MESH_SIZE)
                        .rewind();

        System.out.println("cl initialised");
    }


    public void display(GLAutoDrawable drawable) {

        computeHeightfield();

        GL2 gl = drawable.getGL().getGL2();

        gl.glClear(GL2.GL_COLOR_BUFFER_BIT | GL2.GL_DEPTH_BUFFER_BIT);
        gl.glLoadIdentity();

        usi.interact(gl);

        gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, glObjects[VERTICES]);
        gl.glVertexPointer(4, GL2.GL_FLOAT, 0, 0);

//            gl.glBindBuffer(GL2.GL_ELEMENT_ARRAY_BUFFER, glObjects[INDICES]);

        gl.glEnableClientState(GL2.GL_VERTEX_ARRAY);
        gl.glDrawArrays(GL2.GL_POINTS, 0, MESH_SIZE * MESH_SIZE);
//            gl.glDrawElements(GL2.GL_TRIANGLES, ib.capacity(), GL2.GL_UNSIGNED_INT, 0);
        gl.glDisableClientState(GL2.GL_VERTEX_ARRAY);

//            gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, 0);

        gl.glFinish();
        
    }

    /*
     * Computes a heightfield using a OpenCL kernel.
     */
    private void computeHeightfield() {

        kernel.setArg(2, step += 0.05f);

        commandQueue.putAcquireGLObject(clBuffer.ID)
                    .put2DRangeKernel(kernel, 0, 0, MESH_SIZE, MESH_SIZE, 0, 0)
                    .putReleaseGLObject(clBuffer.ID)
                    .finish();

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
        clContext.release(); // only for demonstration purposes, JVM will cleanup on exit anyway
        System.exit(0);
    }

    public static void main(String[] args) {
        new GLCLInteroperabilityDemo();
    }

}