package com.jogamp.opencl.demos.julia3d;

import java.awt.event.WindowEvent;
import java.awt.event.WindowAdapter;
import java.awt.Canvas;
import java.awt.Dimension;
import javax.swing.JFrame;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.demos.julia3d.structs.Camera;
import com.jogamp.opencl.demos.julia3d.structs.RenderingConfig;
import com.jogamp.opencl.demos.julia3d.structs.Vec;
import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import javax.media.opengl.GLProfile;
import javax.swing.SwingUtilities;

import static com.jogamp.opencl.CLDevice.Type.*;
import static com.jogamp.opencl.util.CLPlatformFilters.*;
import static com.jogamp.opencl.CLMemory.Mem.*;
import static com.jogamp.opencl.CLProgram.CompilerOptions.*;
import static com.jogamp.opencl.demos.julia3d.UserSceneController.*;

/**
 * This sample has been ported from David Buciarelli's juliaGPU v1.2 written in C.
 * @author Michael Bien
 */
public class Julia3d {

    private final CLContext context;
    private CLBuffer<FloatBuffer> pixelBuffer;
    private final CLBuffer<ByteBuffer> configBuffer;
    private final CLCommandQueue commandQueue;
    private final CLProgram program;
    private final CLKernel julia;
    private final CLKernel multiply;

    private final int workGroupSize;
    private final String kernelFileName = "rendering_kernel.cl";

    final RenderingConfig config;

    private Julia3d(RenderingConfig renderConfig) {
        this.config = renderConfig;
        updateCamera();

        //setup, prefere GPUs
        CLDevice device = CLPlatform.getDefault(type(GPU)).getMaxFlopsDevice();
        if(device == null) {
            device = CLPlatform.getDefault().getMaxFlopsDevice();
        }
        context = CLContext.create(device);

        workGroupSize = Math.min(256, device.getMaxWorkGroupSize());

        //allocate buffers
        configBuffer = context.createBuffer(config.getBuffer(), READ_ONLY);
        commandQueue = device.createCommandQueue();
//        update(true);

        try {
            program = context.createProgram(Julia3d.class.getResourceAsStream(kernelFileName))
                             .build(FAST_RELAXED_MATH);
        } catch (IOException ex) {
            throw new RuntimeException("unable to load program from source", ex);
        }

        julia = program.createCLKernel("JuliaGPU");
        multiply = program.createCLKernel("multiply");
        System.out.println(program.getBuildStatus(device));
        System.out.println(program.getBuildLog());

    }

    void update(boolean reallocate) {

        updateCamera();

        int bufferSize = config.getWidth() * config.getHeight() * 3;
        if(reallocate) {
            if(pixelBuffer != null) {
                pixelBuffer.release();
            }

            pixelBuffer = context.createFloatBuffer(bufferSize, READ_WRITE, USE_BUFFER);
        }

        commandQueue.putWriteBuffer(configBuffer, true);
        
        julia.putArg(pixelBuffer)
             .putArg(configBuffer)
             .rewind();

        multiply.putArg(pixelBuffer)
                .putArg(bufferSize)
                .rewind();
    }


    void compute(boolean fastRendering) {

        // calculate workgroup size
        int globalThreads = config.getWidth() * config.getHeight();
        if(globalThreads % workGroupSize != 0)
            globalThreads = (globalThreads / workGroupSize + 1) * workGroupSize;

        int localThreads = workGroupSize;
        int superSamplingSize = config.getSuperSamplingSize();

        if (!fastRendering && superSamplingSize > 1) {
            
            for (int y = 0; y < superSamplingSize; ++y) {
                for (int x = 0; x < superSamplingSize; ++x) {
                    
                    float sampleX = (x + 0.5f) / superSamplingSize;
                    float sampleY = (y + 0.5f) / superSamplingSize;
                    
                    if (x == 0 && y == 0) {
                        // First pass
                        julia.setArg(2, 0)
                             .setArg(3, sampleX)
                             .setArg(4, sampleY);

                        commandQueue.put1DRangeKernel(julia, 0, globalThreads, localThreads);
                            
                    } else if (x == (superSamplingSize - 1) && y == (superSamplingSize - 1)) {
                        // Last pass
                        julia.setArg(2, 1)
                             .setArg(3, sampleX)
                             .setArg(4, sampleY);

                        // normalize the values we accumulated
                        multiply.setArg(2, 1.0f/(superSamplingSize*superSamplingSize));
                        
                        commandQueue.put1DRangeKernel(julia,    0, globalThreads,   localThreads)
                                    .put1DRangeKernel(multiply, 0, globalThreads*3, localThreads);
                    } else {
                        julia.setArg(2, 1)
                             .setArg(3, sampleX)
                             .setArg(4, sampleY);
                        
                        commandQueue.put1DRangeKernel(julia, 0, globalThreads, localThreads);
                        
                    }
                }
            }
            
        }else{
            
            //fast rendering
            julia.setArg(2, 0)
                 .setArg(3, 0.0f)
                 .setArg(4, 0.0f);

            commandQueue.put1DRangeKernel(julia, 0, globalThreads, localThreads);
        }
        
        commandQueue.putBarrier()
                    .putReadBuffer(pixelBuffer, true);

    }

    private void updateCamera() {

        Camera camera = config.getCamera();

        Vec dir    = camera.getDir();
        Vec target = camera.getTarget();
        Vec camX   = camera.getX();
        Vec camY   = camera.getY();
        Vec orig   = camera.getOrig();

        vsub(dir, target, orig);
        vnorm(dir);

        Vec up = Vec.create().setX(0).setY(1).setZ(0);
        vxcross(camX, dir, up);
        vnorm(camX);
        vmul(camX, config.getWidth() * .5135f / config.getHeight(), camX);

        vxcross(camY, camX, dir);
        vnorm(camY);
        vmul(camY, .5135f, camY);
    }

    CLDevice getDevice() {
        return commandQueue.getDevice();
    }


    public static void main(String[] args) {
        
        //false for webstart compatibility
        GLProfile.initSingleton(false);
        
        final RenderingConfig config = RenderingConfig.create()
            .setWidth(640).setHeight(480)
            .setEnableShadow(1)
            .setSuperSamplingSize(2)
            .setActvateFastRendering(1)
            .setMaxIterations(9)
            .setEpsilon(0.003f * 0.75f)
            .setLight(new float[] {5, 10, 15})
            .setMu(new float[] {-0.2f, 0.4f, -0.4f, -0.4f});

        config.getCamera().getOrig()  .setX(1).setY(2).setZ(8);
        config.getCamera().getTarget().setX(0).setY(0).setZ(0);

        final Julia3d julia3d = new Julia3d(config);

        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                
                Renderer renderer = new Renderer(julia3d);
                CLDevice device = julia3d.getDevice();
                
                JFrame frame = new JFrame("Java OpenCL - Julia3D "+device.getType()+" "+device.getName());
                frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                frame.addWindowListener(new WindowAdapter() {
                    @Override
                    public void windowClosed(WindowEvent e) {
                        julia3d.release();
                        System.exit(0);
                    }
                });
                Canvas canvas = renderer.getCanvas();
                canvas.setPreferredSize(new Dimension(config.getWidth(), config.getHeight()));
                frame.add(canvas);
                frame.pack();
                frame.setVisible(true);
                
            }
        });
    }

    Buffer getPixelBuffer() {
        return pixelBuffer.getBuffer();
    }

    void release() {
        context.release();
    }


}
