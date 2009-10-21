package com.mbien.opencl.demos.hellojocl;

import com.mbien.opencl.CL;
import com.mbien.opencl.CLBuffer;
import com.mbien.opencl.CLCommandQueue;
import com.mbien.opencl.CLContext;
import com.mbien.opencl.CLKernel;
import com.mbien.opencl.CLProgram;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Random;

import static java.lang.System.*;
import static com.sun.gluegen.runtime.BufferFactory.*;

/**
 * Hello Java OpenCL example. Adds all elements of buffer A to buffer B
 * and stores the result in buffer C.<br/>
 * Sample was inspired by the Nvidia VectorAdd example written in C/C++
 * which is bundled in the Nvidia OpenCL SDK.
 * @author Michael Bien
 */
public class HelloJOCL {

    public static void main(String[] args) throws IOException {
        
        int elementCount = 11444777;                                // Length of float arrays to process
        int localWorkSize = 256;                                    // set and log Global and Local work size dimensions
        int globalWorkSize = roundUp(localWorkSize, elementCount);  // rounded up to the nearest multiple of the localWorkSize

        // set up
        CLContext context = CLContext.create();

        CLProgram program = context.createProgram(HelloJOCL.class.getResourceAsStream("VectorAdd.cl")).build();

        CLBuffer clBufferA = context.createBuffer(CL.CL_MEM_READ_ONLY,  globalWorkSize*SIZEOF_INT);
        CLBuffer clBufferB = context.createBuffer(CL.CL_MEM_READ_ONLY,  globalWorkSize*SIZEOF_INT);
        CLBuffer clBufferC = context.createBuffer(CL.CL_MEM_WRITE_ONLY, globalWorkSize*SIZEOF_INT);

        out.println("used device memory: "
            + (clBufferA.buffer.capacity()+clBufferB.buffer.capacity()+clBufferC.buffer.capacity())/1000000 +"MB");

        // fill read buffers with random numbers (just to have test data; seed is fixed -> results will not change between).
        fillBuffer(clBufferA.buffer, 12345);
        fillBuffer(clBufferB.buffer, 67890);

        // get a reference to the kernel functon with the name 'VectorAdd' and map the buffers to its input parameters.
        CLKernel kernel = program.getCLKernels().get("VectorAdd");
        kernel.setArg(0, SIZEOF_LONG, clBufferA)
              .setArg(1, SIZEOF_LONG, clBufferB)
              .setArg(2, SIZEOF_LONG, clBufferC)
              .setArg(3, SIZEOF_INT, elementCount);

        // create command queue on first device.
        CLCommandQueue queue = context.getCLDevices()[0].createCommandQueue();

        // asynchronous write of data to GPU device, blocking read later to get the computed results back.
        long time = nanoTime();
        queue.putWriteBuffer(clBufferA, false)
             .putWriteBuffer(clBufferB, false)
             .putNDRangeKernel(kernel, 1, null, new long[]{ globalWorkSize }, new long[]{ localWorkSize })
             .putReadBuffer(clBufferC, true);
        time = nanoTime() - time;

        // cleanup all resources associated with this context.
        context.release();

        // print first few elements of the resulting buffer to the console.
        out.println("a+b=c results snapshot: ");
        for(int i = 0; i < 10; i++)
            out.print(clBufferC.buffer.getInt() + ", ");
        out.println("...; " + clBufferC.buffer.remaining()/SIZEOF_INT + " more");
        
        System.out.println("computation took: "+(time/1000000)+"ms");

    }

    public static final void fillBuffer(ByteBuffer buffer, int seed) {

        Random rnd = new Random(seed);

        while(buffer.remaining() != 0)
            buffer.putInt(rnd.nextInt());

        buffer.rewind();
    }

    public static final int roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }

}
