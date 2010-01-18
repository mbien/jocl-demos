package com.mbien.opencl.demos.hellojocl;

import com.mbien.opencl.CLBuffer;
import com.mbien.opencl.CLCommandQueue;
import com.mbien.opencl.CLContext;
import com.mbien.opencl.CLKernel;
import com.mbien.opencl.CLProgram;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Random;

import static java.lang.System.*;
import static com.mbien.opencl.CLMemory.Mem.*;

/**
 * Hello Java OpenCL example. Adds all elements of buffer A to buffer B
 * and stores the result in buffer C.<br/>
 * Sample was inspired by the Nvidia VectorAdd example written in C/C++
 * which is bundled in the Nvidia OpenCL SDK.
 * @author Michael Bien
 */
public class HelloJOCL {

    public static void main(String[] args) throws IOException {

        int elementCount = 11444777;                                // Length of arrays to process
        int localWorkSize = 256;                                    // Local work size dimensions
        int globalWorkSize = roundUp(localWorkSize, elementCount);  // rounded up to the nearest multiple of the localWorkSize

        // set up
        CLContext context = CLContext.create();

        CLProgram program = context.createProgram(HelloJOCL.class.getResourceAsStream("VectorAdd.cl")).build();

        CLBuffer<FloatBuffer> clBufferA = context.createFloatBuffer(globalWorkSize, READ_ONLY);
        CLBuffer<FloatBuffer> clBufferB = context.createFloatBuffer(globalWorkSize, READ_ONLY);
        CLBuffer<FloatBuffer> clBufferC = context.createFloatBuffer(globalWorkSize, WRITE_ONLY);

        out.println("used device memory: "
            + (clBufferA.getSize()+clBufferB.getSize()+clBufferC.getSize())/1000000 +"MB");

        // fill read buffers with random numbers (just to have test data; seed is fixed -> results will not change between runs).
        fillBuffer(clBufferA.getBuffer(), 12345);
        fillBuffer(clBufferB.getBuffer(), 67890);

        // get a reference to the kernel functon with the name 'VectorAdd'
        // and map the buffers to its input parameters.
        CLKernel kernel = program.getCLKernel("VectorAdd");
        kernel.putArgs(clBufferA, clBufferB, clBufferC).putArg(elementCount);

        // create command queue on fastest device.
        CLCommandQueue queue = context.getMaxFlopsDevice().createCommandQueue();

        // asynchronous write of data to GPU device, blocking read later to get the computed results back.
        long time = nanoTime();
        queue.putWriteBuffer(clBufferA, false)
             .putWriteBuffer(clBufferB, false)
             .put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize)
             .putReadBuffer(clBufferC, true);
        time = nanoTime() - time;

        // cleanup all resources associated with this context.
        context.release();

        // print first few elements of the resulting buffer to the console.
        out.println("a+b=c results snapshot: ");
        for(int i = 0; i < 10; i++)
            out.print(clBufferC.getBuffer().get() + ", ");
        out.println("...; " + clBufferC.getBuffer().remaining() + " more");

        out.println("computation took: "+(time/1000000)+"ms");

    }

    private static final void fillBuffer(FloatBuffer buffer, int seed) {
        Random rnd = new Random(seed);
        while(buffer.remaining() != 0)
            buffer.put(rnd.nextFloat()*100);
        buffer.rewind();
    }

    private static final int roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }

}