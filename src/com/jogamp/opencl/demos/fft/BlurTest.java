package com.jogamp.opencl.demos.fft;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.demos.fft.CLFFTPlan.InvalidContextException;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.swing.BoxLayout;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.JToggleButton;
import javax.swing.SwingUtilities;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

/**
 * Perform some user-controllable blur on an image.
 * @author notzed
 */
public class BlurTest implements Runnable, ChangeListener, ActionListener {

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new BlurTest());
    }
    
    boolean demo = false;
    // must be power of 2 and width must be multiple of 64
    int width = 512;
    int height = 512;
    BufferedImage src;
    BufferedImage psf;
    BufferedImage dst;
    PaintView left;
    ImageView right;
    //
    JSlider sizex;
    JSlider sizey;
    JSlider angle;
    //
    JToggleButton blurButton;
    JToggleButton drawButton;

    public void run() {
        try {
            initCL();
        } catch (Exception x) {
            System.out.println("failed to init cl");
            x.printStackTrace();
            System.exit(1);
        }

        JFileChooser fc = new JFileChooser();
        BufferedImage img = null;

        while (img == null) {
            try {
                File file = null;

                if (true) {
                    fc.setDialogTitle("Select Image File");
                    fc.setPreferredSize(new Dimension(500, 600));
                    if (fc.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
                        file = fc.getSelectedFile();
                    } else {
                        System.exit(0);
                    }

                } else {
                    file = new File("/home/notzed/cat0.jpg");
                }
                img = ImageIO.read(file);
                if (img == null) {
                    JOptionPane.showMessageDialog(null, "Couldn't load file");
                }
            } catch (IOException x) {
                JOptionPane.showMessageDialog(null, "Couldn't load file");
            }
        }

        src = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        dst = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        psf = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        // Ensure loaded image is in known format and size
        Graphics g = src.createGraphics();
        g.drawImage(img, (width - img.getWidth()) / 2, (height - img.getHeight()) / 2, null);
        g.dispose();

        JFrame win = new JFrame("Blur Demo");
        win.setDefaultCloseOperation(win.EXIT_ON_CLOSE);

        JPanel main = new JPanel();
        main.setLayout(new BorderLayout());

        JPanel controls = new JPanel();
        controls.setLayout(new GridBagLayout());

        GridBagConstraints c0 = new GridBagConstraints();
        c0.gridx = 0;
        c0.anchor = GridBagConstraints.BASELINE_LEADING;
        c0.ipadx = 3;
        c0.insets = new Insets(1, 2, 1, 2);

        controls.add(new JLabel("Width"), c0);
        controls.add(new JLabel("Height"), c0);

        GridBagConstraints c2 = (GridBagConstraints) c0.clone();
        c2.gridx = 2;
        controls.add(new JLabel("Angle"), c2);

        c0 = (GridBagConstraints) c0.clone();
        c0.gridx = 1;
        c0.weightx = 1;
        c0.fill = GridBagConstraints.HORIZONTAL;
        sizex = new JSlider(100, 5000, 1000);
        sizey = new JSlider(100, 5000, 100);
        controls.add(sizex, c0);
        controls.add(sizey, c0);

        c2 = (GridBagConstraints) c0.clone();
        c2.gridx = 3;
        angle = new JSlider(0, (int) (Math.PI * 1000));
        controls.add(angle, c2);

        sizex.addChangeListener(this);
        sizey.addChangeListener(this);
        angle.addChangeListener(this);

        JPanel buttons = new JPanel();
        controls.add(buttons, c2);
        JButton b;
        b = new JButton("Clear");
        buttons.add(b);
        b.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                doclear();
            }
        });
        ButtonGroup opt = new ButtonGroup();
        JToggleButton tb;
        blurButton = new JToggleButton("Blur");
        opt.add(blurButton);
        buttons.add(blurButton);
        blurButton.addActionListener(this);
        drawButton = new JToggleButton("Draw");
        opt.add(drawButton);
        buttons.add(drawButton);
        drawButton.addActionListener(this);

        JPanel imgs = new JPanel();
        imgs.setLayout(new BoxLayout(imgs, BoxLayout.X_AXIS));
        left = new PaintView(this, psf);
        right = new ImageView(dst);
        imgs.add(left);
        imgs.add(right);

        main.add(controls, BorderLayout.NORTH);
        main.add(imgs, BorderLayout.CENTER);
        win.getContentPane().add(main);

        win.pack();
        win.setVisible(true);

        // pre-load and transform src, since that wont change
        loadSource(src);

        blurButton.doClick();
    }

    public void stateChanged(ChangeEvent e) {
        if (drawButton.isSelected()) {
            recalc();
        } else {
            double w = sizex.getValue() / 100.0;
            double h = sizey.getValue() / 100.0;
            double a = angle.getValue() / 1000.0;

            Graphics g = psf.createGraphics();

            g.clearRect(0, 0, width, height);
            g.dispose();

            left.drawDot(w, h, a);
        }
    }

    public void actionPerformed(ActionEvent e) {
        stateChanged(null);
    }

    private void doclear() {
        Graphics g = psf.createGraphics();

        g.clearRect(0, 0, width, height);
        g.dispose();
        left.repaint();
        recalc();
    }

    private void dorecalc() {
        loadPSF(psf);

        // convolve each plane in freq domain
        convolve(aCBuffer, psfBuffer, aGBuffer);
        convolve(rCBuffer, psfBuffer, rGBuffer);
        convolve(gCBuffer, psfBuffer, gGBuffer);
        convolve(bCBuffer, psfBuffer, bGBuffer);

        // convert back to spatial domain
        fft.executeInterleaved(q, 1, CLFFTPlan.CLFFTDirection.Inverse, aGBuffer, aBuffer, null, null);
        fft.executeInterleaved(q, 1, CLFFTPlan.CLFFTDirection.Inverse, rGBuffer, rBuffer, null, null);
        fft.executeInterleaved(q, 1, CLFFTPlan.CLFFTDirection.Inverse, gGBuffer, gBuffer, null, null);
        fft.executeInterleaved(q, 1, CLFFTPlan.CLFFTDirection.Inverse, bGBuffer, bBuffer, null, null);

        // while gpu is running, calculate energy of psf
        float scale;

        long total = 0;
        DataBufferByte pd = (DataBufferByte) psf.getRaster().getDataBuffer();
        byte[] data = pd.getData();
        for (int i = 0; i < data.length; i++) {
            total += data[i] & 0xff;
        }
        scale = 255.0f / total / width / height;

        getDestination(argbBuffer, aBuffer, rBuffer, gBuffer, bBuffer, scale);

        // drop back to java, slow-crappy-method
        q.putReadBuffer(argbBuffer, true);
        DataBufferInt db = (DataBufferInt) dst.getRaster().getDataBuffer();
        argbBuffer.getBuffer().position(0);
        argbBuffer.getBuffer().get(db.getData());
        argbBuffer.getBuffer().position(0);
        right.repaint();
    }
    Runnable later;

    void recalc() {
        if (later == null) {
            later = new Runnable() {

                public void run() {
                    later = null;
                    dorecalc();
                }
            };
            SwingUtilities.invokeLater(later);
        }
    }
    CLContext cl;
    CLCommandQueue q;
    CLProgram prog;
    CLKernel kImg2Planes;
    CLKernel kPlanes2Img;
    CLKernel kGrey2Plane;
    CLKernel kConvolve;
    CLKernel kDeconvolve;
    CLFFTPlan fft;
    CLBuffer<IntBuffer> argbBuffer;
    CLBuffer<ByteBuffer> greyBuffer;
    CLBuffer<FloatBuffer> aBuffer;
    CLBuffer<FloatBuffer> rBuffer;
    CLBuffer<FloatBuffer> gBuffer;
    CLBuffer<FloatBuffer> bBuffer;
    CLBuffer<FloatBuffer> aCBuffer;
    CLBuffer<FloatBuffer> rCBuffer;
    CLBuffer<FloatBuffer> gCBuffer;
    CLBuffer<FloatBuffer> bCBuffer;
    CLBuffer<FloatBuffer> aGBuffer;
    CLBuffer<FloatBuffer> rGBuffer;
    CLBuffer<FloatBuffer> gGBuffer;
    CLBuffer<FloatBuffer> bGBuffer;
    CLBuffer<FloatBuffer> psfBuffer;
    CLBuffer<FloatBuffer> tmpBuffer;
    //
    CLKernel fft512;

    void initCL() throws InvalidContextException {
        
        // search a platform with a GPU
        CLPlatform[] platforms = CLPlatform.listCLPlatforms();
        CLDevice gpu = null;
        for (CLPlatform platform : platforms) {
            gpu = platform.getMaxFlopsDevice(CLDevice.Type.GPU);
            if(gpu != null) {
                break;
            }
        }

        cl = CLContext.create(gpu);

        q = cl.getDevices()[0].createCommandQueue();

        prog = cl.createProgram(img2Planes + planes2Img + convolve + grey2Plane + deconvolve);
        prog.build("-cl-mad-enable");

        kImg2Planes = prog.createCLKernel("img2planes");
        kPlanes2Img = prog.createCLKernel("planes2img");
        kGrey2Plane = prog.createCLKernel("grey2plane");
        kConvolve = prog.createCLKernel("convolve");
        kDeconvolve = prog.createCLKernel("deconvolve");

        argbBuffer = cl.createIntBuffer(width * height, Mem.READ_WRITE);
        greyBuffer = cl.createByteBuffer(width * height, Mem.READ_WRITE);
        aBuffer = cl.createFloatBuffer(width * height * 2, Mem.READ_WRITE);
        rBuffer = cl.createFloatBuffer(width * height * 2, Mem.READ_WRITE);
        gBuffer = cl.createFloatBuffer(width * height * 2, Mem.READ_WRITE);
        bBuffer = cl.createFloatBuffer(width * height * 2, Mem.READ_WRITE);
        psfBuffer = cl.createFloatBuffer(width * height * 2, Mem.READ_WRITE);
        tmpBuffer = cl.createFloatBuffer(width * height * 2, Mem.READ_WRITE);

        aCBuffer = cl.createFloatBuffer(width * height * 2, Mem.READ_WRITE);
        rCBuffer = cl.createFloatBuffer(width * height * 2, Mem.READ_WRITE);
        gCBuffer = cl.createFloatBuffer(width * height * 2, Mem.READ_WRITE);
        bCBuffer = cl.createFloatBuffer(width * height * 2, Mem.READ_WRITE);

        aGBuffer = cl.createFloatBuffer(width * height * 2, Mem.READ_WRITE);
        rGBuffer = cl.createFloatBuffer(width * height * 2, Mem.READ_WRITE);
        gGBuffer = cl.createFloatBuffer(width * height * 2, Mem.READ_WRITE);
        bGBuffer = cl.createFloatBuffer(width * height * 2, Mem.READ_WRITE);
        if (false) {
            try {
                CLProgram p = cl.createProgram(new FileInputStream("/home/notzed/cl/fft-512.cl"));
                p.build();
                fft512 = p.createCLKernel("fft0");
            } catch (IOException ex) {
                Logger.getLogger(BlurTest.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            fft = new CLFFTPlan(cl, new int[]{width, height}, CLFFTPlan.CLFFTDataFormat.InterleavedComplexFormat);
        }
        //fft.dumpPlan(null);
    }

    void loadSource(BufferedImage src) {
        DataBufferInt sb = (DataBufferInt) src.getRaster().getDataBuffer();

        argbBuffer.getBuffer().position(0);
        argbBuffer.getBuffer().put(sb.getData());
        argbBuffer.getBuffer().position(0);
        q.putWriteBuffer(argbBuffer, false);

        kImg2Planes.setArg(0, argbBuffer);
        kImg2Planes.setArg(1, 0);
        kImg2Planes.setArg(2, width);
        kImg2Planes.setArg(3, aBuffer);
        kImg2Planes.setArg(4, rBuffer);
        kImg2Planes.setArg(5, gBuffer);
        kImg2Planes.setArg(6, bBuffer);
        kImg2Planes.setArg(7, 0);
        kImg2Planes.setArg(8, width);
        q.put2DRangeKernel(kImg2Planes, 0, 0, width, height, 64, 1);
        q.finish();

        fft.executeInterleaved(q, 1, CLFFTPlan.CLFFTDirection.Forward, aBuffer, aCBuffer, null, null);
        fft.executeInterleaved(q, 1, CLFFTPlan.CLFFTDirection.Forward, rBuffer, rCBuffer, null, null);
        fft.executeInterleaved(q, 1, CLFFTPlan.CLFFTDirection.Forward, gBuffer, gCBuffer, null, null);
        fft.executeInterleaved(q, 1, CLFFTPlan.CLFFTDirection.Forward, bBuffer, bCBuffer, null, null);
    }

    void loadPSF(BufferedImage psf) {
        assert (psf.getType() == BufferedImage.TYPE_BYTE_GRAY);
        DataBufferByte pb = (DataBufferByte) psf.getRaster().getDataBuffer();

        greyBuffer.getBuffer().position(0);
        greyBuffer.getBuffer().put(pb.getData());
        greyBuffer.getBuffer().position(0);
        q.putWriteBuffer(greyBuffer, false);

        kGrey2Plane.setArg(0, greyBuffer);
        kGrey2Plane.setArg(1, 0);
        kGrey2Plane.setArg(2, width);
        kGrey2Plane.setArg(3, tmpBuffer);
        kGrey2Plane.setArg(4, 0);
        kGrey2Plane.setArg(5, width);
        q.put2DRangeKernel(kGrey2Plane, 0, 0, width, height, 64, 1);

        if (true) {
            fft.executeInterleaved(q, 1, CLFFTPlan.CLFFTDirection.Forward, tmpBuffer, psfBuffer, null, null);
        } else if (true) {
            fft512.setArg(0, tmpBuffer);
            fft512.setArg(1, psfBuffer);
            fft512.setArg(2, -1);
            fft512.setArg(3, height);
            //q.put1DRangeKernel(fft512, 0,height*64, 64);
            q.put2DRangeKernel(fft512, 0, 0, height * 64, 1, 64, 1);
            System.out.println("running kernel " + 64 * height + ", " + 64);
        }
    }

    // g = f x h
    void convolve(CLBuffer<FloatBuffer> h, CLBuffer<FloatBuffer> f, CLBuffer<FloatBuffer> g) {
        kConvolve.setArg(0, h);
        kConvolve.setArg(1, f);
        kConvolve.setArg(2, g);
        kConvolve.setArg(3, width);
        q.put2DRangeKernel(kConvolve, 0, 0, width, height, 64, 1);
    }

    // g = h*conj(f) / (abs(f)^2 + k)
    void deconvolve(CLBuffer<FloatBuffer> h, CLBuffer<FloatBuffer> f, CLBuffer<FloatBuffer> g, float k) {
        kDeconvolve.setArg(0, h);
        kDeconvolve.setArg(1, f);
        kDeconvolve.setArg(2, g);
        kDeconvolve.setArg(3, width);
        kDeconvolve.setArg(4, k);
        q.put2DRangeKernel(kDeconvolve, 0, 0, width, height, 64, 1);
    }

    void getDestination(CLBuffer<IntBuffer> dst, CLBuffer<FloatBuffer> a, CLBuffer<FloatBuffer> r, CLBuffer<FloatBuffer> g, CLBuffer<FloatBuffer> b, float scale) {
        kPlanes2Img.setArg(0, dst);
        kPlanes2Img.setArg(1, 0);
        kPlanes2Img.setArg(2, width);
        kPlanes2Img.setArg(3, a);
        kPlanes2Img.setArg(4, r);
        kPlanes2Img.setArg(5, g);
        kPlanes2Img.setArg(6, b);
        kPlanes2Img.setArg(7, 0);
        kPlanes2Img.setArg(8, width);
        kPlanes2Img.setArg(9, scale);
        q.put2DRangeKernel(kPlanes2Img, 0, 0, width, height, 64, 1);
    }
    // Convert packed ARGB byte image to planes of complex floats
    final String img2Planes =
              "kernel void img2planes(global const uchar4 *argb, int soff, int sstride,"
            + "  global float2 *a, global float2 *r, global float2 *g, global float2 *b, int doff, int dstride) {"
            + " int gx = get_global_id(0);"
            + " int gy = get_global_id(1);"
            + " uchar4 v = argb[soff+sstride*gy+gx];"
            + " float4 ff = convert_float4(v) * (float4)(1.0f/255);"
            + " doff += (dstride * gy + gx);"
            + " b[doff] = (float2){ ff.s0, 0 };\n"
            + " g[doff] = (float2){ ff.s1, 0 };"
            + " r[doff] = (float2){ ff.s2, 0 };"
            + " a[doff] = (float2){ ff.s3, 0 };\n"
            + "}\n\n";
    // not the best implementation
    // this also performs an 'fftshift'
    final String grey2Plane =
              "kernel void grey2plane(global const uchar *src, int soff, int sstride,"
            + "  global float2 *dst, int doff, int dstride) {"
            + " int gx = get_global_id(0);"
            + " int gy = get_global_id(1);"
            + " uchar v = src[soff+sstride*gy+gx];"
            + " float ff = convert_float(v) * (1.0f/255);"
            // fftshift
            + " gx ^= get_global_size(0)>>1;"
            + " gy ^= get_global_size(1)>>1;"
            + " doff += (dstride * gy + gx);"
            + " dst[doff] = (float2) { ff, 0 };"
            + "}\n\n";
    // This also does the 'fftscale' after the inverse fft.
    final String planes2Img =
              "kernel void planes2img(global uchar4 *argb, int soff, int sstride, const global float2 *a, const global float2 *r, const global float2 *g, const global float2 *b, int doff, int dstride, float scale) {"
            + " int gx = get_global_id(0);"
            + " int gy = get_global_id(1);"
            + " float4 fr, fi, fa;"
            + " float2 t;"
            + " doff += (dstride * gy + gx);"
            + " float2 s = (float2)scale;"
            + " t = b[doff]*s; fr.s0 = t.s0; fi.s0 = t.s1;"
            + " t = g[doff]*s; fr.s1 = t.s0; fi.s1 = t.s1;"
            + " t = r[doff]*s; fr.s2 = t.s0; fi.s2 = t.s1;"
            + " t = a[doff]*s; fr.s3 = t.s0; fi.s3 = t.s1;"
            + " fa = sqrt(fr*fr + fi*fi) * 255.0f;"
            + " fa = clamp(fa, 0.0f, 255.0f);"
            + " argb[soff +sstride*gy+gx] = convert_uchar4(fa);"
            + "}\n\n";
    final String convolve =
              "kernel void convolve(global const float2 *h, global const float2 *ff, global float2 *g, int stride) {"
            + " int gx = get_global_id(0);"
            + " int gy = get_global_id(1);"
            + " int off = stride * gy + gx;"
            + " float2 a = h[off];"
            + " float2 b = ff[off];"
            + " g[off] = (float2) { a.s0 * b.s0 - a.s1 * b.s1, a.s0 * b.s1 + a.s1 * b.s0 };"
            + "}\n\n";
    final String deconvolve =
              "kernel void deconvolve(global const float2 *h, global const float2 *ff, global float2 *g, int stride, float k) {"
            + " int gx = get_global_id(0);"
            + " int gy = get_global_id(1);"
            + " int off = stride * gy + gx;"
            + " float2 a = h[off];"
            + " float2 b = ff[off];"
            + " float d = b.s0 * b.s0 + b.s1 * b.s1 + k;"
            + " b.s0 /= d;"
            + " b.s1 /= -d;"
            + " g[off] = (float2) { a.s0 * b.s0 - a.s1 * b.s1, a.s0 * b.s1 + a.s1 * b.s0 };"
            + "}\n\n";
}
