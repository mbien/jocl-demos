package com.jogamp.opencl.demos.fft;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import javax.swing.JComponent;

/**
 * Just draws an image.
 * @author notzed
 */
class ImageView extends JComponent {

    BufferedImage img;

    public ImageView(BufferedImage img) {
        this.img = img;
        this.setPreferredSize(new Dimension(img.getWidth(), img.getHeight()));
    }

    @Override
    protected void paintComponent(Graphics g) {
        g.drawImage(img, 0, 0, null);
    }
}
