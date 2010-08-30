package com.jogamp.opencl.demos.fft;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Paint;
import java.awt.RadialGradientPaint;
import java.awt.Rectangle;
import java.awt.Shape;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;

/**
 * Draws an image and lets you draw white dots in it with the mouse.  Or big white dots with code.
 * @author notzed
 */
class PaintView extends ImageView implements MouseListener, MouseMotionListener {

	Graphics2D imgg;
	Paint paint;
	Shape brush;
	BlurTest win;

	public PaintView(BlurTest win, BufferedImage img) {
		super(img);

		this.win = win;

		paint = new RadialGradientPaint(new Point2D.Float(0, 0), 3,
				new float[]{0.0f, 1.0f}, new Color[]{new Color(255, 255, 255, 255), new Color(255, 255, 255, 0)});
		brush = new java.awt.geom.Ellipse2D.Float(-5, -5, 11, 11);

		imgg = img.createGraphics();

		this.addMouseListener(this);
	}

	void drawPaint(double x, double y) {
		Graphics2D gg = (Graphics2D) imgg.create();

		gg.translate(x, y);
		gg.setPaint(paint);
		gg.fill(brush);

		gg.dispose();

		repaint(new Rectangle((int) (x - 6), (int) (y - 6), 12, 12));
		// close your eyes if you're under-age ...
		win.recalc();
	}

	public void drawDot(double width, double height, double angle) {
		Graphics2D gg = (Graphics2D) imgg.create();

		gg.setPaint(paint);
		gg.translate(img.getWidth() / 2, img.getHeight() / 2);
		gg.rotate(angle);
		gg.scale(width, height);
		gg.fill(brush);

		gg.dispose();

		repaint();
		win.recalc();
	}

	public void mouseClicked(MouseEvent e) {
	}

	public void mousePressed(MouseEvent e) {
		if (e.getButton() == e.BUTTON1) {
			addMouseMotionListener(this);
			drawPaint(e.getX(), e.getY());
		}
	}

	public void mouseReleased(MouseEvent e) {
		if (e.getButton() == e.BUTTON1) {
			removeMouseMotionListener(this);
			//drawPaint(e.getX(), e.getY());
		}
	}

	public void mouseEntered(MouseEvent e) {
	}

	public void mouseExited(MouseEvent e) {
	}

	public void mouseDragged(MouseEvent e) {
		drawPaint(e.getX(), e.getY());
	}

	public void mouseMoved(MouseEvent e) {
	}
}
